# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'GuidewireEnv requires gymnasium. Install it before running RL training.'
    ) from exc

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guidewire_scene.config import GUIDEWIRE_BACKEND
from guidewire_scene.runtime import ensure_sofa
from guidewire_scene.scene import createScene, get_rl_handles

Sofa = ensure_sofa()
import Sofa.Simulation  # noqa: E402


class GuidewireEnv(gym.Env):
    """
    Gymnasium environment for SOFA guidewire navigation.

    The environment keeps PPO outside the scene logic and only relies on a small
    RL-facing API exposed by the Python controller.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        *,
        sim_steps_per_action: int = 5,
        max_episode_steps: int = 400,
        success_threshold_mm: float = 6.0,
        contact_safe_clearance_mm: float = 0.35,
        termination_penetration_mm: float = 2.0,
        target_point_mm: np.ndarray | list[float] | tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__()

        self.sim_steps_per_action = int(max(sim_steps_per_action, 1))
        self.max_episode_steps = int(max(max_episode_steps, 1))
        self.success_threshold_mm = float(max(success_threshold_mm, 0.5))
        self.contact_safe_clearance_mm = float(max(contact_safe_clearance_mm, 0.0))
        self.termination_penetration_mm = float(max(termination_penetration_mm, 0.0))
        self._requested_target_point = (
            None
            if target_point_mm is None
            else np.asarray(target_point_mm, dtype=float).reshape(3).copy()
        )
        self._clearance_obs_cap_mm = 25.0
        self._penetration_obs_cap_mm = 25.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        obs_low = np.array(
            [
                -1.0e4, -1.0e4, -1.0e4,
                -1.0, -1.0, -1.0,
                -1.0e4, -1.0e4, -1.0e4,
                -1.0e4, -1.0e4, -1.0e4,
                0.0,
                0.0,
                -self._clearance_obs_cap_mm,
                0.0,
                -1.0, -1.0, -1.0,
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                1.0e4, 1.0e4, 1.0e4,
                1.0, 1.0, 1.0,
                1.0e4, 1.0e4, 1.0e4,
                1.0e4, 1.0e4, 1.0e4,
                1.0e4,
                1.0e4,
                self._clearance_obs_cap_mm,
                self._penetration_obs_cap_mm,
                1.0, 1.0, 1.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.root = None
        self.controller = None
        self.centerline_points = np.zeros((0, 3), dtype=float)
        self.target_point = np.zeros(3, dtype=float)
        self.scene_dt = 0.0
        self.episode_step = 0
        self.initial_distance_mm = 1.0
        self.prev_distance_mm = 1.0
        self.prev_tip_progress_mm = 0.0
        self.last_action = np.zeros(3, dtype=float)

    def _build_scene(self) -> None:
        self.root = Sofa.Core.Node('root')
        createScene(self.root)
        Sofa.Simulation.init(self.root)

        handles = get_rl_handles(self.root)
        self.controller = handles['controller']
        self.centerline_points = np.asarray(handles['centerline_points'], dtype=float)
        self.target_point = (
            self._requested_target_point.copy()
            if self._requested_target_point is not None
            else np.asarray(handles['target_point'], dtype=float).reshape(3).copy()
        )
        self.scene_dt = float(handles['dt'])

        if not hasattr(self.controller, 'enable_rl_control') or not hasattr(self.controller, 'get_rl_state'):
            raise RuntimeError('Guidewire controller does not expose the required RL bridge methods.')
        self.controller.enable_rl_control(
            target_point=self.target_point,
            success_threshold_mm=self.success_threshold_mm,
        )

    def _read_state(self) -> dict[str, Any]:
        if self.controller is None:
            raise RuntimeError('Environment has not been reset. Call reset() before step().')
        state = self.controller.get_rl_state()
        return {
            key: (np.asarray(value, dtype=float).copy() if isinstance(value, np.ndarray) else value)
            for key, value in state.items()
        }

    def _obs_clearance(self, clearance_mm: float) -> float:
        if not np.isfinite(clearance_mm):
            return self._clearance_obs_cap_mm
        return float(np.clip(clearance_mm, -self._clearance_obs_cap_mm, self._clearance_obs_cap_mm))

    def _build_observation(self, state: dict[str, Any]) -> np.ndarray:
        observation = np.concatenate(
            [
                np.asarray(state['tip_position_mm'], dtype=np.float32).reshape(3),
                np.asarray(state['tip_direction'], dtype=np.float32).reshape(3),
                np.asarray(state['target_position_mm'], dtype=np.float32).reshape(3),
                np.asarray(state['target_delta_mm'], dtype=np.float32).reshape(3),
                np.asarray(
                    [
                        float(state['distance_to_goal_mm']),
                        float(state['local_target_distance_mm']),
                        self._obs_clearance(float(state['wall_clearance_mm'])),
                        float(np.clip(state['contact_penetration_mm'], 0.0, self._penetration_obs_cap_mm)),
                    ],
                    dtype=np.float32,
                ),
                np.asarray(state['rl_action'], dtype=np.float32).reshape(3),
            ]
        )
        return observation.astype(np.float32, copy=False)

    def _compute_reward(
        self,
        state: dict[str, Any],
        action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        distance_mm = float(state['distance_to_goal_mm'])
        prev_distance_mm = float(self.prev_distance_mm)
        delta_distance_mm = prev_distance_mm - distance_mm
        tip_progress_mm = float(state['tip_progress_mm'])
        delta_tip_progress_mm = tip_progress_mm - float(self.prev_tip_progress_mm)
        local_target_distance_mm = float(state['local_target_distance_mm'])

        clearance_mm = float(state['wall_clearance_mm'])
        penetration_mm = float(max(state['contact_penetration_mm'], 0.0))
        near_wall_mm = (
            max(self.contact_safe_clearance_mm - clearance_mm, 0.0)
            if np.isfinite(clearance_mm)
            else 0.0
        )

        progress_reward = 2.5 * delta_distance_mm
        path_progress_reward = 0.25 * delta_tip_progress_mm
        distance_penalty = 0.0015 * distance_mm
        local_target_penalty = 0.01 * local_target_distance_mm
        contact_penalty = 0.25 if bool(state['wall_contact_active']) else 0.0
        near_wall_penalty = 0.4 * near_wall_mm
        penetration_penalty = 6.0 * penetration_mm
        compression_penalty = 0.03 * float(max(state['beam_compression_mm'], 0.0))
        action_penalty = 0.01 * float(np.dot(action, action))
        delta_action = action - self.last_action
        smoothness_penalty = 0.03 * float(np.dot(delta_action, delta_action))
        stall_penalty = 0.5 if bool(state['beam_stall_active']) else 0.0
        time_penalty = 0.02
        success_bonus = 120.0 if bool(state['success']) else 0.0
        invalid_penalty = 25.0 if bool(state['invalid']) else 0.0

        reward = (
            progress_reward
            + path_progress_reward
            - distance_penalty
            - local_target_penalty
            - contact_penalty
            - near_wall_penalty
            - penetration_penalty
            - compression_penalty
            - action_penalty
            - smoothness_penalty
            - stall_penalty
            - time_penalty
            + success_bonus
            - invalid_penalty
        )
        breakdown = {
            'progress_reward': float(progress_reward),
            'path_progress_reward': float(path_progress_reward),
            'distance_penalty': float(distance_penalty),
            'local_target_penalty': float(local_target_penalty),
            'contact_penalty': float(contact_penalty),
            'near_wall_penalty': float(near_wall_penalty),
            'penetration_penalty': float(penetration_penalty),
            'compression_penalty': float(compression_penalty),
            'action_penalty': float(action_penalty),
            'smoothness_penalty': float(smoothness_penalty),
            'stall_penalty': float(stall_penalty),
            'time_penalty': float(time_penalty),
            'success_bonus': float(success_bonus),
            'invalid_penalty': float(invalid_penalty),
            'delta_distance_mm': float(delta_distance_mm),
            'delta_tip_progress_mm': float(delta_tip_progress_mm),
        }
        return float(reward), breakdown

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.close()

        if options is not None and 'target_point_mm' in options and options['target_point_mm'] is not None:
            self._requested_target_point = np.asarray(options['target_point_mm'], dtype=float).reshape(3).copy()

        self._build_scene()
        self.episode_step = 0
        self.last_action = np.zeros(3, dtype=float)

        state = self._read_state()
        self.initial_distance_mm = max(float(state['distance_to_goal_mm']), 1.0)
        self.prev_distance_mm = float(state['distance_to_goal_mm'])
        self.prev_tip_progress_mm = float(state['tip_progress_mm'])
        observation = self._build_observation(state)
        info = {
            'backend': GUIDEWIRE_BACKEND,
            'distance_to_goal_mm': float(state['distance_to_goal_mm']),
            'wall_clearance_mm': float(state['wall_clearance_mm']),
            'target_point_mm': np.asarray(state['target_position_mm'], dtype=float).copy(),
            'goal_point_mm': np.asarray(state['goal_position_mm'], dtype=float).copy(),
        }
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.root is None or self.controller is None:
            raise RuntimeError('Environment has not been reset. Call reset() before step().')

        clipped_action = np.clip(np.asarray(action, dtype=float).reshape(3), -1.0, 1.0)
        self.controller.set_rl_action(clipped_action)
        for _ in range(self.sim_steps_per_action):
            Sofa.Simulation.animate(self.root, self.scene_dt)

        state = self._read_state()
        reward, reward_breakdown = self._compute_reward(state, clipped_action)

        terminated = bool(state['success']) or bool(state['invalid'])
        if float(state['contact_penetration_mm']) >= self.termination_penetration_mm:
            terminated = True
            reward -= 20.0
            reward_breakdown['hard_penetration_penalty'] = 20.0
        else:
            reward_breakdown['hard_penetration_penalty'] = 0.0

        self.episode_step += 1
        truncated = self.episode_step >= self.max_episode_steps

        self.prev_distance_mm = float(state['distance_to_goal_mm'])
        self.prev_tip_progress_mm = float(state['tip_progress_mm'])
        self.last_action = clipped_action.copy()
        observation = self._build_observation(state)
        info = {
            'backend': GUIDEWIRE_BACKEND,
            'distance_to_goal_mm': float(state['distance_to_goal_mm']),
            'local_target_distance_mm': float(state['local_target_distance_mm']),
            'wall_clearance_mm': float(state['wall_clearance_mm']),
            'contact_penetration_mm': float(state['contact_penetration_mm']),
            'tip_progress_mm': float(state['tip_progress_mm']),
            'estimated_push_mm': float(state['estimated_push_mm']),
            'commanded_push_mm': float(state['commanded_push_mm']),
            'beam_compression_mm': float(state['beam_compression_mm']),
            'beam_stall_active': bool(state['beam_stall_active']),
            'wall_contact_active': bool(state['wall_contact_active']),
            'target_point_mm': np.asarray(state['target_position_mm'], dtype=float).copy(),
            'goal_point_mm': np.asarray(state['goal_position_mm'], dtype=float).copy(),
            'reward_breakdown': reward_breakdown,
        }
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self.root is not None:
            try:
                unload = getattr(Sofa.Simulation, 'unload', None)
                if callable(unload):
                    unload(self.root)
            except Exception:
                pass
        self.root = None
        self.controller = None
        self.centerline_points = np.zeros((0, 3), dtype=float)
        self.target_point = np.zeros(3, dtype=float)
        self.scene_dt = 0.0
        self.prev_tip_progress_mm = 0.0
