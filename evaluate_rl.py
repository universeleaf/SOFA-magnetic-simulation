# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from stable_baselines3 import PPO
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'evaluate_rl.py requires stable_baselines3 and torch. Install them before loading PPO models.'
    ) from exc

from guidewire_scene.rl_env import GuidewireEnv
from guidewire_scene.train_rl import CHECKPOINT_PATH, OUTPUT_ROOT


EVAL_DIR = OUTPUT_ROOT / 'evaluation'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO guidewire policy headlessly.')
    parser.add_argument('--model', default=str(CHECKPOINT_PATH), help='Path to the PPO zip checkpoint.')
    parser.add_argument('--episodes', type=int, default=1, help='Number of rollout episodes.')
    parser.add_argument('--sim-steps-per-action', type=int, default=5, help='Number of SOFA animate() calls per PPO action.')
    parser.add_argument('--max-episode-steps', type=int, default=400, help='Maximum environment steps per episode.')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic actions instead of deterministic rollout.')
    parser.add_argument('--save-trace', action='store_true', help='Write per-step rollout traces to CSV.')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f'PPO checkpoint not found: {model_path}')

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = EVAL_DIR / 'policy_rollout_summary.csv'
    trace_csv = EVAL_DIR / 'policy_rollout_trace.csv'

    env = GuidewireEnv(
        sim_steps_per_action=int(args.sim_steps_per_action),
        max_episode_steps=int(args.max_episode_steps),
    )
    model = PPO.load(str(model_path), device='auto')

    summary_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    try:
        for episode_idx in range(int(args.episodes)):
            obs, info = env.reset()
            done = False
            episode_reward = 0.0
            step_idx = 0
            while not done:
                action, _ = model.predict(obs, deterministic=not bool(args.stochastic))
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = bool(terminated or truncated)
                episode_reward += float(reward)
                step_idx += 1
                if bool(args.save_trace):
                    trace_rows.append(
                        {
                            'episode': episode_idx,
                            'step': step_idx,
                            'reward': float(reward),
                            'distance_to_goal_mm': float(step_info.get('distance_to_goal_mm', 0.0)),
                            'local_target_distance_mm': float(step_info.get('local_target_distance_mm', 0.0)),
                            'wall_clearance_mm': float(step_info.get('wall_clearance_mm', 0.0)),
                            'contact_penetration_mm': float(step_info.get('contact_penetration_mm', 0.0)),
                            'tip_progress_mm': float(step_info.get('tip_progress_mm', 0.0)),
                            'tip_centerline_offset_mm': float(step_info.get('tip_centerline_offset_mm', 0.0)),
                            'centerline_alignment_cos': float(step_info.get('centerline_alignment_cos', 0.0)),
                            'wall_contact_force_n': float(step_info.get('wall_contact_force_n', 0.0)),
                            'contact_load_n': float(step_info.get('contact_load_n', 0.0)),
                            'barrier_force_n': float(step_info.get('barrier_force_n', 0.0)),
                            'drive_reaction_n': float(step_info.get('drive_reaction_n', 0.0)),
                            'magnetic_force_n': float(step_info.get('magnetic_force_n', 0.0)),
                            'magnetic_torque_nm': float(step_info.get('magnetic_torque_nm', 0.0)),
                            'success': bool(step_info.get('success', False)),
                            'invalid': bool(step_info.get('invalid', False)),
                        }
                    )

            summary_rows.append(
                {
                    'episode': episode_idx,
                    'episode_reward': float(episode_reward),
                    'episode_steps': int(step_idx),
                    'distance_to_goal_mm': float(step_info.get('distance_to_goal_mm', 0.0)),
                    'wall_clearance_mm': float(step_info.get('wall_clearance_mm', 0.0)),
                    'tip_centerline_offset_mm': float(step_info.get('tip_centerline_offset_mm', 0.0)),
                    'wall_contact_force_n': float(step_info.get('wall_contact_force_n', 0.0)),
                    'magnetic_force_n': float(step_info.get('magnetic_force_n', 0.0)),
                    'magnetic_torque_nm': float(step_info.get('magnetic_torque_nm', 0.0)),
                    'success': bool(step_info.get('success', False)),
                    'invalid': bool(step_info.get('invalid', False)),
                }
            )
            print(
                '[RL-EVAL] '
                f'episode={episode_idx} '
                f'reward={episode_reward:.3f} '
                f'steps={step_idx} '
                f'success={step_info.get("success", False)} '
                f'dist={float(step_info.get("distance_to_goal_mm", 0.0)):.3f} mm '
                f'clearance={float(step_info.get("wall_clearance_mm", 0.0)):.3f} mm '
                f'offset={float(step_info.get("tip_centerline_offset_mm", 0.0)):.3f} mm'
            )
    finally:
        env.close()

    if summary_rows:
        with summary_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    if trace_rows and bool(args.save_trace):
        with trace_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trace_rows)

    print(f'[RL-EVAL] Summary CSV: {summary_csv}')
    if trace_rows and bool(args.save_trace):
        print(f'[RL-EVAL] Trace CSV: {trace_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
