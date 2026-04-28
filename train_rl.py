# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'train_rl.py requires stable_baselines3 and torch. Install them before launching PPO training.'
    ) from exc

from guidewire_scene.rl_env import GuidewireEnv

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(r'E:\Tongyu\guidewire_rl')
CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
TENSORBOARD_LOG_DIR = OUTPUT_ROOT / 'tensorboard'
METRICS_DIR = OUTPUT_ROOT / 'metrics'
MONITOR_DIR = METRICS_DIR / 'monitors'
CHECKPOINT_PATH = CHECKPOINT_DIR / 'ppo_guidewire_latest.zip'
MONITOR_CSV_PATH = MONITOR_DIR / 'ppo_guidewire_monitor_env00.csv'
EPISODE_CSV_PATH = METRICS_DIR / 'episode_metrics.csv'
STATUS_JSON_PATH = METRICS_DIR / 'latest_status.json'


def ensure_output_dirs() -> None:
    for path in (OUTPUT_ROOT, CHECKPOINT_DIR, TENSORBOARD_LOG_DIR, METRICS_DIR, MONITOR_DIR):
        path.mkdir(parents=True, exist_ok=True)


class TrainingStatusCallback(BaseCallback):
    def __init__(
        self,
        *,
        total_timesteps: int,
        num_envs: int,
        save_freq: int,
        save_path: Path,
        episode_csv_path: Path,
        status_json_path: Path,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.total_timesteps = int(max(total_timesteps, 1))
        self.num_envs = int(max(num_envs, 1))
        self.save_freq = int(max(save_freq, 1))
        self.save_path = Path(save_path)
        self.episode_csv_path = Path(episode_csv_path)
        self.status_json_path = Path(status_json_path)
        self._last_saved_step = 0
        self._episode_index = 0
        self._episode_csv_file = None
        self._episode_writer = None
        self._episode_metrics_by_env = {
            idx: self._new_episode_metrics()
            for idx in range(self.num_envs)
        }
        self._run_start_timesteps = 0
        self._progress_value = 0
        self._progress_bar = None
        self._run_start_time = 0.0
        self._last_status_write_time = 0.0
        self._last_status_write_completed = -1
        self._latest_episode_row: dict[str, Any] = {}

    def _new_episode_metrics(self) -> dict[str, Any]:
        return {
            'steps': 0,
            'reward_sum': 0.0,
            'wall_contact_steps': 0,
            'min_wall_clearance_mm': float('inf'),
            'max_contact_penetration_mm': 0.0,
            'centerline_offset_sum': 0.0,
            'centerline_offset_max_mm': 0.0,
            'wall_force_sum': 0.0,
            'wall_force_max_n': 0.0,
            'contact_load_sum': 0.0,
            'contact_load_max_n': 0.0,
            'magnetic_force_sum': 0.0,
            'magnetic_force_max_n': 0.0,
            'magnetic_torque_sum': 0.0,
            'magnetic_torque_max_nm': 0.0,
            'last_info': {},
        }

    def _open_writer(self) -> None:
        self.episode_csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.episode_csv_path.exists()
        self._episode_csv_file = self.episode_csv_path.open('a', newline='', encoding='utf-8')
        fieldnames = [
            'episode',
            'env_index',
            'timesteps',
            'episode_reward',
            'episode_length',
            'success',
            'invalid',
            'distance_to_goal_mm',
            'min_wall_clearance_mm',
            'max_contact_penetration_mm',
            'mean_centerline_offset_mm',
            'max_centerline_offset_mm',
            'mean_wall_contact_force_n',
            'max_wall_contact_force_n',
            'mean_contact_load_n',
            'max_contact_load_n',
            'mean_magnetic_force_n',
            'max_magnetic_force_n',
            'mean_magnetic_torque_nm',
            'max_magnetic_torque_nm',
            'wall_contact_steps',
            'reset_mode',
            'target_point_mm',
            'goal_point_mm',
        ]
        self._episode_writer = csv.DictWriter(self._episode_csv_file, fieldnames=fieldnames)
        if not file_exists:
            self._episode_writer.writeheader()
            self._episode_csv_file.flush()

    def _save_latest(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.save_path))
        self._last_saved_step = int(self.num_timesteps)
        if self.verbose:
            self._log_line(f'[RL] Checkpoint saved: {self.save_path}')

    def save_latest(self) -> None:
        self._save_latest()

    def close_progress(self) -> None:
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None

    def _write_status_json(self, payload: dict[str, Any]) -> None:
        self.status_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _status_payload(self) -> dict[str, Any]:
        return {
            'timesteps': int(self.num_timesteps),
            'episode': int(self._episode_index),
            'run_timesteps_completed': int(self._run_timesteps_completed()),
            'run_total_timesteps': int(self.total_timesteps),
            'run_progress_percent': float(self._run_progress_percent()),
            'num_envs': int(self.num_envs),
            'checkpoint_path': str(self.save_path),
            'tensorboard_log_dir': str(TENSORBOARD_LOG_DIR),
            'monitor_csv_path': str(MONITOR_CSV_PATH),
            'monitor_dir': str(MONITOR_DIR),
            'episode_csv_path': str(self.episode_csv_path),
            'latest_episode': self._latest_episode_row,
        }

    def _maybe_write_live_status(self, force: bool = False) -> None:
        completed = int(self._run_timesteps_completed())
        now = time.time()
        should_write = force
        if not should_write and completed != self._last_status_write_completed:
            if (now - self._last_status_write_time) >= 1.0:
                should_write = True
        if should_write:
            self._write_status_json(self._status_payload())
            self._last_status_write_time = now
            self._last_status_write_completed = completed

    def _log_line(self, message: str) -> None:
        if self._progress_bar is not None:
            self._progress_bar.write(message)
        else:
            print(message)

    def _run_timesteps_completed(self) -> int:
        return max(int(self.num_timesteps) - int(self._run_start_timesteps), 0)

    def _run_progress_percent(self) -> float:
        completed = min(self._run_timesteps_completed(), self.total_timesteps)
        return 100.0 * float(completed) / float(self.total_timesteps)

    def _sync_progress(self, force: bool = False) -> None:
        completed = min(self._run_timesteps_completed(), self.total_timesteps)
        delta = completed - self._progress_value
        if self._progress_bar is not None and delta > 0:
            self._progress_bar.update(delta)
            self._progress_value = completed

        if self._progress_bar is not None and (delta > 0 or force):
            elapsed = max(time.time() - self._run_start_time, 1.0e-6)
            step_rate = float(completed) / elapsed if completed > 0 else 0.0
            remaining = max(self.total_timesteps - completed, 0)
            eta_seconds = int(remaining / step_rate) if step_rate > 0 else -1
            self._progress_bar.set_postfix(
                episode=int(self._episode_index),
                percent=f'{self._run_progress_percent():.1f}%',
                step_rate=f'{step_rate:.1f} ts/s',
                eta=f'{eta_seconds if eta_seconds >= 0 else "?"} s',
            )
            if force:
                self._progress_bar.refresh()
        self._maybe_write_live_status(force=force)

    def _on_training_start(self) -> None:
        ensure_output_dirs()
        self._open_writer()
        self.num_envs = int(max(getattr(self.training_env, 'num_envs', self.num_envs), 1))
        self._episode_metrics_by_env = {
            idx: self._new_episode_metrics()
            for idx in range(self.num_envs)
        }
        self._run_start_timesteps = int(getattr(self.model, 'num_timesteps', 0))
        self._last_saved_step = int(self._run_start_timesteps)
        self._run_start_time = time.time()
        self._progress_value = 0
        self._latest_episode_row = {}
        self._last_status_write_time = 0.0
        self._last_status_write_completed = -1
        if tqdm is not None:
            self._progress_bar = tqdm(
                total=self.total_timesteps,
                desc='[RL] Training',
                unit='ts',
                dynamic_ncols=True,
                mininterval=1.0,
            )
        self._sync_progress(force=True)

    def _metrics_for_env(self, env_index: int) -> dict[str, Any]:
        if env_index not in self._episode_metrics_by_env:
            self._episode_metrics_by_env[env_index] = self._new_episode_metrics()
        return self._episode_metrics_by_env[env_index]

    def _update_episode_metrics(self, env_index: int, info: dict[str, Any], reward: float) -> None:
        metrics = self._metrics_for_env(env_index)
        metrics['steps'] += 1
        metrics['reward_sum'] += float(reward)
        if bool(info.get('wall_contact_active', False)):
            metrics['wall_contact_steps'] += 1
        wall_clearance = float(info.get('wall_clearance_mm', float('inf')))
        if wall_clearance < metrics['min_wall_clearance_mm']:
            metrics['min_wall_clearance_mm'] = wall_clearance
        penetration = float(info.get('contact_penetration_mm', 0.0))
        metrics['max_contact_penetration_mm'] = max(metrics['max_contact_penetration_mm'], penetration)
        centerline_offset = float(info.get('tip_centerline_offset_mm', 0.0))
        metrics['centerline_offset_sum'] += centerline_offset
        metrics['centerline_offset_max_mm'] = max(metrics['centerline_offset_max_mm'], centerline_offset)
        wall_force = float(info.get('wall_contact_force_n', 0.0))
        metrics['wall_force_sum'] += wall_force
        metrics['wall_force_max_n'] = max(metrics['wall_force_max_n'], wall_force)
        contact_load = float(info.get('contact_load_n', 0.0))
        metrics['contact_load_sum'] += contact_load
        metrics['contact_load_max_n'] = max(metrics['contact_load_max_n'], contact_load)
        magnetic_force = float(info.get('magnetic_force_n', 0.0))
        metrics['magnetic_force_sum'] += magnetic_force
        metrics['magnetic_force_max_n'] = max(metrics['magnetic_force_max_n'], magnetic_force)
        magnetic_torque = float(info.get('magnetic_torque_nm', 0.0))
        metrics['magnetic_torque_sum'] += magnetic_torque
        metrics['magnetic_torque_max_nm'] = max(metrics['magnetic_torque_max_nm'], magnetic_torque)
        metrics['last_info'] = dict(info)

    def _finalize_episode(self, env_index: int, info: dict[str, Any], reward: float) -> None:
        self._update_episode_metrics(env_index, info, reward)
        metrics = self._metrics_for_env(env_index)
        episode_info = dict(info.get('episode', {}))
        steps = max(int(metrics['steps']), 1)
        row = {
            'episode': int(self._episode_index),
            'env_index': int(env_index),
            'timesteps': int(self.num_timesteps),
            'episode_reward': float(episode_info.get('r', metrics['reward_sum'])),
            'episode_length': int(episode_info.get('l', steps)),
            'success': bool(info.get('success', False)),
            'invalid': bool(info.get('invalid', False)),
            'distance_to_goal_mm': float(info.get('distance_to_goal_mm', 0.0)),
            'min_wall_clearance_mm': float(metrics['min_wall_clearance_mm']),
            'max_contact_penetration_mm': float(metrics['max_contact_penetration_mm']),
            'mean_centerline_offset_mm': float(metrics['centerline_offset_sum'] / steps),
            'max_centerline_offset_mm': float(metrics['centerline_offset_max_mm']),
            'mean_wall_contact_force_n': float(metrics['wall_force_sum'] / steps),
            'max_wall_contact_force_n': float(metrics['wall_force_max_n']),
            'mean_contact_load_n': float(metrics['contact_load_sum'] / steps),
            'max_contact_load_n': float(metrics['contact_load_max_n']),
            'mean_magnetic_force_n': float(metrics['magnetic_force_sum'] / steps),
            'max_magnetic_force_n': float(metrics['magnetic_force_max_n']),
            'mean_magnetic_torque_nm': float(metrics['magnetic_torque_sum'] / steps),
            'max_magnetic_torque_nm': float(metrics['magnetic_torque_max_nm']),
            'wall_contact_steps': int(metrics['wall_contact_steps']),
            'reset_mode': str(info.get('reset_mode', 'unknown')),
            'target_point_mm': list(map(float, info.get('target_point_mm', [0.0, 0.0, 0.0]))),
            'goal_point_mm': list(map(float, info.get('goal_point_mm', [0.0, 0.0, 0.0]))),
        }
        if self._episode_writer is not None and self._episode_csv_file is not None:
            self._episode_writer.writerow(row)
            self._episode_csv_file.flush()

        self.logger.record('rollout/episode_reward', float(row['episode_reward']))
        self.logger.record('rollout/episode_length', float(row['episode_length']))
        self.logger.record('rollout/distance_to_goal_mm', float(row['distance_to_goal_mm']))
        self.logger.record('rollout/min_wall_clearance_mm', float(row['min_wall_clearance_mm']))
        self.logger.record('rollout/max_contact_penetration_mm', float(row['max_contact_penetration_mm']))
        self.logger.record('rollout/mean_centerline_offset_mm', float(row['mean_centerline_offset_mm']))
        self.logger.record('rollout/mean_wall_contact_force_n', float(row['mean_wall_contact_force_n']))
        self.logger.record('rollout/mean_contact_load_n', float(row['mean_contact_load_n']))
        self.logger.record('rollout/mean_magnetic_force_n', float(row['mean_magnetic_force_n']))
        self.logger.record('rollout/mean_magnetic_torque_nm', float(row['mean_magnetic_torque_nm']))
        self.logger.record('rollout/success', float(bool(row['success'])))

        self._latest_episode_row = row
        self._maybe_write_live_status(force=True)

        self._log_line(
            '[RL] '
            f'episode={row["episode"]} '
            f'env={row["env_index"]} '
            f'reward={row["episode_reward"]:.3f} '
            f'len={row["episode_length"]} '
            f'success={row["success"]} '
            f'dist={row["distance_to_goal_mm"]:.3f} mm '
            f'minClr={row["min_wall_clearance_mm"]:.3f} mm '
            f'wallF={row["mean_wall_contact_force_n"]:.3f}/{row["max_wall_contact_force_n"]:.3f} N '
            f'magF={row["mean_magnetic_force_n"]:.3f}/{row["max_magnetic_force_n"]:.3f} N '
            f'magT={row["mean_magnetic_torque_nm"]:.5f}/{row["max_magnetic_torque_nm"]:.5f} N.m'
        )

        self._episode_index += 1
        self._episode_metrics_by_env[env_index] = self._new_episode_metrics()

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])
        for idx, info in enumerate(infos):
            reward = float(rewards[idx]) if idx < len(rewards) else 0.0
            done = bool(dones[idx]) if idx < len(dones) else False
            if done:
                self._finalize_episode(idx, info, reward)
            else:
                self._update_episode_metrics(idx, info, reward)

        self._sync_progress()
        if (self.num_timesteps - self._last_saved_step) >= self.save_freq:
            self._save_latest()
        return True

    def _on_training_end(self) -> None:
        self._sync_progress(force=True)
        self._save_latest()
        self.close_progress()
        if self._episode_csv_file is not None:
            self._episode_csv_file.flush()
            self._episode_csv_file.close()
            self._episode_csv_file = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train PPO on the SOFA guidewire magnetic-navigation environment.')
    parser.add_argument('--timesteps', type=int, default=200000, help='Total PPO timesteps for this training run.')
    parser.add_argument('--checkpoint-freq', type=int, default=2000, help='Save the latest PPO checkpoint every N timesteps.')
    parser.add_argument('--device', default='auto', help='Stable Baselines3 device setting, e.g. auto/cpu/cuda.')
    parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel SOFA environments. Use 1 to disable subprocess parallelism.')
    parser.add_argument('--sim-steps-per-action', type=int, default=5, help='Number of SOFA animate() calls per PPO action.')
    parser.add_argument('--max-episode-steps', type=int, default=400, help='Maximum environment steps per episode.')
    parser.add_argument('--n-steps', type=int, default=256, help='PPO rollout length before each update.')
    parser.add_argument('--batch-size', type=int, default=64, help='PPO minibatch size.')
    parser.add_argument('--learning-rate', type=float, default=3.0e-4, help='PPO learning rate.')
    parser.add_argument('--gamma', type=float, default=0.995, help='Discount factor.')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda.')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient.')
    parser.add_argument('--rebuild-on-reset', action='store_true', help='Disable in-place scene reset and rebuild the full SOFA scene every episode.')
    parser.add_argument('--fresh-run', action='store_true', help='Ignore the latest checkpoint and start a new PPO model.')
    return parser.parse_args(argv)


def make_env(args: argparse.Namespace, env_index: int):
    def _init():
        monitor_path = MONITOR_DIR / f'ppo_guidewire_monitor_env{int(env_index):02d}.csv'
        env = GuidewireEnv(
            sim_steps_per_action=int(args.sim_steps_per_action),
            max_episode_steps=int(args.max_episode_steps),
            reuse_scene_on_reset=not bool(args.rebuild_on_reset),
        )
        return Monitor(env, filename=str(monitor_path))

    return _init


def build_env(args: argparse.Namespace) -> VecEnv:
    env_fns = [make_env(args, env_index) for env_index in range(int(max(args.num_envs, 1)))]
    if len(env_fns) == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method='spawn')


def load_or_create_model(env: VecEnv, args: argparse.Namespace) -> PPO:
    ensure_output_dirs()
    if CHECKPOINT_PATH.exists() and not bool(args.fresh_run):
        print(f'[RL] Resuming from checkpoint: {CHECKPOINT_PATH}')
        model = PPO.load(
            str(CHECKPOINT_PATH),
            env=env,
            device=args.device,
            tensorboard_log=str(TENSORBOARD_LOG_DIR),
        )
    else:
        if bool(args.fresh_run):
            print('[RL] Fresh run requested. Creating a new PPO model.')
        else:
            print('[RL] No checkpoint found. Creating a new PPO model.')
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log=str(TENSORBOARD_LOG_DIR),
            device=args.device,
            n_steps=int(max(args.n_steps, 32)),
            batch_size=int(max(args.batch_size, 16)),
            learning_rate=float(args.learning_rate),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            ent_coef=float(args.ent_coef),
            policy_kwargs={'net_arch': [128, 128]},
        )
    return model


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_output_dirs()
    env = build_env(args)
    callback = TrainingStatusCallback(
        total_timesteps=int(args.timesteps),
        num_envs=int(max(args.num_envs, 1)),
        save_freq=int(args.checkpoint_freq),
        save_path=CHECKPOINT_PATH,
        episode_csv_path=EPISODE_CSV_PATH,
        status_json_path=STATUS_JSON_PATH,
    )

    model: PPO | None = None
    interrupted = False
    try:
        model = load_or_create_model(env, args)
        print(f'[RL] Output root: {OUTPUT_ROOT}')
        print(
            f'[RL] Parallel envs: {int(max(args.num_envs, 1))} | '
            f'reset mode: {"rebuild" if bool(args.rebuild_on_reset) else "in-place reuse"}'
        )
        model.learn(
            total_timesteps=int(args.timesteps),
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name='ppo_guidewire',
        )
        model.save(str(CHECKPOINT_PATH))
        print(f'[RL] Latest model saved to: {CHECKPOINT_PATH}')
        print(f'[RL] TensorBoard log dir: {TENSORBOARD_LOG_DIR}')
        print(f'[RL] Episode metrics CSV: {EPISODE_CSV_PATH}')
        print(f'[RL] Monitor dir: {MONITOR_DIR}')
    except KeyboardInterrupt:
        interrupted = True
        print('[RL] Training interrupted by user. Saving latest checkpoint...')
        if model is not None:
            callback.save_latest()
            model.save(str(CHECKPOINT_PATH))
            print(f'[RL] Latest model saved to: {CHECKPOINT_PATH}')
    finally:
        callback.close_progress()
        env.close()

    if interrupted:
        print('[RL] Safe to resume later by re-running the same training command.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
