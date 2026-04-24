# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'train_rl.py requires stable_baselines3 and torch. Install them before launching PPO training.'
    ) from exc

from guidewire_scene.rl_env import GuidewireEnv


ROOT_DIR = Path(__file__).resolve().parent
TENSORBOARD_LOG_DIR = ROOT_DIR / 'ppo_guidewire_tensorboard'
CHECKPOINT_PATH = ROOT_DIR / 'ppo_guidewire_latest.zip'
MONITOR_CSV_PATH = ROOT_DIR / 'ppo_guidewire_monitor.csv'


class SafeCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: Path, verbose: int = 1) -> None:
        super().__init__(verbose=verbose)
        self.save_freq = int(max(save_freq, 1))
        self.save_path = Path(save_path)
        self._last_saved_step = 0

    def _save_latest(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.save_path))
        self._last_saved_step = int(self.num_timesteps)
        print('Checkpoint saved. Safe to interrupt.')

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_saved_step) >= self.save_freq:
            self._save_latest()
        return True

    def _on_training_end(self) -> None:
        self._save_latest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train PPO on the beam-based SOFA guidewire environment.')
    parser.add_argument('--timesteps', type=int, default=200000, help='Total PPO timesteps for this training run.')
    parser.add_argument('--checkpoint-freq', type=int, default=5000, help='Save the latest PPO checkpoint every N timesteps.')
    parser.add_argument('--device', default='auto', help='Stable Baselines3 device setting, e.g. auto/cpu/cuda.')
    parser.add_argument('--sim-steps-per-action', type=int, default=5, help='Number of SOFA animate() calls per PPO action.')
    parser.add_argument('--max-episode-steps', type=int, default=400, help='Maximum environment steps per episode.')
    return parser.parse_args(argv)


def build_env(args: argparse.Namespace) -> Monitor:
    env = GuidewireEnv(
        sim_steps_per_action=int(args.sim_steps_per_action),
        max_episode_steps=int(args.max_episode_steps),
    )
    return Monitor(env, filename=str(MONITOR_CSV_PATH))


def load_or_create_model(env: Monitor, args: argparse.Namespace) -> PPO:
    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_PATH.exists():
        print(f'[RL] Resuming from checkpoint: {CHECKPOINT_PATH}')
        model = PPO.load(
            str(CHECKPOINT_PATH),
            env=env,
            device=args.device,
            tensorboard_log=str(TENSORBOARD_LOG_DIR),
        )
    else:
        print('[RL] No checkpoint found. Creating a new PPO model.')
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log=str(TENSORBOARD_LOG_DIR),
            device=args.device,
        )
    return model


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env = build_env(args)
    callback = SafeCheckpointCallback(
        save_freq=int(args.checkpoint_freq),
        save_path=CHECKPOINT_PATH,
    )

    try:
        model = load_or_create_model(env, args)
        model.learn(
            total_timesteps=int(args.timesteps),
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name='ppo_guidewire',
        )
        model.save(str(CHECKPOINT_PATH))
        print(f'[RL] Latest model saved to: {CHECKPOINT_PATH}')
        print(f'[RL] TensorBoard log dir: {TENSORBOARD_LOG_DIR}')
    finally:
        env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
