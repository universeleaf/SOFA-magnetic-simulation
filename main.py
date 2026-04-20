# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guidewire_scene.config import SCENE_AUTOPLAY
from guidewire_scene.runtime import launch_runsofa_with_scene
from guidewire_scene.scene import createScene


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Launch the guidewire SOFA scene.')
    parser.add_argument(
        '--autoplay',
        action='store_true',
        default=SCENE_AUTOPLAY,
        help='Start animation immediately after the scene opens.',
    )
    parser.add_argument(
        '--no-autoplay',
        action='store_false',
        dest='autoplay',
        help='Open the scene in a static preview state.',
    )
    return parser.parse_args(argv)


def launch_current_scene(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return launch_runsofa_with_scene(Path(__file__).resolve(), autoplay=bool(args.autoplay))


if __name__ == '__main__':
    exe_name = Path(sys.executable).name.lower()
    if 'runsofa' not in exe_name:
        raise SystemExit(launch_current_scene(sys.argv[1:]))
