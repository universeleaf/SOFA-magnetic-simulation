# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

from .config import (
    ALLOW_PLUGIN_MISSING_FALLBACK,
    ELASTIC_ROD_PLUGIN_BUILD_DIR,
    ELASTIC_ROD_PLUGIN_NAME,
    ELASTIC_ROD_PLUGIN_SOURCE_DIR,
    ELASTICROD_STABILIZATION_MODE,
    GUIDEWIRE_BACKEND,
    SCENE_AUTOPLAY,
)


def _candidate_sofa_roots() -> List[Path]:
    roots = []
    env_root = os.environ.get('SOFA_ROOT', '').strip()
    if env_root:
        roots.append(Path(env_root))
    roots.extend([
        Path(r'D:\SOFA_v25.06.00_Win64\SOFA_v25.06.00_Win64'),
        Path(r'D:\Sofa\magnetic-soft-robots-main\SOFA_v24.12.00_Win64'),
    ])
    out, seen = [], set()
    for root in roots:
        key = str(root).lower()
        if key not in seen and root.exists():
            seen.add(key)
            out.append(root)
    return out


def _candidate_local_plugin_dirs() -> Iterable[Path]:
    yield ELASTIC_ROD_PLUGIN_BUILD_DIR
    yield ELASTIC_ROD_PLUGIN_BUILD_DIR / 'bin'
    yield ELASTIC_ROD_PLUGIN_BUILD_DIR / 'Release'
    yield ELASTIC_ROD_PLUGIN_BUILD_DIR / 'RelWithDebInfo'
    yield ELASTIC_ROD_PLUGIN_BUILD_DIR / 'Debug'
    yield ELASTIC_ROD_PLUGIN_SOURCE_DIR / 'bin'


def _candidate_local_plugin_files() -> Iterable[Path]:
    names = [
        f'{ELASTIC_ROD_PLUGIN_NAME}_hotfix_densecollision.dll',
        f'{ELASTIC_ROD_PLUGIN_NAME}_hotfix_contactrelief.dll',
        f'{ELASTIC_ROD_PLUGIN_NAME}_hotfix_headbarrier.dll',
        f'{ELASTIC_ROD_PLUGIN_NAME}_hotfix_recenterpush.dll',
        f'{ELASTIC_ROD_PLUGIN_NAME}_hotfix_signclamp.dll',
        f'{ELASTIC_ROD_PLUGIN_NAME}_hotfix_turnfix.dll',
        f'{ELASTIC_ROD_PLUGIN_NAME}_hotfix_branchcommit.dll',
        f'{ELASTIC_ROD_PLUGIN_NAME}_hotfix_turnhold.dll',
        f'{ELASTIC_ROD_PLUGIN_NAME}.dll',
        f'{ELASTIC_ROD_PLUGIN_NAME}_d.dll',
        f'lib{ELASTIC_ROD_PLUGIN_NAME}.so',
        f'lib{ELASTIC_ROD_PLUGIN_NAME}.dylib',
    ]
    for folder in _candidate_local_plugin_dirs():
        for name in names:
            yield folder / name


def find_runsofa_exe() -> Path | None:
    for root in _candidate_sofa_roots():
        exe = root / 'bin' / 'runSofa.exe'
        if exe.exists():
            return exe
    return None


def bootstrap_sofa_python() -> None:
    for root in _candidate_sofa_roots():
        site = root / 'plugins' / 'SofaPython3' / 'lib' / 'python3' / 'site-packages'
        bin_dir = root / 'bin'
        lib_dir = root / 'lib'
        if not (site.exists() and bin_dir.exists()):
            continue
        for path in (str(site), str(bin_dir)):
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            os.add_dll_directory(str(bin_dir))
            if lib_dir.exists():
                os.add_dll_directory(str(lib_dir))
        except Exception:
            os.environ['PATH'] = str(bin_dir) + ';' + str(lib_dir) + ';' + os.environ.get('PATH', '')
        return


def ensure_sofa():
    try:
        import Sofa
        import Sofa.Core
    except ModuleNotFoundError:
        bootstrap_sofa_python()
        import Sofa
        import Sofa.Core
    return Sofa


def add_local_plugin_search_paths() -> None:
    for folder in _candidate_local_plugin_dirs():
        if not folder.exists():
            continue
        try:
            os.add_dll_directory(str(folder))
        except Exception:
            os.environ['PATH'] = str(folder) + ';' + os.environ.get('PATH', '')


def load_elastic_rod_plugin() -> str:
    """
    优先按本地开发版 DLL 绝对路径加载，避免 runSofa 在插件仓库里先报一条
    “Plugin not found: ElasticRodGuidewire”的误导性错误。

    若本地 DLL 不存在，再回退到按插件名加载，兼容已经安装到 SOFA 插件目录的场景。
    """
    ensure_sofa()
    add_local_plugin_search_paths()
    import SofaRuntime

    errors: list[str] = []

    for dll_path in _candidate_local_plugin_files():
        if not dll_path.exists():
            continue
        try:
            loaded = bool(SofaRuntime.importPlugin(str(dll_path)))
            if loaded:
                print(f'[INFO] Loaded SOFA plugin by path: {dll_path}')
                return str(dll_path)
            errors.append(f'{dll_path}:importPlugin returned False')
        except Exception as exc:
            errors.append(f'{dll_path}:{exc}')

    try:
        loaded = bool(SofaRuntime.importPlugin(ELASTIC_ROD_PLUGIN_NAME))
        if loaded:
            print(f'[INFO] Loaded SOFA plugin by name: {ELASTIC_ROD_PLUGIN_NAME}')
            return ELASTIC_ROD_PLUGIN_NAME
        errors.append('name:importPlugin returned False')
    except Exception as exc:
        errors.append(f'name:{exc}')

    joined = ' | '.join(errors) if errors else '未找到已编译的插件库文件。'
    raise RuntimeError(
        '无法加载 ElasticRodGuidewire 插件。请先编译 '
        f'{ELASTIC_ROD_PLUGIN_SOURCE_DIR}，并确认 {ELASTIC_ROD_PLUGIN_BUILD_DIR} 下存在 '
        f'{ELASTIC_ROD_PLUGIN_NAME}.dll。详情: {joined}'
    )


def launch_runsofa_with_scene(scene_file: Path, autoplay: bool = SCENE_AUTOPLAY) -> int:
    runsofa = find_runsofa_exe()
    if runsofa is None:
        print('[ERROR] runSofa.exe not found. Please set SOFA_ROOT.')
        return 1
    cmd = [str(runsofa), '-g', 'qt', '-l', 'SofaPython3']
    if autoplay:
        cmd.append('-a')
    cmd.append(str(scene_file.resolve()))
    env = os.environ.copy()
    if GUIDEWIRE_BACKEND == 'elasticrod':
        env['GUIDEWIRE_ELASTICROD_GUI_WALLCLOCK'] = '1'
    print('[INFO] Launching scene with runSofa:')
    print(f'       runSofa: {runsofa}')
    print(f'       autoplay: {bool(autoplay)}')
    print('       ' + ' '.join(cmd))
    return subprocess.call(cmd, env=env)
