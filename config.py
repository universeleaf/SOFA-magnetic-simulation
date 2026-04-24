# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / 'assets'
CENTERLINE_DIR = ASSETS_DIR / 'centerline'
EXTRACTED_CENTERLINE_DIR = CENTERLINE_DIR / 'extracted_paths'
EXTRACTED_BRANCH_DIR = EXTRACTED_CENTERLINE_DIR / 'branches'
EXTRACTED_ROUTE_DIR = EXTRACTED_CENTERLINE_DIR / 'routes'
ROUTE_CATALOG_FILE = EXTRACTED_CENTERLINE_DIR / 'route_catalog.json'
OPTION_TXT = ASSETS_DIR / 'option.txt'
VESSEL_OBJ = ASSETS_DIR / 'vessel.obj'

# 路线切换入口：
# 以后做 PPT 或对比实验时，只需要改 `SELECTED_ROUTE_NAME` 这一处。
# 每个键都映射到一个具体的 npy 路径文件，不需要再手动复制 alias 文件。
ROUTE_LIBRARY = {
    # 右下最外侧入口 -> 右上最外侧出口（当前手工修过的小折返版本）
    'right_outer_main': EXTRACTED_ROUTE_DIR / 'route_rightmost_lower_inlet_to_rightmost_upper_outlet.npy',
    'right_alt_inlet_outer_main': EXTRACTED_ROUTE_DIR / 'route_bottom_right_to_upper_right_outer_clean.npy',
    'left_inlet_outer_main': EXTRACTED_ROUTE_DIR / 'route_seg20_to_seg27.npy',
    # 同一入口，改走右上偏内侧的另一条通路，适合做 PPT 多路线展示
    'right_upper_inner_alt': EXTRACTED_ROUTE_DIR / 'route_seg11_to_seg2.npy',
    # 同一入口，改走右上中间偏外通路
    'right_upper_mid_alt': EXTRACTED_ROUTE_DIR / 'route_seg11_to_seg4.npy',
    # 同一入口，改走左上外侧通路
    'left_outer_alt': EXTRACTED_ROUTE_DIR / 'route_seg11_to_seg27.npy',
    # 同一入口，改走左上中外侧通路
    'left_mid_alt': EXTRACTED_ROUTE_DIR / 'route_seg11_to_seg29.npy',
}

ROUTE_DESCRIPTIONS = {
    'right_outer_main': '右下最外侧入口 -> 右上最外侧出口',
    'right_alt_inlet_outer_main': '右下备用入口 -> 右上最外侧出口',
    'left_inlet_outer_main': '左下入口 -> 左上最外侧出口',
    'right_upper_inner_alt': '右下最外侧入口 -> 右上偏内侧出口',
    'right_upper_mid_alt': '右下最外侧入口 -> 右上中间偏外出口',
    'left_outer_alt': '右下最外侧入口 -> 左上最外侧出口',
    'left_mid_alt': '右下最外侧入口 -> 左上中外侧出口',
}

# 路线合法性判断：
# - 入口必须落在血管下方区域；
# - 出口必须落在血管上方区域；
# - 整条路径必须有足够大的纵向跨度。
# 这样可以避免把“入口在下方、出口仍然停在中下部”的路径误当成一条完整展示通路。
LOWER_ENTRY_MAX_Y_MM = 40.0
UPPER_EXIT_MIN_Y_MM = 180.0
MIN_ROUTE_VERTICAL_SPAN_MM = 140.0

# 默认切回上一条正确路线：
# 右下最外侧入口 -> 右上最外侧出口。
SELECTED_ROUTE_NAME = 'right_outer_main'


def _route_endpoints_in_scene(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(np.load(path), dtype=float)
    pts = np.atleast_2d(pts)
    if pts.shape[1] > 3:
        pts = pts[:, :3]
    if pts.shape[0] < 2:
        raise ValueError(f'Route file must contain at least 2 points: {path}')
    start = pts[0, :3].astype(float)
    end = pts[-1, :3].astype(float)
    if float(start[1]) > float(end[1]):
        start, end = end, start
    return start, end


def _validate_selected_route(route_name: str) -> Path:
    path = ROUTE_LIBRARY[route_name]
    if not path.exists():
        raise FileNotFoundError(f'Selected route file not found: {path}')
    start, end = _route_endpoints_in_scene(path)
    vertical_span = float(end[1] - start[1])
    if float(start[1]) > LOWER_ENTRY_MAX_Y_MM:
        raise ValueError(
            f'Selected route "{route_name}" has an invalid lower inlet: startY={start[1]:.3f} mm > {LOWER_ENTRY_MAX_Y_MM:.1f} mm'
        )
    if float(end[1]) < UPPER_EXIT_MIN_Y_MM:
        raise ValueError(
            f'Selected route "{route_name}" has an invalid upper outlet: endY={end[1]:.3f} mm < {UPPER_EXIT_MIN_Y_MM:.1f} mm'
        )
    if vertical_span < MIN_ROUTE_VERTICAL_SPAN_MM:
        raise ValueError(
            f'Selected route "{route_name}" has insufficient vertical span: {vertical_span:.3f} mm < {MIN_ROUTE_VERTICAL_SPAN_MM:.1f} mm'
        )
    return path


SELECTED_ROUTE_FILE = _validate_selected_route(SELECTED_ROUTE_NAME)
CENTERLINE_FILE = SELECTED_ROUTE_FILE
CENTERLINE_CANDIDATES = [
    SELECTED_ROUTE_FILE,
    ROUTE_LIBRARY['right_outer_main'],
    ROUTE_LIBRARY['right_upper_mid_alt'],
    ROUTE_LIBRARY['left_outer_alt'],
    ROUTE_LIBRARY['left_mid_alt'],
    EXTRACTED_ROUTE_DIR / 'route_bottom_right_to_upper_right_outer_clean.npy',
    EXTRACTED_ROUTE_DIR / 'route_seg23_to_seg0.npy',
    EXTRACTED_ROUTE_DIR / 'route_seg23_to_seg27.npy',
    EXTRACTED_ROUTE_DIR / 'route_seg20_to_seg27.npy',
    EXTRACTED_ROUTE_DIR / 'route_seg23_to_seg29.npy',
    EXTRACTED_ROUTE_DIR / 'route_seg20_to_seg29.npy',
    CENTERLINE_DIR / 'edge_lower_nav_path.npy',
    CENTERLINE_DIR / 'nav_path.npy',
    CENTERLINE_DIR / 'main_nav_path.npy',
    CENTERLINE_DIR / 'vessel_centerline_world.npy',
    CENTERLINE_DIR / 'vessel_centerline_transformed.npy',
    CENTERLINE_DIR / 'vessel_centerline_3.npy',
]
ROUTE_REVISIT_TOLERANCE_MM = 0.15
ROUTE_REVISIT_MIN_INDEX_GAP = 8
ROUTE_REVISIT_MIN_ARC_MM = 5.0
SCENE_BACKGROUND_RGBA = [1.0, 1.0, 1.0, 1.0]
SCENE_AUTOPLAY = False
ENABLE_CAMERA_FOLLOW = False
VESSEL_VISUAL_RGBA = [0.78, 0.26, 0.26, 0.62]
VESSEL_COLLISION_DEBUG_RGBA = [0.85, 0.28, 0.28, 0.02]

ELASTIC_ROD_PLUGIN_NAME = 'ElasticRodGuidewire'
ELASTIC_ROD_PLUGIN_BUILD_DIR = PACKAGE_DIR / 'build' / ELASTIC_ROD_PLUGIN_NAME
ELASTIC_ROD_PLUGIN_SOURCE_DIR = PACKAGE_DIR / 'native' / ELASTIC_ROD_PLUGIN_NAME
ELASTIC_ROD_PLUGIN_BUILD_SCRIPT = PACKAGE_DIR / 'build_plugin.bat'
ALLOW_PLUGIN_MISSING_FALLBACK = True
# 当前原生 `elasticrod` 后端仍处于实验状态：
# - 伸长与近端边界项已经接进 SOFA；
# - 但弯曲项还没有配套的隐式切线刚度矩阵，`EulerImplicitSolver`
#   在点击 Animate 后会把这根高刚度细杆近似当成“显式积分”，
#   结果就是第一两个时间步就把自由节点速度打爆，出现“导丝直接飞走”。
# 当前原生 `elasticrod` 路径已经补上了解析伸长/弯曲切线刚度，并把磁场项改回
# 更接近学长原始实现的“只加力、不单独加 Jacobian”语义。
# 因此默认入口切回 `elasticrod`，`beam` 继续保留为回退接口。
GUIDEWIRE_BACKEND = 'elasticrod'
SCENE_AUTOPLAY = GUIDEWIRE_BACKEND == 'elasticrod'
BEAM_RUNTIME_PROFILE = 'recording_10min'
if BEAM_RUNTIME_PROFILE not in {'quality', 'recording_10min'}:
    raise ValueError(f'Unsupported BEAM_RUNTIME_PROFILE: {BEAM_RUNTIME_PROFILE}')
# `safe`:
#   原生 rod + SOFA 接触之外，再保留一层“求解后腔内安全投影”兜底，优先保证动画稳定。
# `strict`:
#   关闭这层兜底，只保留原生 rod + SOFA 接触/约束链，用于纯接触实验。
ELASTICROD_STABILIZATION_MODE = 'strict'
if ELASTICROD_STABILIZATION_MODE not in {'safe', 'strict'}:
    raise ValueError(f'Unsupported ELASTICROD_STABILIZATION_MODE: {ELASTICROD_STABILIZATION_MODE}')
ELASTICROD_RUNTIME_PROFILE = 'realtime_gui_10min'
if ELASTICROD_RUNTIME_PROFILE not in {'quality', 'realtime_gui_10min'}:
    raise ValueError(f'Unsupported ELASTICROD_RUNTIME_PROFILE: {ELASTICROD_RUNTIME_PROFILE}')
ELASTICROD_DIAGNOSTIC_PROFILE = 'realtime'
if ELASTICROD_DIAGNOSTIC_PROFILE not in {'debug', 'realtime'}:
    raise ValueError(f'Unsupported ELASTICROD_DIAGNOSTIC_PROFILE: {ELASTICROD_DIAGNOSTIC_PROFILE}')


def _is_elasticrod_gui_wallclock_launch() -> bool:
    if os.environ.get('GUIDEWIRE_ELASTICROD_GUI_WALLCLOCK', '').strip() == '1':
        return True
    exe_name = Path(sys.executable).name.lower() if sys.executable else ''
    return 'runsofa' in exe_name


ELASTICROD_REALTIME_NODE_COUNT = 61
# A 61-node strict profile keeps the distal magnetic head resolved over
# multiple segments while staying close to the realtime RL budget once the
# contact-band feed and surface-monitoring heuristics are tuned.
ELASTICROD_GUI_NODE_COUNT = 61
ELASTICROD_REALTIME_STARTUP_RAMP_S = 0.15
ELASTICROD_REALTIME_DT_FREE_S = 5.5e-3
ELASTICROD_REALTIME_DT_TRANSITION_S = 4.2e-3
ELASTICROD_REALTIME_DT_CONTACT_S = 2.4e-3
ELASTICROD_GUI_WALLCLOCK_CONTROL = True
# Strict GUI was previously capped at 30 ms of wall-clock insertion time per
# frame plus an extra 1.8x speed scale. That looks acceptable only when the GUI
# is already running at high FPS; once runSofa drops near ~10 FPS or below, the
ELASTICROD_GUI_WALLCLOCK_DT_MAX_S = 0.03
ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_MAX_S = 0.0120
ELASTICROD_GUI_WALLCLOCK_PUSH_SPEED_SCALE = 3.0
ELASTICROD_GUI_WALLCLOCK_STARTUP_RAMP_S = 0.45
ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_FREE_SCALE = 1.75
ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_TRANSITION_SCALE = 1.30
ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_CONTACT_SCALE = 1.80
ELASTICROD_STRICT_GUI_LIGHT_CONTACT_INSERTION_DT_S = 0.0045
ELASTICROD_STRICT_GUI_LIGHT_CONTACT_CLEARANCE_MM = 0.22
ELASTICROD_STRICT_GUI_LIGHT_CONTACT_WALL_GAP_MM = 0.18
ELASTICROD_STRICT_GUI_LIGHT_CONTACT_MAX_BARRIER_NODES = 2
ELASTICROD_STRICT_GUI_GUIDED_CONTACT_INSERTION_DT_S = 0.0040
ELASTICROD_STRICT_GUI_GUIDED_CONTACT_CLEARANCE_MM = 0.16
ELASTICROD_STRICT_GUI_GUIDED_CONTACT_WALL_GAP_MM = 0.10
ELASTICROD_STRICT_GUI_GUIDED_CONTACT_MAX_BARRIER_NODES = 3
ELASTICROD_STRICT_GUI_GUIDED_PRECONTACT_CLEARANCE_MM = 0.28
ELASTICROD_STRICT_GUI_GUIDED_PRECONTACT_WALL_GAP_MM = 0.14
ELASTICROD_STRICT_GUI_GUIDED_PRECONTACT_MAX_BARRIER_NODES = 2
ELASTICROD_STRICT_RUNTIME_RELEASE_TRANSITION_HOLD_STEPS = 32
ELASTICROD_STRICT_RUNTIME_RELEASE_CONTACT_HOLD_STEPS = 18
ELASTICROD_GUI_DIAGNOSTIC_MIN_LOG_INTERVAL_STEPS = 80
ELASTICROD_GUI_DIAGNOSTIC_LINEAR_SPEED_WARN_MM_S = 450.0
# The older strict GUI wall-clock band pushed dt and solver looseness too far for
# the 81-node model and could blow up around step 9. Keep GUI on the same
# validated strict band as solver-time diagnostics; this gives up some speed but
# makes runSofa launch into a stable native path again.
ELASTICROD_GUI_DT_FREE_S = 5.2e-3
ELASTICROD_GUI_DT_TRANSITION_S = 4.0e-3
ELASTICROD_GUI_DT_CONTACT_S = 2.1e-3
ELASTICROD_GUI_SOLVER_MAX_ITER_FREE = 38 if ELASTICROD_STABILIZATION_MODE == 'strict' else 80
ELASTICROD_GUI_SOLVER_MAX_ITER_TRANSITION = 38 if ELASTICROD_STABILIZATION_MODE == 'strict' else 110
ELASTICROD_GUI_SOLVER_MAX_ITER_CONTACT = 170 if ELASTICROD_STABILIZATION_MODE == 'strict' else 160
ELASTICROD_GUI_SOLVER_TOL_FREE = 2.8e-4 if ELASTICROD_STABILIZATION_MODE == 'strict' else 1.0e-4
ELASTICROD_GUI_SOLVER_TOL_TRANSITION = 2.2e-4 if ELASTICROD_STABILIZATION_MODE == 'strict' else 7.0e-5
ELASTICROD_GUI_SOLVER_TOL_CONTACT = 1.1e-5 if ELASTICROD_STABILIZATION_MODE == 'strict' else 1.5e-5
ELASTICROD_REALTIME_SOLVER_MAX_ITER_FREE = 34
ELASTICROD_REALTIME_SOLVER_MAX_ITER_TRANSITION = 34
ELASTICROD_REALTIME_SOLVER_MAX_ITER_CONTACT = 130
ELASTICROD_REALTIME_SOLVER_TOL_FREE = 3.2e-4
ELASTICROD_REALTIME_SOLVER_TOL_TRANSITION = 2.8e-4
ELASTICROD_REALTIME_SOLVER_TOL_CONTACT = 1.1e-5
ELASTICROD_REALTIME_SPEED_SCALE_FREE = 4.0  # Increased from 2.8 for faster advancement
ELASTICROD_REALTIME_SPEED_SCALE_TRANSITION = 3.2
ELASTICROD_REALTIME_SPEED_SCALE_CONTACT = 1.80
ELASTICROD_REALTIME_STEERING_ENTER_DEG = 35.0
ELASTICROD_REALTIME_STEERING_CONTACT_DEG = 55.0
ELASTICROD_REALTIME_CLEARANCE_TRANSITION_MM = 0.65
ELASTICROD_REALTIME_CLEARANCE_CONTACT_MM = 0.40
ELASTICROD_ENABLE_LUMEN_SAFETY_PROJECTION = ELASTICROD_STABILIZATION_MODE == 'safe'
ELASTICROD_ENABLE_SAFE_RECOVERY = ELASTICROD_STABILIZATION_MODE == 'safe'
ELASTICROD_ENABLE_DISPLACEMENT_PUSH = ELASTICROD_STABILIZATION_MODE == 'safe'
ELASTICROD_DISPLACEMENT_PUSH_VELOCITY_MM_PER_S = 36.0
ELASTICROD_DISPLACEMENT_PUSH_RELEASE_MM = 24.0
ELASTICROD_DISPLACEMENT_PUSH_CENTERING_GAIN = 0.78
# Strict keeps native rod/contact/magnetic actuation as the primary physics, but
# the first bend can generate a tiny mesh-level wall violation after prolonged
# wall-following. Keep a minimal post-solve guard enabled only as an emergency
# backstop so RL episodes keep running instead of stopping on a -0.02 mm scrape.
ELASTICROD_ENABLE_STRICT_POSTSOLVE_GUARD = True
ELASTICROD_ENABLE_STRICT_LUMEN_CLAMP = True
ELASTICROD_STRICT_LUMEN_CLAMP_TOLERANCE_MM = 0.005
ELASTICROD_STRICT_MAX_LINEAR_SPEED_MM_S = 2000.0
ELASTICROD_STRICT_NATIVE_LUMEN_BARRIER = False
ELASTICROD_STRICT_EXTERNAL_SUPPORT_LENGTH_MM = 42.0
ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_N = 1.55
ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_MAX_N = 2.20
ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT = 5
ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER = True
ELASTICROD_STRICT_SUPPORT_WINDOW_LENGTH_MM = ELASTICROD_STRICT_EXTERNAL_SUPPORT_LENGTH_MM
ELASTICROD_STRICT_SUPPORT_RELEASE_MM = 18.0
ELASTICROD_STRICT_DRIVE_WINDOW_LENGTH_MM = ELASTICROD_STRICT_EXTERNAL_SUPPORT_LENGTH_MM
ELASTICROD_STRICT_DRIVE_WINDOW_OUTSIDE_OFFSET_MM = 0.0
ELASTICROD_STRICT_DRIVE_WINDOW_MIN_NODE_COUNT = 3
ELASTICROD_STRICT_BARRIER_ACTIVATION_MARGIN_MM = 0.60  # Keep the native profile barrier as a soft pre-contact guide.
ELASTICROD_STRICT_BARRIER_SAFETY_MARGIN_MM = 0.12
ELASTICROD_STRICT_BARRIER_STIFFNESS_N_PER_M = 18000.0
ELASTICROD_STRICT_BARRIER_DAMPING_N_S_PER_M = 3.0
ELASTICROD_STRICT_BARRIER_MAX_FORCE_PER_NODE_N = 1.6
ELASTICROD_STRICT_HEAD_STRETCH_SOFT_LIMIT = 0.025
ELASTICROD_STRICT_HEAD_STRETCH_LIMIT = 0.07
ELASTICROD_STRICT_GLOBAL_STRETCH_SOFT_LIMIT = 0.035
ELASTICROD_STRICT_GLOBAL_STRETCH_HARD_LIMIT = 0.08
ELASTICROD_STRICT_AXIAL_STIFFNESS_SCALE = 16.0
ELASTICROD_STRICT_AXIAL_USE_BODY_FLOOR = True
ELASTICROD_SAFE_AXIAL_STIFFNESS_SCALE = 5.4
ELASTICROD_SAFE_AXIAL_USE_BODY_FLOOR = True
ELASTICROD_STRICT_SINGULARITY_HOLD_STEPS = 0
ELASTICROD_STRICT_PUSH_MIN_SCALE_CONTACT = 1.0
ELASTICROD_STRICT_PUSH_SCALE_WHEN_BARRIER_ACTIVE = ELASTICROD_STRICT_PUSH_MIN_SCALE_CONTACT
ELASTICROD_STRICT_RECOVERY_PUSH_MIN_SCALE = 1.0
ELASTICROD_STRICT_AXIAL_ASSIST_SCALE_WHEN_STEERING = 0.0
ELASTICROD_STRICT_AXIAL_ASSIST_SCALE_RECOVERY = 0.0
ELASTICROD_STRICT_ALWAYS_PUSH_FORWARD = True
# Strict magnetic steering should stay quiet for the first short straight
# entry segment, but the previous 6 mm + 8 mm release gate kept the field
# effectively off even after the head had already started bending toward the
# first wall. Shorten the dormant interval so the magnetic tip can start
# recentring once the head has actually entered the lumen, without turning the
# field into an artificial forward pull at t=0.
ELASTICROD_STRICT_INITIAL_STRAIGHT_PUSH_MM = 0.5
ELASTICROD_STRICT_MAGNETIC_RELEASE_SPAN_MM = 1.5
ELASTICROD_STRICT_MAGNETIC_LOOKAHEAD_MM = 20.0
ELASTICROD_STRICT_MAGNETIC_RECOVERY_LOOKAHEAD_MM = 10.0
ELASTICROD_STRICT_MAGNETIC_RECOVERY_SCALE_FLOOR = 0.85
ELASTICROD_STRICT_MAGNETIC_RECENTER = True
ELASTICROD_STRICT_MAGNETIC_ASSIST = True
ELASTICROD_STRICT_MAGNETIC_PREVIEW_SCALING = True
# Optional strict lateral centering cap kept for experiments, but disabled by
# default because it is not generated by a uniform magnetic field itself.
ELASTICROD_STRICT_TIP_TARGET_FORCE_N = 0.30
ELASTICROD_STRICT_INITIAL_AXIS_HOLD_MM = 0.0
ELASTICROD_STRICT_BEND_LOOKAHEAD_MM = 22.0
ELASTICROD_STRICT_BEND_NEAR_WINDOW_MM = 6.5
ELASTICROD_STRICT_BEND_TURN_MEDIUM_DEG = 9.0
ELASTICROD_STRICT_BEND_TURN_HIGH_DEG = 20.0
ELASTICROD_STRICT_FIELD_SCALE_STRAIGHT = 0.68 if ELASTICROD_STRICT_MAGNETIC_PREVIEW_SCALING else 1.0
ELASTICROD_STRICT_FIELD_SCALE_BEND = 0.93 if ELASTICROD_STRICT_MAGNETIC_PREVIEW_SCALING else 1.0
ELASTICROD_STRICT_PUSH_SCALE_STRAIGHT_CONTACT = 1.00
ELASTICROD_STRICT_PUSH_SCALE_BEND_CONTACT = 1.00
ELASTICROD_STRICT_LIGHT_CONTACT_PUSH_SCALE = 1.00
ELASTICROD_STRICT_GUIDED_CONTACT_PUSH_SCALE = 1.00
ELASTICROD_STRICT_RUNTIME_PROGRESS_GATE_MM = max(
    ELASTICROD_STRICT_INITIAL_STRAIGHT_PUSH_MM,
    ELASTICROD_STRICT_MAGNETIC_RELEASE_SPAN_MM,
)
ELASTICROD_STRICT_RUNTIME_BEND_SEVERITY_TRANSITION = 0.20
ELASTICROD_STRICT_RUNTIME_BEND_SEVERITY_CONTACT = 0.70
ELASTICROD_STRICT_RECENTER_CLEARANCE_MM = 0.80 if ELASTICROD_STRICT_MAGNETIC_RECENTER else 0.0
ELASTICROD_STRICT_RECENTER_OFFSET_MM = 0.30 if ELASTICROD_STRICT_MAGNETIC_RECENTER else 0.0
ELASTICROD_STRICT_RECENTER_BLEND = 0.92 if ELASTICROD_STRICT_MAGNETIC_RECENTER else 0.0
ELASTICROD_STRICT_HEAD_STRETCH_RELIEF_START = 0.006
ELASTICROD_STRICT_HEAD_STRETCH_RELIEF_FULL = 0.018
ELASTICROD_STRICT_DRIVER_AXIAL_ERROR_SOFT_LIMIT_MM = 0.0
ELASTICROD_STRICT_DRIVER_AXIAL_ERROR_HARD_LIMIT_MM = 0.0
ELASTICROD_STRICT_DRIVER_REACTION_SOFT_LIMIT_N = 0.0
ELASTICROD_STRICT_DRIVER_REACTION_HARD_LIMIT_N = 0.0
ELASTICROD_STRICT_HEAD_STRETCH_FIELD_SCALE_FLOOR = 0.20
ELASTICROD_STRICT_GUI_EXACT_SURFACE_RECHECK_STEPS = 28
ELASTICROD_STRICT_GUI_SURFACE_REFRESH_NEAR_STEPS = 6
ELASTICROD_STRICT_GUI_SURFACE_REFRESH_FAR_STEPS = 28
ELASTICROD_STRICT_GUI_FAR_CLEARANCE_MM = 1.0
ELASTICROD_STRICT_GUI_FAR_OFFSET_MM = 1.0
# `elasticrod` 当前最容易把系统打爆的是“自碰撞 + 高刚度 native rod + 接触约束”这条链。
# 本轮先聚焦“导丝-刚性血管壁”主接触问题，默认关闭 native 自碰撞，避免把问题源混在一起。
ELASTICROD_ENABLE_SELF_COLLISION = False
ELASTICROD_STARTUP_RAMP_TIME_S = 0.20
ELASTICROD_DIAGNOSTIC_STEP_WINDOW = 50
ELASTICROD_DIAGNOSTIC_PRINT_EVERY = 20
ELASTICROD_DIAGNOSTIC_DISPLACEMENT_WARN_MM = 3.0
ELASTICROD_DIAGNOSTIC_LINEAR_SPEED_WARN_MM_S = 500.0
ELASTICROD_DIAGNOSTIC_ANGULAR_SPEED_WARN_RAD_S = 80.0
ELASTICROD_FAILFAST_MAX_STRETCH = 3.0
ELASTICROD_FAILFAST_EDGE_STRETCH_RATIO = 2.0
ELASTICROD_RECOVERY_TRIGGER_DISPLACEMENT_MM = 6.0
ELASTICROD_RECOVERY_TRIGGER_LINEAR_SPEED_MM_S = 1500.0
ELASTICROD_RECOVERY_TRIGGER_MAX_STRETCH = 0.50
ELASTICROD_SAFE_RECOVERY_DISPLACEMENT_MM = 5.0
ELASTICROD_SAFE_RECOVERY_LINEAR_SPEED_MM_S = 1500.0
ELASTICROD_SAFE_RECOVERY_ANGULAR_SPEED_RAD_S = 500.0
ELASTICROD_SAFE_RECOVERY_MAX_STRETCH = 2.0
ELASTICROD_SAFE_RECOVERY_COOLDOWN_STEPS = 8
ELASTICROD_SAFE_RECOVERY_RETRACT_MM = 0.25
ELASTICROD_SAFE_TIP_WALL_CONTACT_RELEASE_HOLD_STEPS = 3
ELASTICROD_ENABLE_INTRODUCER = ELASTICROD_STABILIZATION_MODE != 'strict'
ELASTICROD_INTRODUCER_LENGTH_MM = 30.0
# Start the native rod only slightly inside the lumen. The previous 10 mm
# preset already reaches a ~15 deg turning region of the chosen route, so the
# head appears pre-bent before any pushing happens.
ELASTICROD_INITIAL_TIP_INSERTION_MM = 2.0
# Use a short local tangent at the vessel entry as the elasticrod insertion
# axis. Averaging over the first 8 mm was pulling the external segment toward
# downstream curvature and made the wire enter at an oblique angle.
# The first 2 mm of this route are too short/noisy to define a stable physical
# insertion axis. Use a longer local average so the proximal push axis matches
# the actual vessel entry direction instead of immediately driving the head into
# the first wall.
ELASTICROD_ENTRY_AXIS_SAMPLE_LENGTH_MM = 8.0
ELASTICROD_INTRODUCER_PHYSICAL_CLEARANCE_MM = 0.05
ELASTICROD_INTRODUCER_WALL_THICKNESS_MM = 0.12
ELASTICROD_INTRODUCER_RADIAL_SEGMENTS = 20
ELASTICROD_INTRODUCER_VISUAL_RGBA = [0.38, 0.42, 0.46, 0.18]
ELASTICROD_STRICT_ENTRY_PUSH_BAND_ENABLED = False
ELASTICROD_ENTRY_PUSH_BAND_LENGTH_MM = ELASTICROD_STRICT_EXTERNAL_SUPPORT_LENGTH_MM
ELASTICROD_ENTRY_PUSH_BAND_OUTSIDE_OFFSET_MM = 0.0
ELASTICROD_ENTRY_PUSH_BAND_MIN_NODE_COUNT = ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT
ELASTICROD_STRICT_SIMPLE_TAIL_DRIVE = True
# Strict now exposes a free-entry variant for debugging the real entry
# mechanics. Disabling the virtual sheath must not silently replace it with a
# stronger proximal lateral guide, otherwise "sheath off" is not actually
# testing unconstrained insertion.
ELASTICROD_ENABLE_VIRTUAL_SHEATH = ELASTICROD_STABILIZATION_MODE != 'strict'
ELASTICROD_STRICT_HAND_PUSH_NODE_COUNT = ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT
ELASTICROD_STRICT_HAND_PUSH_TOTAL_FORCE_N = ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_N
ELASTICROD_STRICT_HAND_PUSH_MAX_TOTAL_N = ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_MAX_N
ELASTICROD_STRICT_HAND_PUSH_MIN_FORCE_SCALE = 1.0
# 近端 Vec6d 推力在 strict 下不与速度帽对消时仍可能被 thrust limit 压得过低，设下限保证可推进。
ELASTICROD_STRICT_HAND_PUSH_MIN_FORCE_SCALE = 1.0
ELASTICROD_STRICT_GUI_MAX_INSERTION_STEP_MM = 0.20
ELASTICROD_STRICT_FEED_BOOST_START_MM = 8.0
ELASTICROD_STRICT_FEED_BOOST_BEND = 1.16
ELASTICROD_STRICT_FEED_BOOST_CONTACT = 1.36
ELASTICROD_STRICT_FEED_BOOST_HEAD_STRETCH_LIMIT = 0.012
ELASTICROD_STRICT_MAX_COMMAND_BACKLOG_MM = 0.0
# strict 的主推进命令仍然会驱动 native support/drive window。若命令推进长期跑在
# 真实整杆位移前面，入口释放段最容易把误差积累成“头部呼吸式伸缩”。
# 这里给 strict 一个很小的 backlog 上限：允许少量前馈，但不允许支撑块长期甩开真实杆体。
ELASTICROD_STRICT_MAX_COMMAND_BACKLOG_MM = 0.0
ELASTICROD_SHEATH_LENGTH_MM = 30.0
ELASTICROD_SHEATH_STIFFNESS_N_PER_M = 420.0
ELASTICROD_SHEATH_EXIT_STIFFNESS_RATIO = 0.15
ELASTICROD_VIRTUAL_SHEATH_RELEASE_NODE_COUNT = 4
# Strict elasticrod now uses an introducer-feed push model: material still in
# the short rigid introducer is advanced kinematically and released exactly at
# the sheath exit, while the free rod stays fully dynamic.
ELASTICROD_PUSH_MODEL = 'introducer_feed'
if ELASTICROD_PUSH_MODEL not in {'boundary_penalty', 'introducer_feed'}:
    raise ValueError(f'Unsupported ELASTICROD_PUSH_MODEL: {ELASTICROD_PUSH_MODEL}')
ELASTICROD_USE_KINEMATIC_SHEATH_DRIVER = (
    ELASTICROD_PUSH_MODEL == 'introducer_feed'
    and not (ELASTICROD_STABILIZATION_MODE == 'strict' and ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER)
)
ELASTICROD_ENABLE_THRUST_LIMIT = ELASTICROD_STABILIZATION_MODE != 'strict' and not ELASTICROD_ENABLE_DISPLACEMENT_PUSH
ELASTICROD_THRUST_FORCE_N = 0.25
ELASTICROD_ENABLE_AXIAL_PATH_ASSIST = ELASTICROD_STABILIZATION_MODE != 'strict'
ELASTICROD_AXIAL_PATH_ASSIST_FORCE_N = 0.28
ELASTICROD_AXIAL_PATH_ASSIST_DEFICIT_MM = 0.40
ELASTICROD_AXIAL_PATH_ASSIST_CONTACT_FORCE_SCALE = 1.75
ELASTICROD_AXIAL_PATH_ASSIST_CONTACT_MIN_SCALE = 0.55
ELASTICROD_AXIAL_PATH_ASSIST_WINDOW_BACK_MM = 28.0
ELASTICROD_AXIAL_PATH_ASSIST_WINDOW_FRONT_MM = 6.0
ELASTICROD_AXIAL_PATH_ASSIST_WINDOW_MIN_NODE_COUNT = 2
ELASTICROD_AXIAL_PATH_ASSIST_MAX_PUSH_SCALE = 2.00
ELASTICROD_ENABLE_FIELD_GRADIENT = False
ELASTICROD_AXIAL_DRIVE_NODE_COUNT = None
# 运动优先默认把 axial drive 覆盖到整个 native support block；如果调用方显式指定，
# 仍然尊重外部覆盖值。
ELASTICROD_AXIAL_DRIVE_NODE_COUNT = None
ELASTICROD_MATERIAL_PRESET = 'niti_segmented_soft_tip'
if ELASTICROD_MATERIAL_PRESET not in {'niti_segmented_soft_tip'}:
    raise ValueError(f'Unsupported ELASTICROD_MATERIAL_PRESET: {ELASTICROD_MATERIAL_PRESET}')

# C++ 插件参数头里的默认物理量是导丝本体更可信的真值源。
# 当前 Python beam 过渡场景继续保留 option.txt 的时间步、接触阈值和中心线等运行参数，
# 但机械参数必须回到这组高刚度量级，否则导丝会表现得像极软塑料丝而不是金属导丝。
PLUGIN_DEFAULT_RHO = 20000.0
PLUGIN_DEFAULT_ROD_RADIUS_MM = 0.18
PLUGIN_DEFAULT_DT_S = 0.005
PLUGIN_DEFAULT_YOUNG_HEAD_PA = 6.0e9
PLUGIN_DEFAULT_YOUNG_BODY_PA = 7.0e10
PLUGIN_DEFAULT_SHEAR_HEAD_PA = 2.3e9
PLUGIN_DEFAULT_SHEAR_BODY_PA = 2.63e10


@dataclass(frozen=True)
class OptionParameters:
    """
    `option.txt` 是本场景的唯一数值真值源。

    这里保留两套视图：
    1. `*_si` / 原始字段：按 `option.txt` 的 SI 制读取，不在这里偷偷改单位。
    2. `*_mm` / 场景字段：仅把长度相关量换算到当前 SOFA Python 场景使用的毫米制。

    `elasticRod.cpp` 的分段语义仍然保留：
    - `youngM`      -> 磁性头段杨氏模量
    - `noMagYoungM` -> 非磁主体杨氏模量
    - 最后 `5` 条边始终视为磁性边

    `option.txt` 没有显式给 `shearM / noMagShearM`，因此严格按
    `G = E / (2 * (1 + Poisson))` 推导，并在启动日志中打印出来。
    """

    raw: dict[str, Any]
    rod_radius_m: float
    rod_length_m: float
    num_vertices: int
    young_head_pa: float
    young_body_pa: float
    poisson: float
    density: float
    dt_s: float
    speed_m_s: float
    contact_distance_m: float
    alarm_distance_m: float
    friction_mu: float
    tol: float
    max_iter: int
    gravity_m_s2: np.ndarray
    ba_vector_ref: np.ndarray
    br_vector: np.ndarray
    mu_zero: float
    tube_radius_m: float
    magnetic_edge_count: int

    @property
    def rod_radius_mm(self) -> float:
        return 1.0e3 * self.rod_radius_m

    @property
    def rod_length_mm(self) -> float:
        return 1.0e3 * self.rod_length_m

    @property
    def speed_mm_s(self) -> float:
        return 1.0e3 * self.speed_m_s

    @property
    def contact_distance_mm(self) -> float:
        return 1.0e3 * self.contact_distance_m

    @property
    def alarm_distance_mm(self) -> float:
        return 1.0e3 * self.alarm_distance_m

    @property
    def gravity_mm_s2(self) -> np.ndarray:
        return 1.0e3 * self.gravity_m_s2

    @property
    def tube_radius_mm(self) -> float:
        return 1.0e3 * self.tube_radius_m

    @property
    def shear_head_pa(self) -> float:
        return self.young_head_pa / (2.0 * (1.0 + self.poisson))

    @property
    def shear_body_pa(self) -> float:
        return self.young_body_pa / (2.0 * (1.0 + self.poisson))


DEFAULT_INSERTION_DIR = np.array([0.0, 0.0, 1.0], dtype=float)


def _parse_option_txt(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'option.txt not found: {path}')
    values: dict[str, Any] = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
        for raw_line in handle:
            line = raw_line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            key, tokens = parts[0], parts[1:]
            if not tokens:
                continue
            if len(tokens) == 1:
                token = tokens[0]
                try:
                    number = float(token)
                except ValueError:
                    values[key] = token
                    continue
                if number.is_integer() and 'e' not in token.lower() and '.' not in token:
                    values[key] = int(number)
                else:
                    values[key] = number
                continue
            parsed: list[Any] = []
            for token in tokens:
                try:
                    parsed.append(float(token))
                except ValueError:
                    parsed.append(token)
            values[key] = parsed
    return values


def _scalar(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f'option.txt key {key} must be scalar, got {value!r}')
    return float(value)


def _integer(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f'option.txt key {key} must be integer, got {value!r}')
    return int(value)


def _vec3(data: dict[str, Any], key: str, default: tuple[float, float, float]) -> np.ndarray:
    value = data.get(key, default)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f'option.txt key {key} must contain 3 numbers, got {value!r}')
    return arr.astype(float)


_OPTION_RAW = _parse_option_txt(OPTION_TXT)
OPTION_PARAMETERS = OptionParameters(
    raw=_OPTION_RAW,
    rod_radius_m=_scalar(_OPTION_RAW, 'rodRadius', 0.25e-3),
    rod_length_m=_scalar(_OPTION_RAW, 'RodLength', 250e-3),
    num_vertices=_integer(_OPTION_RAW, 'numVertices', 201),
    young_head_pa=_scalar(_OPTION_RAW, 'youngM', 4.0e6),
    young_body_pa=_scalar(_OPTION_RAW, 'noMagYoungM', 50.0e6),
    poisson=_scalar(_OPTION_RAW, 'Poisson', 0.5),
    density=_scalar(_OPTION_RAW, 'density', 2400.0),
    dt_s=_scalar(_OPTION_RAW, 'deltaTime', 3.0e-3),
    speed_m_s=_scalar(_OPTION_RAW, 'speed', 1.0e-3),
    contact_distance_m=_scalar(_OPTION_RAW, 'col_limit', 0.1e-3),
    alarm_distance_m=2.0 * _scalar(_OPTION_RAW, 'col_limit', 0.1e-3),
    friction_mu=_scalar(_OPTION_RAW, 'mu', 0.05),
    tol=_scalar(_OPTION_RAW, 'tol', 1.0e-3),
    max_iter=_integer(_OPTION_RAW, 'maxIter', 101),
    gravity_m_s2=_vec3(_OPTION_RAW, 'gVector', (0.0, 0.0, 0.0)),
    ba_vector_ref=_vec3(_OPTION_RAW, 'baVector', (0.0, 0.0, 0.0)),
    br_vector=_vec3(_OPTION_RAW, 'brVector', (128.0e3, 0.0, 0.0)),
    mu_zero=_scalar(_OPTION_RAW, 'muZero', 1.0),
    tube_radius_m=_scalar(_OPTION_RAW, 'tubeRadius', 4.0e-3),
    magnetic_edge_count=5,
)

ELASTICROD_CONTACT_OUTER_RADIUS_MM = 0.35
ELASTICROD_MECHANICAL_CORE_RADIUS_MM = 0.20
ELASTICROD_MATERIAL_PROFILE = 'niti_segmented_soft_tip'
if ELASTICROD_MATERIAL_PROFILE not in {'niti_segmented_soft_tip'}:
    raise ValueError(f'Unsupported ELASTICROD_MATERIAL_PROFILE: {ELASTICROD_MATERIAL_PROFILE}')
ELASTICROD_BODY_YOUNG_PA = 76.0e9  # Slightly stiffer shaft so proximal push reaches the first bend instead of folding into the distal segment.
ELASTICROD_BODY_POISSON = 0.33
ELASTICROD_BODY_DENSITY_KG_M3 = 6500.0
ELASTICROD_HEAD_EFFECTIVE_YOUNG_PA = 8.00e9  # Keep the distal tip softer than the shaft while preserving a coherent metal-like torsional response.
ELASTICROD_DISTAL_SOFT_LENGTH_MM = 13.0
# The distal magnetic section should bend more easily than the shaft, but it
# must stay torsionally coherent like a metal guidewire tip instead of letting
# each reduced rod node visibly twist on its own.
ELASTICROD_HEAD_EFFECTIVE_SHEAR_SCALE = 9.00
ELASTICROD_HEAD_EFFECTIVE_SHEAR_PA = (
    ELASTICROD_HEAD_EFFECTIVE_SHEAR_SCALE
    * ELASTICROD_HEAD_EFFECTIVE_YOUNG_PA
    / (2.0 * (1.0 + ELASTICROD_BODY_POISSON))
)
ELASTICROD_BODY_SHEAR_PA = ELASTICROD_BODY_YOUNG_PA / (2.0 * (1.0 + ELASTICROD_BODY_POISSON))
ELASTICROD_MAGNETIC_CORE_RADIUS_MM = ELASTICROD_MECHANICAL_CORE_RADIUS_MM


DT = OPTION_PARAMETERS.dt_s
ROOT_GRAVITY = OPTION_PARAMETERS.gravity_mm_s2
# 用户明确希望把导丝半径再放大一点；这里在 option.txt 与插件默认值之上取一个温和的放大量。
# 更大的半径会同时提升接触厚度和截面惯性矩，能显著抑制入口受压后的欧拉屈曲。
WIRE_RADIUS_MM = max(OPTION_PARAMETERS.rod_radius_mm, PLUGIN_DEFAULT_ROD_RADIUS_MM, 0.32)
WIRE_TOTAL_LENGTH_MM = OPTION_PARAMETERS.rod_length_mm
WIRE_NODE_COUNT = OPTION_PARAMETERS.num_vertices
WIRE_N_SEGMENTS = OPTION_PARAMETERS.num_vertices - 1
BEAM_RUNTIME_IS_RECORDING = BEAM_RUNTIME_PROFILE == 'recording_10min'
BEAM_REALTIME_NODE_COUNT = 101
BEAM_REALTIME_PUSH_SPEED_SCALE = 2.0
BEAM_BASE_PUSH_SPEED_MM_S = 10.0
BEAM_ACTIVE_NODE_COUNT = (
    min(WIRE_NODE_COUNT, BEAM_REALTIME_NODE_COUNT)
    if BEAM_RUNTIME_IS_RECORDING
    else WIRE_NODE_COUNT
)
BEAM_ACTIVE_DT_S = 1.5e-2 if BEAM_RUNTIME_IS_RECORDING else DT
BEAM_ACTIVE_PUSH_SPEED_MM_S = (
    BEAM_BASE_PUSH_SPEED_MM_S * BEAM_REALTIME_PUSH_SPEED_SCALE
    if BEAM_RUNTIME_IS_RECORDING
    else BEAM_BASE_PUSH_SPEED_MM_S
)
BEAM_VESSEL_QUERY_FACE_CANDIDATE_COUNT = 128 if BEAM_RUNTIME_IS_RECORDING else 1024
BEAM_SURFACE_REFRESH_NEAR_STEPS = 4 if BEAM_RUNTIME_IS_RECORDING else 2
BEAM_SURFACE_REFRESH_FAR_STEPS = 16 if BEAM_RUNTIME_IS_RECORDING else 10
BEAM_HEAD_CLEARANCE_REFRESH_NEAR_STEPS = 2 if BEAM_RUNTIME_IS_RECORDING else 1
BEAM_HEAD_CLEARANCE_REFRESH_FAR_STEPS = 8 if BEAM_RUNTIME_IS_RECORDING else 1
ELASTICROD_RUNTIME_IS_REALTIME = ELASTICROD_RUNTIME_PROFILE == 'realtime_gui_10min'
ELASTICROD_RUNTIME_GUI_WALLCLOCK_LAUNCH = _is_elasticrod_gui_wallclock_launch()
ELASTICROD_RUNTIME_GUI_NODE_BUDGET = (
    ELASTICROD_RUNTIME_IS_REALTIME
    and ELASTICROD_STABILIZATION_MODE == 'strict'
    and ELASTICROD_RUNTIME_GUI_WALLCLOCK_LAUNCH
)
ELASTICROD_ACTIVE_NODE_COUNT = (
    ELASTICROD_GUI_NODE_COUNT
    if ELASTICROD_RUNTIME_GUI_NODE_BUDGET
    else (ELASTICROD_REALTIME_NODE_COUNT if ELASTICROD_RUNTIME_IS_REALTIME else WIRE_NODE_COUNT)
)
ELASTICROD_WIRE_TOTAL_LENGTH_MM = 400.0
WIRE_MASS_DENSITY = PLUGIN_DEFAULT_RHO
# 用插件头文件对应的 E/G 反推，可得到更合理的泊松比约 0.33；
# 比 option.txt 里的 0.5 更接近当前高刚度金属导丝分段参数。
WIRE_POISSON = 0.33
WIRE_HEAD_YOUNG_MODULUS_PA = PLUGIN_DEFAULT_YOUNG_HEAD_PA
WIRE_BODY_YOUNG_MODULUS_PA = PLUGIN_DEFAULT_YOUNG_BODY_PA
WIRE_HEAD_SHEAR_MODULUS_PA = PLUGIN_DEFAULT_SHEAR_HEAD_PA
WIRE_BODY_SHEAR_MODULUS_PA = PLUGIN_DEFAULT_SHEAR_BODY_PA
NATIVE_WIRE_RADIUS_MM = float(ELASTICROD_CONTACT_OUTER_RADIUS_MM)
NATIVE_WIRE_MECHANICAL_CORE_RADIUS_MM = float(ELASTICROD_MECHANICAL_CORE_RADIUS_MM)
NATIVE_WIRE_TOTAL_LENGTH_MM = ELASTICROD_WIRE_TOTAL_LENGTH_MM
NATIVE_WIRE_MASS_DENSITY = float(ELASTICROD_BODY_DENSITY_KG_M3)
NATIVE_WIRE_HEAD_YOUNG_MODULUS_PA = float(ELASTICROD_HEAD_EFFECTIVE_YOUNG_PA)
NATIVE_WIRE_BODY_YOUNG_MODULUS_PA = float(ELASTICROD_BODY_YOUNG_PA)
NATIVE_WIRE_HEAD_SHEAR_MODULUS_PA = float(ELASTICROD_HEAD_EFFECTIVE_SHEAR_PA)
NATIVE_WIRE_BODY_SHEAR_MODULUS_PA = float(ELASTICROD_BODY_SHEAR_PA)
ELASTICROD_REFERENCE_SPACING_MM = OPTION_PARAMETERS.rod_length_mm / max(ELASTICROD_ACTIVE_NODE_COUNT - 1, 1)
if ELASTICROD_RUNTIME_IS_REALTIME:
    # Realtime elasticrod should honor the actual runtime node budget directly.
    # Preserving the 250 mm option spacing while extending to a 400 mm wire silently
    # inflated the native count from ~100 to 161 nodes, which dominated wall-clock cost.
    NATIVE_WIRE_NODE_COUNT = max(2, int(ELASTICROD_ACTIVE_NODE_COUNT))
else:
    NATIVE_WIRE_NODE_COUNT = max(
        2,
        int(math.ceil(NATIVE_WIRE_TOTAL_LENGTH_MM / max(ELASTICROD_REFERENCE_SPACING_MM, 1.0e-6))) + 1,
    )
NATIVE_WIRE_SPACING_MM = NATIVE_WIRE_TOTAL_LENGTH_MM / max(NATIVE_WIRE_NODE_COUNT - 1, 1)
NATIVE_WIRE_SOFT_TIP_EDGE_COUNT = max(
    1,
    min(
        NATIVE_WIRE_NODE_COUNT - 1,
        int(math.ceil(ELASTICROD_DISTAL_SOFT_LENGTH_MM / max(NATIVE_WIRE_SPACING_MM, 1.0e-6))),
    ),
)
ELASTICROD_STRICT_SUPPORT_RELEASE_MM = min(
    float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_LENGTH_MM),
    max(
        float(ELASTICROD_STRICT_SUPPORT_RELEASE_MM),
        2.0 * float(NATIVE_WIRE_SPACING_MM),
        1.0e-6,
    ),
)
ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM = (
    float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_LENGTH_MM)
    if ELASTICROD_STABILIZATION_MODE == 'strict'
    else (
        max(
            float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_LENGTH_MM),
            float(ELASTICROD_INTRODUCER_LENGTH_MM),
        )
        if ELASTICROD_ENABLE_INTRODUCER
        else 0.0
    )
)
ELASTICROD_STRICT_SUPPORT_WINDOW_LENGTH_MM = ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM
ELASTICROD_STRICT_DRIVE_WINDOW_LENGTH_MM = ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM
ELASTICROD_STRICT_DRIVE_WINDOW_MIN_NODE_COUNT = max(
    2,
    min(
        int(ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT),
        int(math.ceil(ELASTICROD_STRICT_DRIVE_WINDOW_LENGTH_MM / max(NATIVE_WIRE_SPACING_MM, 1.0e-6))) + 1,
    ),
)
ELASTICROD_STRICT_BARRIER_ENTRY_EXTENSION_MM = ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM

CONTACT_DISTANCE_MM = OPTION_PARAMETERS.contact_distance_mm
CONTACT_ALARM_DISTANCE_MM = OPTION_PARAMETERS.alarm_distance_mm
# 之前为了避免卡顿，这里把摩擦系数强行压到了 0.01，数值上过于“放滑”。
# 现在恢复到 option.txt 的 0.05：
# - 这更接近文献里“血液/盐水润滑下，血管内器械-血管壁”常见的 10^-2 到 10^-1 量级；
# - 对当前这套“beam + 刚性血管表面 + 无显式流体润滑膜”的场景来说，
#   0.05 是一个更正常的基线值；
# - 若后续切到更真实的涂层导丝 + 流体/润滑模型，再考虑往 0.02~0.03 调。
CONTACT_FRICTION_MU = float(OPTION_PARAMETERS.friction_mu)
CONSTRAINT_SOLVER_TOLERANCE = OPTION_PARAMETERS.tol
CONSTRAINT_SOLVER_MAX_ITER = OPTION_PARAMETERS.max_iter
ELASTICROD_CONTACT_FRICTION_MU = 0.012
ELASTICROD_CONTACT_DISTANCE_MM = (
    max(0.60 * NATIVE_WIRE_RADIUS_MM, 0.21)
    if ELASTICROD_STABILIZATION_MODE == 'strict'
    else max(0.25 * NATIVE_WIRE_RADIUS_MM, 0.02)
)
ELASTICROD_CONTACT_ALARM_DISTANCE_MM = (
    max(1.20 * NATIVE_WIRE_RADIUS_MM, ELASTICROD_CONTACT_DISTANCE_MM + 0.24)
    if ELASTICROD_STABILIZATION_MODE == 'strict'
    else max(0.75 * NATIVE_WIRE_RADIUS_MM, ELASTICROD_CONTACT_DISTANCE_MM + 0.05)
)
ELASTICROD_CONSTRAINT_SOLVER_TOLERANCE = (
    1.0e-5  # Tightened from 2e-4 for better contact constraint solving
    if (
        ELASTICROD_STABILIZATION_MODE == 'strict'
        and ELASTICROD_RUNTIME_PROFILE == 'realtime_gui_10min'
    )
    else 5.0e-6
)
ELASTICROD_CONSTRAINT_SOLVER_MAX_ITER = (
    150  # Increased from 48 for better contact constraint solving
    if (
        ELASTICROD_STABILIZATION_MODE == 'strict'
        and ELASTICROD_RUNTIME_PROFILE == 'realtime_gui_10min'
    )
    else 500
)
ELASTICROD_DT_S = 1.0e-4
ELASTICROD_ACTIVE_DT_S = ELASTICROD_REALTIME_DT_FREE_S if ELASTICROD_RUNTIME_IS_REALTIME else ELASTICROD_DT_S
ELASTICROD_ACTIVE_SOLVER_MAX_ITER = (
    (
        ELASTICROD_GUI_SOLVER_MAX_ITER_FREE
        if ELASTICROD_RUNTIME_GUI_NODE_BUDGET
        else ELASTICROD_REALTIME_SOLVER_MAX_ITER_FREE
    )
    if ELASTICROD_RUNTIME_IS_REALTIME
    else ELASTICROD_CONSTRAINT_SOLVER_MAX_ITER
)
ELASTICROD_ACTIVE_SOLVER_TOLERANCE = (
    (
        ELASTICROD_GUI_SOLVER_TOL_FREE
        if ELASTICROD_RUNTIME_GUI_NODE_BUDGET
        else ELASTICROD_REALTIME_SOLVER_TOL_FREE
    )
    if ELASTICROD_RUNTIME_IS_REALTIME
    else ELASTICROD_CONSTRAINT_SOLVER_TOLERANCE
)
ELASTICROD_GUIDEWIRE_CONTACT_STIFFNESS = 820.0 if ELASTICROD_STABILIZATION_MODE == 'strict' else 180.0
# 物理上 introducer 内壁和导丝之间可以很小，但 SOFA 接触还会额外包一层
# alarm/contact 缓冲。如果几何净间隙小于这层缓冲，strict 一开始就会处在
# “预接触/预压缩”状态里，首帧直接炸掉。这里显式把 introducer 的有效净间隙
# 拉到大于 native 接触缓冲的量级。
ELASTICROD_INTRODUCER_CLEARANCE_MM = max(
    ELASTICROD_INTRODUCER_PHYSICAL_CLEARANCE_MM,
    ELASTICROD_CONTACT_ALARM_DISTANCE_MM + 0.05,
    0.25,
)
GUIDEWIRE_CONTACT_STIFFNESS = 100.0
# 录屏版 beam 不开启自碰撞，避免在入口附近出现不自然的自卡和折线感。
BEAM_ENABLE_SELF_COLLISION = False

NAVIGATION_MODE = 2
# 防穿壁兜底需要保留，否则 beam + 细杆接触在大步长下仍可能从三角网格间隙漏出去。
# 录屏版 beam 仍保留一段很短的体外入口直线导向，让近端在进入开口前不要先折起来。
ENABLE_LUMEN_SAFETY_PROJECTION = True
ENABLE_VIRTUAL_SHEATH = True

# `beam` 恢复到录视频用的稳定运动学送丝路径：
# - 近端短段沿入口轴线前送；
# - 入口外仍保留访问走廊/短鞘管约束；
# - 不把整根 beam 切回最近这轮的“尾端恒力推送”实验语义。
BEAM_USE_KINEMATIC_INSERTION = True
BEAM_ENABLE_LEGACY_DRIVE_CONSTRAINT = False
# 这不是“长虚拟鞘管”，而是一个很短的入口外访问走廊：
# 只在血管入口外的近端几毫米范围内，把导丝限制在真实插入轴附近，
# 目的是防止体外段从血管侧壁穿入，而不是在血管内继续支撑导丝。
BEAM_PRE_ENTRY_ACCESS_GUIDE_MM = 12.0
# 用户当前明确要求用 1 cm/s 作为送丝速度，因此这里继续保留 10 mm/s。
PUSH_FORCE_TARGET_SPEED_MM_S = 13.0
# 录屏版 beam 使用短 proximal drive block，而不是只拽住单个根节点。
PUSH_FORCE_NODE_COUNT = 4
# `elasticrod` 运动优先路径把 30 mm virtual sheath 整段都作为 native support block：
# - 前 4 个节点由 kinematic introducer 直接导向；
# - 后面仅保留很短的 release 过渡，再接回自由杆段；
# - 这样既保留长 sheath 的送丝支撑，又不会把整段都硬焊成刚体。
ELASTICROD_PUSH_NODE_COUNT = (
    0
    if (ELASTICROD_STABILIZATION_MODE == 'strict' and ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER)
    else max(
        2,
        min(
            ELASTICROD_ACTIVE_NODE_COUNT,
            (
                int(
                    math.ceil(
                        max(
                            ELASTICROD_SHEATH_LENGTH_MM if ELASTICROD_ENABLE_VIRTUAL_SHEATH else ELASTICROD_INTRODUCER_LENGTH_MM,
                            NATIVE_WIRE_SPACING_MM,
                        )
                        / max(NATIVE_WIRE_SPACING_MM, 1.0e-6)
                    )
                )
                + (3 if ELASTICROD_USE_KINEMATIC_SHEATH_DRIVER else 1)
            ),
        ),
    )
)
if ELASTICROD_AXIAL_DRIVE_NODE_COUNT is None:
    ELASTICROD_AXIAL_DRIVE_NODE_COUNT = (
        0
        if (ELASTICROD_STABILIZATION_MODE == 'strict' and ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER)
        else (
            ELASTICROD_PUSH_NODE_COUNT
            if ELASTICROD_USE_KINEMATIC_SHEATH_DRIVER
            else max(
                2,
                min(
                    ELASTICROD_PUSH_NODE_COUNT,
                    int(
                        math.ceil(
                            max(ELASTICROD_INTRODUCER_LENGTH_MM, NATIVE_WIRE_SPACING_MM)
                            / max(NATIVE_WIRE_SPACING_MM, 1.0e-6)
                        )
                    )
                    + 1,
                ),
            )
        )
    )
ELASTICROD_PROXIMAL_AXIAL_STIFFNESS_N_PER_M = (
    0.0
    if (ELASTICROD_STABILIZATION_MODE == 'strict' and ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER)
    else (
        8000.0
        if (ELASTICROD_STABILIZATION_MODE == 'strict' and ELASTICROD_PUSH_MODEL == 'introducer_feed')
        else (
            8000.0
            if ELASTICROD_STABILIZATION_MODE == 'strict'
            else (
                760.0
                if ELASTICROD_USE_KINEMATIC_SHEATH_DRIVER
                else 120.0
            )
        )
    )
)
ELASTICROD_PROXIMAL_LATERAL_STIFFNESS_N_PER_M = (
    0.0
    if (ELASTICROD_STABILIZATION_MODE == 'strict' and ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER)
    else (
        4000.0
        if (ELASTICROD_STABILIZATION_MODE == 'strict' and ELASTICROD_PUSH_MODEL == 'introducer_feed')
        else (
            260.0
            if ELASTICROD_USE_KINEMATIC_SHEATH_DRIVER
            else (80.0 if ELASTICROD_ENABLE_VIRTUAL_SHEATH else 0.0)
        )
    )
)
ELASTICROD_PROXIMAL_ANGULAR_STIFFNESS_NM_PER_RAD = (
    0.0
    if (ELASTICROD_STABILIZATION_MODE == 'strict' and ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER)
    else 1.0e-3
)
ELASTICROD_PROXIMAL_LINEAR_DAMPING_N_S_PER_M = (
    0.0
    if (ELASTICROD_STABILIZATION_MODE == 'strict' and ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER)
    else (
        0.34
        if ELASTICROD_USE_KINEMATIC_SHEATH_DRIVER
        else (0.0 if ELASTICROD_STABILIZATION_MODE == 'strict' else 0.18)
    )
)
# `strict` 当前的主炸点已经定位到近端角阻尼项：
# 它在 `Rigid3d` 全姿态自由度上的隐式线性化与 rod 的真实 twist 语义不一致，
# 会在预接触阶段给系统注入数值能量。修到 twist-consistent damping 之前，
# strict 默认把这项关掉。
ELASTICROD_PROXIMAL_ANGULAR_DAMPING_NM_S_PER_RAD = (
    0.0 if ELASTICROD_STABILIZATION_MODE == 'strict' else 0.18
)
# 近端 support block 和固定参考态修正后，strict 路径在高刚度金属参数下
# 必须依赖 bend/twist 的隐式切线才能把 free-space / pre-contact 阶段压回稳定区。
ELASTICROD_USE_IMPLICIT_BEND_TWIST = True
ELASTICROD_ACTIVE_STARTUP_RAMP_TIME_S = (
    ELASTICROD_REALTIME_STARTUP_RAMP_S if ELASTICROD_RUNTIME_IS_REALTIME else ELASTICROD_STARTUP_RAMP_TIME_S
)
# 过大的世界系分布阻尼会把 native rod 近似“钉”在原地，推进量传不过去就只会在近端积压。
# strict 默认收回到较小量级，主要负责压高频数值振动，不再承担主稳定器职责。
ELASTICROD_DISTRIBUTED_TRANSLATIONAL_DAMPING_N_S_PER_M = 0.22 if ELASTICROD_STABILIZATION_MODE == 'strict' else 0.18
ELASTICROD_DISTRIBUTED_TWIST_DAMPING_NM_S_PER_RAD = 0.120 if ELASTICROD_STABILIZATION_MODE == 'strict' else 0.300
PUSH_FORCE_REDUCED_SCALE_ON_WALL = 0.40
PUSH_FORCE_REDUCED_SCALE_ON_STEERING = 0.30
PUSH_FORCE_REDUCED_SCALE_ON_STALL = 0.30
# `beam` 后端的推进是运动学送丝，不是真正的力平衡。
# 因此一旦近端命令送入量明显大于尖端真实前进量，就必须主动降速，
# 否则多余长度只能在入口和近端弯折段里堆积，最后表现为屈曲/团丝。
PUSH_FORCE_REDUCED_SCALE_ON_COMPRESSION = 1.00
PUSH_FORCE_SCALE_DROP_TIME_S = 0.02
PUSH_FORCE_SCALE_RISE_TIME_S = 0.06
PUSH_FORCE_INITIAL_TOTAL = 0.08
PUSH_FORCE_MIN_TOTAL = 0.002
PUSH_FORCE_MAX_TOTAL = 4.00
PUSH_FORCE_CALIBRATION_TIME_S = 3.0
PUSH_FORCE_CALIBRATION_REGION_MM = 30.0
PUSH_FORCE_CALIBRATION_ALPHA = 0.25
PUSH_FORCE_CALIBRATION_RAMP_PER_S = 0.80
PUSH_FORCE_CALIBRATION_LOCK_NEG_SPEED_MM_S = -0.25
PUSH_FORCE_CALIBRATION_LOCK_COMPRESSION_MM = 1.0
TIP_SPEED_FILTER_ALPHA = 0.25

MAGNETIC_HEAD_EDGES = OPTION_PARAMETERS.magnetic_edge_count
DISTAL_VISUAL_NODE_COUNT = MAGNETIC_HEAD_EDGES + 1
DISTAL_FORCE_NODE_COUNT = MAGNETIC_HEAD_EDGES + 1
# Keep the native magnetic tip length tied to the original DER physical tip
# length instead of to a fixed edge count. Once the native rod was extended to
# The user clarified that the distal magnetic head is a softer NdFeB+PDMS
# composite segment, not a point magnet at the extreme tip. Keep the magnetic
# head shorter than the whole rod, but let it cover most of the soft distal
# segment so the steering torque and lateral pull act over a real composite head
# instead of only the last ~16 mm.
ELASTICROD_MAGNETIC_HEAD_LENGTH_MM = 24.0
ELASTICROD_MAGNETIC_HEAD_EDGES = max(
    1,
    min(
        NATIVE_WIRE_NODE_COUNT - 1,
        int(math.ceil(ELASTICROD_MAGNETIC_HEAD_LENGTH_MM / max(NATIVE_WIRE_SPACING_MM, 1.0e-6))),
    ),
)
ELASTICROD_DISTAL_VISUAL_NODE_COUNT = ELASTICROD_MAGNETIC_HEAD_EDGES + 1
# `beam` 和 `elasticrod` 对磁场量级的可承受范围不同：
# - `beam` 后端需要更强的等效磁驱动，才能在接触和低阶梁离散下看出明显转向；
# - `elasticrod` 后端现在改成了更物理的 SI 单位链，若继续用 4x 放大后的 `brVector`，
#   在贴壁瞬间会过强，因此单独保留较小倍率。
BEAM_MAGNETIC_STRENGTH_SCALE = 4.0
ELASTICROD_MAGNETIC_STRENGTH_SCALE = 2.40
# strict 只保留真实 magnetic torque，不再强行把场方向扭到最小 torque-sine 以上。
ELASTICROD_NATIVE_MAGNETIC_MIN_TORQUE_SIN = 0.0 if ELASTICROD_STABILIZATION_MODE == 'strict' else 0.22
# Keep the reduced strict path on physical torque-only steering, but do not let
# the controller cold-start the field exactly when the head first reaches the
# bend. The release logic is warmed separately in the Python controller.
ELASTICROD_STRICT_PHYSICAL_TORQUE_ONLY = True
MAGNETIC_STRENGTH_SCALE = BEAM_MAGNETIC_STRENGTH_SCALE
MAGNETIC_BR_VECTOR = BEAM_MAGNETIC_STRENGTH_SCALE * OPTION_PARAMETERS.br_vector.astype(float)
_elasticrod_native_br_vector = OPTION_PARAMETERS.br_vector.astype(float)
if ELASTICROD_STABILIZATION_MODE == 'strict':
    # Keep the strict magnetic remanence transverse so the applied field can
    # create a real bending couple on the distal magnetic segment. The native
    # DER frame handedness is opposite to the user-facing steering convention
    # on this route set, so flip the sign to bend toward the forward target.
    _elasticrod_native_br_vector = -_elasticrod_native_br_vector
ELASTICROD_MAGNETIC_BR_VECTOR = ELASTICROD_MAGNETIC_STRENGTH_SCALE * _elasticrod_native_br_vector
MAGNETIC_BA_VECTOR_REF = OPTION_PARAMETERS.ba_vector_ref.astype(float)
MAGNETIC_MU_ZERO = OPTION_PARAMETERS.mu_zero

INITIAL_TIP_ARC_MM = 12.0
INITIAL_ENTRY_ADVANCE_MM = 2.5
# `beam` 要回到录视频时那种短入口支撑：
# 鞘管只在入口附近提供一个很短的直线导向，避免把导丝“拖着走太久”。
# 真正关键不是把鞘管做长，而是让被约束的节点严格落在入口直线上。
VIRTUAL_SHEATH_RELEASE_S_MM = 0.0
VIRTUAL_SHEATH_BLEND_OUT_MM = 1.0
VIRTUAL_SHEATH_BACKSLIP_TOL_MM = 0.02
# 压缩保护的滞回阈值：
# - 进入阈值稍大，避免把正常的小幅几何误差误判成屈曲；
# - 退出阈值更小，避免保护开关抖动。
BEAM_COMPRESSION_ENTER_MM = 2.50
BEAM_COMPRESSION_EXIT_MM = 1.00
BEAM_STALL_COMPRESSION_ENTER_MM = 1.20
BEAM_STALL_COMPRESSION_EXIT_MM = 0.40
BEAM_STALL_SPEED_ENTER_MM_S = 0.80
BEAM_STALL_SPEED_EXIT_MM_S = 2.50
MIN_SEGMENT_LENGTH_RATIO = 0.97
MAX_SEGMENT_LENGTH_RATIO = 1.03
LUMEN_CONSTRAINT_TOLERANCE_MM = 0.06
# 这里是“安全投影兜底”的边界裕量，不是导丝真实半径。
# 碰撞厚度已经由 CollisionModel 负责；如果这里再减去整根导丝半径，
# 就会把可用腔径缩得过小，导致节点被过早拉回并显著拖慢推进。
LUMEN_CLEARANCE_MM = max(CONTACT_DISTANCE_MM, 0.10)
CENTERLINE_LUMEN_NEAREST_K = 96
CENTERLINE_LUMEN_RADIUS_PERCENTILE = 16.0
TIP_WALL_CONTACT_ENTER_MM = 0.15
TIP_WALL_CONTACT_EXIT_MM = 0.30
ELASTICROD_STRICT_TIP_WALL_CONTACT_ENTER_MM = max(
    TIP_WALL_CONTACT_ENTER_MM,
    ELASTICROD_CONTACT_DISTANCE_MM + 0.04,
    0.18,
)
ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM = max(
    TIP_WALL_CONTACT_EXIT_MM,
    ELASTICROD_CONTACT_DISTANCE_MM + 0.08,
    0.55,
)
# Strict 首弯里经常出现“已经转成稳定贴壁滑行，但 wallContact 因退出阈值过高
# 长时间卡死”的情况。缩短 release hold，避免 runtime 长时间被锁在 contact band。
ELASTICROD_STRICT_TIP_WALL_CONTACT_RELEASE_HOLD_STEPS = 8
WALL_CONTACT_TIP_PROBE_NODES = 3
DEBUG_PRINT_EVERY = 100

TARGET_MARKER_SIZE_MM = 0.70
MAGNETIC_FORCE_ARROW_LENGTH_MM = 13.0
MAGNETIC_FORCE_ARROW_HEAD_LENGTH_MM = 4.2
MAGNETIC_FORCE_ARROW_HEAD_WIDTH_MM = 2.4
MAGNETIC_FORCE_ARROW_ANCHOR_REL = np.array([0.78, 0.84, 0.58], dtype=float)
DEFAULT_CAMERA_ROUTE_FOCUS_FRACTION = 0.52
DEFAULT_CAMERA_ROUTE_FOCUS_WEIGHT = 0.72
CAMERA_FOLLOW_DIRECTION = np.array([0.22, -0.96, 0.30], dtype=float)
CAMERA_FOLLOW_YAW_DEG = 270.0
CAMERA_FOLLOW_DISTANCE_SCALE = 0.24
CAMERA_FOLLOW_DISTANCE_MIN_MM = 62.0
CAMERA_FOLLOW_DISTANCE_MAX_MM = 118.0
MAGNETIC_LOOKAHEAD_DISTANCE_MM = 15.0  # Increased from 6.0 for faster target point updates
MAGNETIC_FIELD_SMOOTHING_ALPHA = 0.90
MAGNETIC_MAX_TURN_ANGLE_DEG = 60.0
MAGNETIC_FIELD_RAMP_TIME_S = 0.20
MAGNETIC_LATERAL_FORCE_SCALE = 0.0
ELASTICROD_MAGNETIC_LOOKAHEAD_DISTANCE_MM = (
    ELASTICROD_STRICT_MAGNETIC_LOOKAHEAD_MM
    if ELASTICROD_STABILIZATION_MODE == 'strict'
    else 9.5
)
# strict 下磁场方向只做保守 slew/ramp，避免把数值稳定性交给 release gate 或非物理节流。
ELASTICROD_MAGNETIC_FIELD_SMOOTHING_ALPHA = 0.12 if ELASTICROD_STABILIZATION_MODE == 'strict' else 0.50
ELASTICROD_MAGNETIC_MAX_TURN_ANGLE_DEG = 36.0 if ELASTICROD_STABILIZATION_MODE == 'strict' else 76.0
ELASTICROD_MAGNETIC_FIELD_RAMP_TIME_S = 0.34 if ELASTICROD_STABILIZATION_MODE == 'strict' else 0.12
ELASTICROD_MAGNETIC_LATERAL_FORCE_SCALE = 0.0 if ELASTICROD_STABILIZATION_MODE == 'strict' else 0.55
STEERING_MISALIGN_ENTER_DEG = 20.0
STEERING_MISALIGN_EXIT_DEG = 12.0

CONTACT_MANAGER_RESPONSE = 'FrictionContactConstraint'
CONTACT_MANAGER_RESPONSE_PARAMS = f'mu={CONTACT_FRICTION_MU}'
ELASTICROD_CONTACT_MANAGER_RESPONSE_PARAMS = f'mu={ELASTICROD_CONTACT_FRICTION_MU}'


# 后端分离后，beam 与 elasticrod 使用各自独立的数值参数组。
# 这里把 beam 恢复到旧录屏版本的强阻尼档，优先保证展示时不过度屈曲、弯折更顺。
BEAM_RAYLEIGH_STIFFNESS = 0.95
BEAM_RAYLEIGH_MASS = 0.95
ELASTICROD_RAYLEIGH_STIFFNESS = 1.0e-5  # Reduced for less damping
ELASTICROD_RAYLEIGH_MASS = 0.03  # Reduced from 0.05 for less damping

# beam 在插件缺失时仍允许通过 Python fallback 保持可运行。
# fallback 只负责“有磁导航能力”，不追求与原生 C++ 磁力完全同量级。
BEAM_PYTHON_MAGNETIC_FIELD_STRENGTH = MAGNETIC_STRENGTH_SCALE
BEAM_PYTHON_MAGNETIC_MOMENT = 1.0
BEAM_PYTHON_MAGNETIC_FORCE_GAIN = 0.0


def segmented_young(edge_count: int) -> list[float]:
    """
    与 `elasticRod.cpp::computeElasticStiffness()` 保持同一分段语义：
    前 `ne - magnetic_edge_count` 条边使用主体参数，最后 `magnetic_edge_count` 条边使用磁头参数。
    """
    split = max(0, int(edge_count) - MAGNETIC_HEAD_EDGES)
    return [WIRE_BODY_YOUNG_MODULUS_PA] * split + [WIRE_HEAD_YOUNG_MODULUS_PA] * (int(edge_count) - split)


def option_parameter_lines(backend_name: str | None = None) -> list[str]:
    backend_label = str(backend_name or GUIDEWIRE_BACKEND)
    return [
        f'  backend: {backend_label}',
        f'  beamRuntimeProfile: {BEAM_RUNTIME_PROFILE}',
        f'  elasticrodStabilizationMode: {ELASTICROD_STABILIZATION_MODE}',
        f'  elasticrodRuntimeProfile: {ELASTICROD_RUNTIME_PROFILE}',
        f'  elasticrodDiagnosticProfile: {ELASTICROD_DIAGNOSTIC_PROFILE}',
        '  parameter source: beam keeps its existing path; elasticrod strict uses ELASTICROD_* metal defaults while option.txt stays compat-only',
        f'  elasticrodMaterialProfile: {ELASTICROD_MATERIAL_PROFILE}',
        f'  elasticrodPushModel: {ELASTICROD_PUSH_MODEL}',
        f'  rodRadius(raw option / beam / native): {OPTION_PARAMETERS.rod_radius_m:.6e} m / {WIRE_RADIUS_MM:.3f} mm / {NATIVE_WIRE_RADIUS_MM:.3f} mm',
        f'  nativeRadius(contact/core/magnetic): {ELASTICROD_CONTACT_OUTER_RADIUS_MM:.3f} / {ELASTICROD_MECHANICAL_CORE_RADIUS_MM:.3f} / {ELASTICROD_MAGNETIC_CORE_RADIUS_MM:.3f} mm',
        f'  rodLength(raw/beam/native): {OPTION_PARAMETERS.rod_length_m:.6e} m -> {WIRE_TOTAL_LENGTH_MM:.3f} / {NATIVE_WIRE_TOTAL_LENGTH_MM:.3f} mm',
        f'  numVertices(raw/beamActive/nativeActive): {WIRE_NODE_COUNT} / {BEAM_ACTIVE_NODE_COUNT} / {NATIVE_WIRE_NODE_COUNT}',
        f'  nativeSoftTip(length/edges): {ELASTICROD_DISTAL_SOFT_LENGTH_MM:.3f} mm / {NATIVE_WIRE_SOFT_TIP_EDGE_COUNT}',
        f'  deltaTime(option / beamActive / nativeQuality / nativeActive): {OPTION_PARAMETERS.dt_s:.6f} s / {BEAM_ACTIVE_DT_S:.6f} s / {ELASTICROD_DT_S:.6f} s / {ELASTICROD_ACTIVE_DT_S:.6f} s',
        f'  speed(raw/beamActive): {OPTION_PARAMETERS.speed_m_s:.6f} m/s -> {BEAM_ACTIVE_PUSH_SPEED_MM_S:.3f} mm/s ; sharedTarget={PUSH_FORCE_TARGET_SPEED_MM_S:.3f} mm/s',
        f'  density(raw option / beam / native): {OPTION_PARAMETERS.density:.3f} / {WIRE_MASS_DENSITY:.3f} / {NATIVE_WIRE_MASS_DENSITY:.3f} kg/m^3',
        f'  Poisson(raw option / beam): {OPTION_PARAMETERS.poisson:.3f} / {WIRE_POISSON:.3f}',
        f'  young body/head(raw option): {OPTION_PARAMETERS.young_body_pa:.3e} / {OPTION_PARAMETERS.young_head_pa:.3e} Pa',
        f'  young body/head(beam / native strict): {WIRE_BODY_YOUNG_MODULUS_PA:.3e} / {WIRE_HEAD_YOUNG_MODULUS_PA:.3e} Pa ; {NATIVE_WIRE_BODY_YOUNG_MODULUS_PA:.3e} / {NATIVE_WIRE_HEAD_YOUNG_MODULUS_PA:.3e} Pa',
        f'  shear body/head(beam / native strict): {WIRE_BODY_SHEAR_MODULUS_PA:.3e} / {WIRE_HEAD_SHEAR_MODULUS_PA:.3e} Pa ; {NATIVE_WIRE_BODY_SHEAR_MODULUS_PA:.3e} / {NATIVE_WIRE_HEAD_SHEAR_MODULUS_PA:.3e} Pa',
        f'  contactDistance: {OPTION_PARAMETERS.contact_distance_m:.6e} m -> {CONTACT_DISTANCE_MM:.3f} mm',
        f'  alarmDistance: {OPTION_PARAMETERS.alarm_distance_m:.6e} m -> {CONTACT_ALARM_DISTANCE_MM:.3f} mm',
        f'  mu: {CONTACT_FRICTION_MU:.3f}',
        f'  solver tolerance / maxIter: {CONSTRAINT_SOLVER_TOLERANCE:.3e} / {CONSTRAINT_SOLVER_MAX_ITER}',
        f'  nativeContact(mu/alarm/contact/tol/maxIter): {ELASTICROD_CONTACT_FRICTION_MU:.3f} / {ELASTICROD_CONTACT_ALARM_DISTANCE_MM:.3f} mm / {ELASTICROD_CONTACT_DISTANCE_MM:.3f} mm / {ELASTICROD_CONSTRAINT_SOLVER_TOLERANCE:.3e} / {ELASTICROD_CONSTRAINT_SOLVER_MAX_ITER}',
        f'  nativeActiveSolver(tol/maxIter): {ELASTICROD_ACTIVE_SOLVER_TOLERANCE:.3e} / {ELASTICROD_ACTIVE_SOLVER_MAX_ITER}',
        f'  nativeMaterialPreset: {ELASTICROD_MATERIAL_PRESET}',
        f'  gravity: {np.round(ROOT_GRAVITY, 6).tolist()} mm/s^2',
        f'  magneticStrengthScale(beam/native): {BEAM_MAGNETIC_STRENGTH_SCALE:.2f} / {ELASTICROD_MAGNETIC_STRENGTH_SCALE:.2f}',
        f'  selfCollision(beam/native): {BEAM_ENABLE_SELF_COLLISION} / {ELASTICROD_ENABLE_SELF_COLLISION}',
        f'  brVector(beam/native): {np.round(MAGNETIC_BR_VECTOR, 6).tolist()} / {np.round(ELASTICROD_MAGNETIC_BR_VECTOR, 6).tolist()}',
        f'  baVectorRef(beam/shared): {np.round(MAGNETIC_BA_VECTOR_REF, 6).tolist()}',
        f'  muZero: {MAGNETIC_MU_ZERO:.6g}',
        f'  magneticEdgeCount(beam/native): {MAGNETIC_HEAD_EDGES} / {ELASTICROD_MAGNETIC_HEAD_EDGES}',
        f'  pushNodeCount(scene): {PUSH_FORCE_NODE_COUNT}',
        f'  nativePushNodeCount(scene): {ELASTICROD_PUSH_NODE_COUNT}',
        f'  nativeAxialDriveNodeCount(scene): {ELASTICROD_AXIAL_DRIVE_NODE_COUNT}',
        f'  nativeBoundaryStiffness(axial/lateral/angular): {ELASTICROD_PROXIMAL_AXIAL_STIFFNESS_N_PER_M:.3f} N/m / {ELASTICROD_PROXIMAL_LATERAL_STIFFNESS_N_PER_M:.3f} N/m / {ELASTICROD_PROXIMAL_ANGULAR_STIFFNESS_NM_PER_RAD:.3e} N.m/rad',
        f'  nativeBoundaryDamping(linear/angular): {ELASTICROD_PROXIMAL_LINEAR_DAMPING_N_S_PER_M:.3f} N.s/m / {ELASTICROD_PROXIMAL_ANGULAR_DAMPING_NM_S_PER_RAD:.3e} N.m.s/rad',
        f'  nativeReducedDamping(trans/twist): {ELASTICROD_DISTRIBUTED_TRANSLATIONAL_DAMPING_N_S_PER_M:.3f} N.s/m / {ELASTICROD_DISTRIBUTED_TWIST_DAMPING_NM_S_PER_RAD:.3e} N.m.s/rad',
        f'  nativeUseImplicitBendTwist: {ELASTICROD_USE_IMPLICIT_BEND_TWIST}',
        f'  nativeContactStiffness: {ELASTICROD_GUIDEWIRE_CONTACT_STIFFNESS:.3f}',
        f'  introducer(enabled/length/initialTipInsertion/effectiveClearance): {ELASTICROD_ENABLE_INTRODUCER} / {ELASTICROD_INTRODUCER_LENGTH_MM:.3f} mm / {ELASTICROD_INITIAL_TIP_INSERTION_MM:.3f} mm / {ELASTICROD_INTRODUCER_CLEARANCE_MM:.3f} mm',
        f'  strictExternalSupport(length/push/maxPush/maxNodes): {ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM:.3f} mm / {ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_N:.3f} N / {ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_MAX_N:.3f} N / {ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT}',
        f'  strictNativeBoundaryDriverDisabled: {ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER}',
        f'  nativeVirtualSheath(enabled/length/stiffness): {ELASTICROD_ENABLE_VIRTUAL_SHEATH} / {ELASTICROD_SHEATH_LENGTH_MM:.3f} mm / {ELASTICROD_SHEATH_STIFFNESS_N_PER_M:.3f} N/m',
        f'  nativeThrustLimit(enabled/force): {ELASTICROD_ENABLE_THRUST_LIMIT} / {ELASTICROD_THRUST_FORCE_N:.3f} N',
        f'  strictDriverLagLimit(axial soft/hard, reaction soft/hard): '
        f'{ELASTICROD_STRICT_DRIVER_AXIAL_ERROR_SOFT_LIMIT_MM:.3f}/{ELASTICROD_STRICT_DRIVER_AXIAL_ERROR_HARD_LIMIT_MM:.3f} mm ; '
        f'{ELASTICROD_STRICT_DRIVER_REACTION_SOFT_LIMIT_N:.3f}/{ELASTICROD_STRICT_DRIVER_REACTION_HARD_LIMIT_N:.3f} N',
        f'  nativeAxialAssist(enabled/force/deficit): {ELASTICROD_ENABLE_AXIAL_PATH_ASSIST} / {ELASTICROD_AXIAL_PATH_ASSIST_FORCE_N:.3f} N / {ELASTICROD_AXIAL_PATH_ASSIST_DEFICIT_MM:.3f} mm',
        f'  nativeFieldGradient: {ELASTICROD_ENABLE_FIELD_GRADIENT}',
        f'  strictMagneticFlags(recenter/assist/previewScaling/physicalTorqueOnly): {ELASTICROD_STRICT_MAGNETIC_RECENTER} / {ELASTICROD_STRICT_MAGNETIC_ASSIST} / {ELASTICROD_STRICT_MAGNETIC_PREVIEW_SCALING} / {ELASTICROD_STRICT_PHYSICAL_TORQUE_ONLY}',
        f'  strictTipTargetForce: {ELASTICROD_STRICT_TIP_TARGET_FORCE_N:.3f} N',
        f'  strictMagneticPreview(near/far/turns/field/push/recenter): {ELASTICROD_STRICT_BEND_NEAR_WINDOW_MM:.2f} / {ELASTICROD_STRICT_BEND_LOOKAHEAD_MM:.2f} mm ; {ELASTICROD_STRICT_BEND_TURN_MEDIUM_DEG:.1f} / {ELASTICROD_STRICT_BEND_TURN_HIGH_DEG:.1f} deg ; {ELASTICROD_STRICT_FIELD_SCALE_STRAIGHT:.2f} / {ELASTICROD_STRICT_FIELD_SCALE_BEND:.2f} ; {ELASTICROD_STRICT_PUSH_SCALE_STRAIGHT_CONTACT:.2f} / {ELASTICROD_STRICT_PUSH_SCALE_BEND_CONTACT:.2f} ; {ELASTICROD_STRICT_RECENTER_CLEARANCE_MM:.2f} / {ELASTICROD_STRICT_RECENTER_OFFSET_MM:.2f} mm',
        f'  nativeRealtime(dt free/trans/contact): {ELASTICROD_REALTIME_DT_FREE_S:.6f} / {ELASTICROD_REALTIME_DT_TRANSITION_S:.6f} / {ELASTICROD_REALTIME_DT_CONTACT_S:.6f} s',
        f'  nativeGuiWallclock(enabled/maxDt): {ELASTICROD_GUI_WALLCLOCK_CONTROL} / {ELASTICROD_GUI_WALLCLOCK_DT_MAX_S:.3f} s',
        f'  nativeGuiInsertionDtScale(free/trans/contact): {ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_FREE_SCALE:.2f} / {ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_TRANSITION_SCALE:.2f} / {ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_CONTACT_SCALE:.2f}',
        f'  nativeGuiRuntime(dt free/trans/contact): {ELASTICROD_GUI_DT_FREE_S:.6f} / {ELASTICROD_GUI_DT_TRANSITION_S:.6f} / {ELASTICROD_GUI_DT_CONTACT_S:.6f} s',
        f'  nativeRealtime(iter free/trans/contact): {ELASTICROD_REALTIME_SOLVER_MAX_ITER_FREE} / {ELASTICROD_REALTIME_SOLVER_MAX_ITER_TRANSITION} / {ELASTICROD_REALTIME_SOLVER_MAX_ITER_CONTACT}',
        f'  nativeRealtime(tol free/trans/contact): {ELASTICROD_REALTIME_SOLVER_TOL_FREE:.3e} / {ELASTICROD_REALTIME_SOLVER_TOL_TRANSITION:.3e} / {ELASTICROD_REALTIME_SOLVER_TOL_CONTACT:.3e}',
        f'  nativeRealtime(speedScale free/trans/contact): {ELASTICROD_REALTIME_SPEED_SCALE_FREE:.2f} / {ELASTICROD_REALTIME_SPEED_SCALE_TRANSITION:.2f} / {ELASTICROD_REALTIME_SPEED_SCALE_CONTACT:.2f}',
        f'  nativeRealtime(steering enter/contact, clearance transition/contact): {ELASTICROD_REALTIME_STEERING_ENTER_DEG:.1f} / {ELASTICROD_REALTIME_STEERING_CONTACT_DEG:.1f} deg ; {ELASTICROD_REALTIME_CLEARANCE_TRANSITION_MM:.2f} / {ELASTICROD_REALTIME_CLEARANCE_CONTACT_MM:.2f} mm',
        f'  magnetic look-ahead / smoothing / maxTurn / ramp / lateralScale(beam/shared): {MAGNETIC_LOOKAHEAD_DISTANCE_MM:.2f} mm / {MAGNETIC_FIELD_SMOOTHING_ALPHA:.2f} / {MAGNETIC_MAX_TURN_ANGLE_DEG:.1f} deg / {MAGNETIC_FIELD_RAMP_TIME_S:.2f} s / {MAGNETIC_LATERAL_FORCE_SCALE:.2f}',
        f'  native magnetic steering(lookAhead/smoothing/maxTurn/ramp): {ELASTICROD_MAGNETIC_LOOKAHEAD_DISTANCE_MM:.2f} mm / {ELASTICROD_MAGNETIC_FIELD_SMOOTHING_ALPHA:.2f} / {ELASTICROD_MAGNETIC_MAX_TURN_ANGLE_DEG:.1f} deg / {ELASTICROD_MAGNETIC_FIELD_RAMP_TIME_S:.2f} s',
        f'  native magnetic assist(strength/minTorqueSin/lateralScale): {ELASTICROD_MAGNETIC_STRENGTH_SCALE:.2f} / {ELASTICROD_NATIVE_MAGNETIC_MIN_TORQUE_SIN:.2f} / {ELASTICROD_MAGNETIC_LATERAL_FORCE_SCALE:.2f}',
        f'  strict gui light-contact(pushScale/dt/clearance/gap/barrierNodes): {ELASTICROD_STRICT_LIGHT_CONTACT_PUSH_SCALE:.2f} / {ELASTICROD_STRICT_GUI_LIGHT_CONTACT_INSERTION_DT_S:.4f} s / {ELASTICROD_STRICT_GUI_LIGHT_CONTACT_CLEARANCE_MM:.2f} / {ELASTICROD_STRICT_GUI_LIGHT_CONTACT_WALL_GAP_MM:.2f} mm / {ELASTICROD_STRICT_GUI_LIGHT_CONTACT_MAX_BARRIER_NODES}',
        f'  strict gui guided-contact(pushScale/dt/clearance/gap/barrierNodes): {ELASTICROD_STRICT_GUIDED_CONTACT_PUSH_SCALE:.2f} / {ELASTICROD_STRICT_GUI_GUIDED_CONTACT_INSERTION_DT_S:.4f} s / {ELASTICROD_STRICT_GUI_GUIDED_CONTACT_CLEARANCE_MM:.2f} / {ELASTICROD_STRICT_GUI_GUIDED_CONTACT_WALL_GAP_MM:.2f} mm / {ELASTICROD_STRICT_GUI_GUIDED_CONTACT_MAX_BARRIER_NODES}',
        f'  strict gui guided-precontact(clearance/gap/barrierNodes): {ELASTICROD_STRICT_GUI_GUIDED_PRECONTACT_CLEARANCE_MM:.2f} / {ELASTICROD_STRICT_GUI_GUIDED_PRECONTACT_WALL_GAP_MM:.2f} mm / {ELASTICROD_STRICT_GUI_GUIDED_PRECONTACT_MAX_BARRIER_NODES}',
        f'  elasticrod strict guard(enabled/lumenClamp/speedCap): {ELASTICROD_ENABLE_STRICT_POSTSOLVE_GUARD} / {ELASTICROD_ENABLE_STRICT_LUMEN_CLAMP} / {ELASTICROD_STRICT_MAX_LINEAR_SPEED_MM_S:.1f} mm/s',
        f'  elasticrod strict barrier(enabled/activation/safety): {ELASTICROD_STRICT_NATIVE_LUMEN_BARRIER} / {ELASTICROD_STRICT_BARRIER_ACTIVATION_MARGIN_MM:.2f} mm / {ELASTICROD_STRICT_BARRIER_SAFETY_MARGIN_MM:.2f} mm',
        f'  elasticrod strict barrier(stiffness/damping/maxForce): {ELASTICROD_STRICT_BARRIER_STIFFNESS_N_PER_M:.1f} N/m / {ELASTICROD_STRICT_BARRIER_DAMPING_N_S_PER_M:.2f} N.s/m / {ELASTICROD_STRICT_BARRIER_MAX_FORCE_PER_NODE_N:.2f} N',
        f'  elasticrod strict steering(pushScale/axialAssist/headStretch): {ELASTICROD_STRICT_PUSH_SCALE_WHEN_BARRIER_ACTIVE:.2f} / {ELASTICROD_STRICT_AXIAL_ASSIST_SCALE_WHEN_STEERING:.2f} / {ELASTICROD_STRICT_HEAD_STRETCH_LIMIT:.3f}',
    ]
