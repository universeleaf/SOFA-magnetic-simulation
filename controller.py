# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import time
import os
from pathlib import Path
from typing import Tuple

import numpy as np

from .config import (
    BEAM_HEAD_CLEARANCE_REFRESH_FAR_STEPS,
    BEAM_HEAD_CLEARANCE_REFRESH_NEAR_STEPS,
    BEAM_PRE_ENTRY_ACCESS_GUIDE_MM,
    BEAM_RUNTIME_IS_RECORDING,
    BEAM_SURFACE_REFRESH_FAR_STEPS,
    BEAM_SURFACE_REFRESH_NEAR_STEPS,
    BEAM_USE_KINEMATIC_INSERTION,
    BEAM_PYTHON_MAGNETIC_FIELD_STRENGTH,
    BEAM_PYTHON_MAGNETIC_FORCE_GAIN,
    BEAM_PYTHON_MAGNETIC_MOMENT,
    BEAM_COMPRESSION_ENTER_MM,
    BEAM_COMPRESSION_EXIT_MM,
    BEAM_STALL_COMPRESSION_ENTER_MM,
    BEAM_STALL_COMPRESSION_EXIT_MM,
    BEAM_STALL_SPEED_ENTER_MM_S,
    BEAM_STALL_SPEED_EXIT_MM_S,
    CONTACT_DISTANCE_MM,
    CONTACT_ALARM_DISTANCE_MM,
    DEBUG_PRINT_EVERY,
    DEFAULT_INSERTION_DIR,
    DISTAL_VISUAL_NODE_COUNT,
    ELASTICROD_DISTAL_VISUAL_NODE_COUNT,
    ELASTICROD_DIAGNOSTIC_ANGULAR_SPEED_WARN_RAD_S,
    ELASTICROD_DIAGNOSTIC_DISPLACEMENT_WARN_MM,
    ELASTICROD_DIAGNOSTIC_PROFILE,
    ELASTICROD_DIAGNOSTIC_LINEAR_SPEED_WARN_MM_S,
    ELASTICROD_DIAGNOSTIC_PRINT_EVERY,
    ELASTICROD_DIAGNOSTIC_STEP_WINDOW,
    ELASTICROD_DT_S,
    ELASTICROD_FAILFAST_EDGE_STRETCH_RATIO,
    ELASTICROD_FAILFAST_MAX_STRETCH,
    ELASTICROD_ACTIVE_STARTUP_RAMP_TIME_S,
    ELASTICROD_ENABLE_INTRODUCER,
    ELASTICROD_ENABLE_SAFE_RECOVERY,
    ELASTICROD_ENABLE_AXIAL_PATH_ASSIST,
    ELASTICROD_AXIAL_DRIVE_NODE_COUNT,
    ELASTICROD_ENABLE_THRUST_LIMIT,
    ELASTICROD_ENABLE_VIRTUAL_SHEATH,
    ELASTICROD_AXIAL_PATH_ASSIST_DEFICIT_MM,
    ELASTICROD_AXIAL_PATH_ASSIST_FORCE_N,
    ELASTICROD_AXIAL_PATH_ASSIST_CONTACT_FORCE_SCALE,
    ELASTICROD_AXIAL_PATH_ASSIST_CONTACT_MIN_SCALE,
    ELASTICROD_AXIAL_PATH_ASSIST_WINDOW_BACK_MM,
    ELASTICROD_AXIAL_PATH_ASSIST_WINDOW_FRONT_MM,
    ELASTICROD_AXIAL_PATH_ASSIST_WINDOW_MIN_NODE_COUNT,
    ELASTICROD_AXIAL_PATH_ASSIST_MAX_PUSH_SCALE,
    ELASTICROD_ENTRY_PUSH_BAND_LENGTH_MM,
    ELASTICROD_ENTRY_PUSH_BAND_MIN_NODE_COUNT,
    ELASTICROD_ENTRY_PUSH_BAND_OUTSIDE_OFFSET_MM,
    ELASTICROD_GUI_WALLCLOCK_CONTROL,
    ELASTICROD_GUI_DT_CONTACT_S,
    ELASTICROD_GUI_DT_FREE_S,
    ELASTICROD_GUI_DT_TRANSITION_S,
    ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_CONTACT_SCALE,
    ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_FREE_SCALE,
    ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_TRANSITION_SCALE,
    ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_MAX_S,
    ELASTICROD_GUI_WALLCLOCK_DT_MAX_S,
    ELASTICROD_GUI_DIAGNOSTIC_LINEAR_SPEED_WARN_MM_S,
    ELASTICROD_GUI_DIAGNOSTIC_MIN_LOG_INTERVAL_STEPS,
    ELASTICROD_GUI_WALLCLOCK_PUSH_SPEED_SCALE,
    ELASTICROD_GUI_WALLCLOCK_STARTUP_RAMP_S,
    ELASTICROD_GUI_SOLVER_MAX_ITER_CONTACT,
    ELASTICROD_GUI_SOLVER_MAX_ITER_FREE,
    ELASTICROD_GUI_SOLVER_MAX_ITER_TRANSITION,
    ELASTICROD_GUI_SOLVER_TOL_CONTACT,
    ELASTICROD_GUI_SOLVER_TOL_FREE,
    ELASTICROD_GUI_SOLVER_TOL_TRANSITION,
    ELASTICROD_INTRODUCER_LENGTH_MM,
    ELASTICROD_REALTIME_CLEARANCE_CONTACT_MM,
    ELASTICROD_REALTIME_CLEARANCE_TRANSITION_MM,
    ELASTICROD_REALTIME_DT_CONTACT_S,
    ELASTICROD_REALTIME_DT_FREE_S,
    ELASTICROD_REALTIME_DT_TRANSITION_S,
    ELASTICROD_REALTIME_SOLVER_MAX_ITER_CONTACT,
    ELASTICROD_REALTIME_SOLVER_MAX_ITER_FREE,
    ELASTICROD_REALTIME_SOLVER_MAX_ITER_TRANSITION,
    ELASTICROD_REALTIME_SOLVER_TOL_CONTACT,
    ELASTICROD_REALTIME_SOLVER_TOL_FREE,
    ELASTICROD_REALTIME_SOLVER_TOL_TRANSITION,
    ELASTICROD_REALTIME_SPEED_SCALE_CONTACT,
    ELASTICROD_REALTIME_SPEED_SCALE_FREE,
    ELASTICROD_REALTIME_SPEED_SCALE_TRANSITION,
    ELASTICROD_REALTIME_STEERING_CONTACT_DEG,
    ELASTICROD_REALTIME_STEERING_ENTER_DEG,
    ELASTICROD_RUNTIME_PROFILE,
    ELASTICROD_CONSTRAINT_SOLVER_MAX_ITER,
    ELASTICROD_CONSTRAINT_SOLVER_TOLERANCE,
    ELASTICROD_CONTACT_DISTANCE_MM,
    ELASTICROD_SAFE_RECOVERY_ANGULAR_SPEED_RAD_S,
    ELASTICROD_SAFE_RECOVERY_COOLDOWN_STEPS,
    ELASTICROD_SAFE_RECOVERY_DISPLACEMENT_MM,
    ELASTICROD_SAFE_RECOVERY_LINEAR_SPEED_MM_S,
    ELASTICROD_SAFE_RECOVERY_MAX_STRETCH,
    ELASTICROD_SAFE_RECOVERY_RETRACT_MM,
    ELASTICROD_SAFE_TIP_WALL_CONTACT_RELEASE_HOLD_STEPS,
    ELASTICROD_ENABLE_DISPLACEMENT_PUSH,
    ELASTICROD_DISPLACEMENT_PUSH_VELOCITY_MM_PER_S,
    ELASTICROD_DISPLACEMENT_PUSH_RELEASE_MM,
    ELASTICROD_DISPLACEMENT_PUSH_CENTERING_GAIN,
    ELASTICROD_RECOVERY_TRIGGER_DISPLACEMENT_MM,
    ELASTICROD_RECOVERY_TRIGGER_LINEAR_SPEED_MM_S,
    ELASTICROD_RECOVERY_TRIGGER_MAX_STRETCH,
    ELASTICROD_SHEATH_LENGTH_MM,
    ELASTICROD_STABILIZATION_MODE,
    ELASTICROD_STARTUP_RAMP_TIME_S,
    ELASTICROD_STRICT_BARRIER_ACTIVATION_MARGIN_MM,
    ELASTICROD_STRICT_BARRIER_SAFETY_MARGIN_MM,
    ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER,
    ELASTICROD_STRICT_DRIVER_AXIAL_ERROR_HARD_LIMIT_MM,
    ELASTICROD_STRICT_DRIVER_AXIAL_ERROR_SOFT_LIMIT_MM,
    ELASTICROD_STRICT_DRIVER_REACTION_HARD_LIMIT_N,
    ELASTICROD_STRICT_DRIVER_REACTION_SOFT_LIMIT_N,
    ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_MAX_N,
    ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_N,
    ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT,
    ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM,
    ELASTICROD_STRICT_EXTERNAL_SUPPORT_LENGTH_MM,
    ELASTICROD_ENABLE_STRICT_LUMEN_CLAMP,
    ELASTICROD_ENABLE_STRICT_POSTSOLVE_GUARD,
    ELASTICROD_STRICT_PUSH_SCALE_STRAIGHT_CONTACT,
    ELASTICROD_STRICT_PUSH_SCALE_BEND_CONTACT,
    ELASTICROD_STRICT_GUIDED_CONTACT_PUSH_SCALE,
    ELASTICROD_STRICT_LIGHT_CONTACT_PUSH_SCALE,
    ELASTICROD_STRICT_RECENTER_CLEARANCE_MM,
    ELASTICROD_STRICT_LUMEN_CLAMP_TOLERANCE_MM,
    ELASTICROD_STRICT_MAX_LINEAR_SPEED_MM_S,
    ELASTICROD_STRICT_HEAD_STRETCH_LIMIT,
    ELASTICROD_STRICT_HEAD_STRETCH_SOFT_LIMIT,
    ELASTICROD_STRICT_GLOBAL_STRETCH_HARD_LIMIT,
    ELASTICROD_STRICT_GLOBAL_STRETCH_SOFT_LIMIT,
    ELASTICROD_STRICT_INITIAL_STRAIGHT_PUSH_MM,
    ELASTICROD_STRICT_GUI_EXACT_SURFACE_RECHECK_STEPS,
    ELASTICROD_STRICT_GUI_FAR_CLEARANCE_MM,
    ELASTICROD_STRICT_GUI_FAR_OFFSET_MM,
    ELASTICROD_STRICT_GUI_GUIDED_CONTACT_CLEARANCE_MM,
    ELASTICROD_STRICT_GUI_GUIDED_CONTACT_INSERTION_DT_S,
    ELASTICROD_STRICT_GUI_GUIDED_CONTACT_MAX_BARRIER_NODES,
    ELASTICROD_STRICT_GUI_GUIDED_CONTACT_WALL_GAP_MM,
    ELASTICROD_STRICT_GUI_GUIDED_PRECONTACT_CLEARANCE_MM,
    ELASTICROD_STRICT_GUI_GUIDED_PRECONTACT_MAX_BARRIER_NODES,
    ELASTICROD_STRICT_GUI_GUIDED_PRECONTACT_WALL_GAP_MM,
    ELASTICROD_STRICT_GUI_LIGHT_CONTACT_CLEARANCE_MM,
    ELASTICROD_STRICT_GUI_LIGHT_CONTACT_INSERTION_DT_S,
    ELASTICROD_STRICT_GUI_LIGHT_CONTACT_MAX_BARRIER_NODES,
    ELASTICROD_STRICT_GUI_LIGHT_CONTACT_WALL_GAP_MM,
    ELASTICROD_STRICT_GUI_MAX_INSERTION_STEP_MM,
    ELASTICROD_STRICT_FEED_BOOST_BEND,
    ELASTICROD_STRICT_FEED_BOOST_CONTACT,
    ELASTICROD_STRICT_FEED_BOOST_HEAD_STRETCH_LIMIT,
    ELASTICROD_STRICT_FEED_BOOST_START_MM,
    ELASTICROD_STRICT_GUI_SURFACE_REFRESH_FAR_STEPS,
    ELASTICROD_STRICT_GUI_SURFACE_REFRESH_NEAR_STEPS,
    ELASTICROD_STRICT_ENTRY_PUSH_BAND_ENABLED,
    ELASTICROD_STRICT_DRIVE_WINDOW_LENGTH_MM,
    ELASTICROD_STRICT_DRIVE_WINDOW_MIN_NODE_COUNT,
    ELASTICROD_STRICT_DRIVE_WINDOW_OUTSIDE_OFFSET_MM,
    ELASTICROD_STRICT_MAGNETIC_RELEASE_SPAN_MM,
    ELASTICROD_STRICT_RUNTIME_BEND_SEVERITY_CONTACT,
    ELASTICROD_STRICT_RUNTIME_BEND_SEVERITY_TRANSITION,
    ELASTICROD_STRICT_RUNTIME_RELEASE_CONTACT_HOLD_STEPS,
    ELASTICROD_STRICT_RUNTIME_PROGRESS_GATE_MM,
    ELASTICROD_STRICT_RUNTIME_RELEASE_TRANSITION_HOLD_STEPS,
    ELASTICROD_STRICT_SUPPORT_RELEASE_MM,
    ELASTICROD_STRICT_SUPPORT_WINDOW_LENGTH_MM,
    ELASTICROD_USE_KINEMATIC_SHEATH_DRIVER,
    ELASTICROD_STRICT_TIP_WALL_CONTACT_ENTER_MM,
    ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM,
    ELASTICROD_STRICT_TIP_WALL_CONTACT_RELEASE_HOLD_STEPS,
    ELASTICROD_STRICT_SIMPLE_TAIL_DRIVE,
    ELASTICROD_STRICT_ALWAYS_PUSH_FORWARD,
    ELASTICROD_STRICT_HAND_PUSH_NODE_COUNT,
    ELASTICROD_STRICT_NATIVE_LUMEN_BARRIER,
    ELASTICROD_THRUST_FORCE_N,
    ELASTICROD_MAGNETIC_HEAD_EDGES,
    ENABLE_LUMEN_SAFETY_PROJECTION,
    ENABLE_VIRTUAL_SHEATH,
    LUMEN_CLEARANCE_MM,
    LUMEN_CONSTRAINT_TOLERANCE_MM,
    MAGNETIC_FORCE_ARROW_HEAD_LENGTH_MM,
    MAGNETIC_FORCE_ARROW_HEAD_WIDTH_MM,
    MAGNETIC_FORCE_ARROW_LENGTH_MM,
    MAX_SEGMENT_LENGTH_RATIO,
    MIN_SEGMENT_LENGTH_RATIO,
    NAVIGATION_MODE,
    MAGNETIC_LOOKAHEAD_DISTANCE_MM,
    PUSH_FORCE_CALIBRATION_ALPHA,
    PUSH_FORCE_CALIBRATION_LOCK_NEG_SPEED_MM_S,
    PUSH_FORCE_CALIBRATION_RAMP_PER_S,
    PUSH_FORCE_CALIBRATION_REGION_MM,
    PUSH_FORCE_CALIBRATION_TIME_S,
    PUSH_FORCE_INITIAL_TOTAL,
    PUSH_FORCE_MAX_TOTAL,
    PUSH_FORCE_MIN_TOTAL,
    ELASTICROD_PUSH_NODE_COUNT,
    PUSH_FORCE_NODE_COUNT,
    PUSH_FORCE_REDUCED_SCALE_ON_COMPRESSION,
    PUSH_FORCE_REDUCED_SCALE_ON_STALL,
    PUSH_FORCE_REDUCED_SCALE_ON_STEERING,
    PUSH_FORCE_REDUCED_SCALE_ON_WALL,
    PUSH_FORCE_SCALE_DROP_TIME_S,
    PUSH_FORCE_SCALE_RISE_TIME_S,
    PUSH_FORCE_TARGET_SPEED_MM_S,
    STEERING_MISALIGN_ENTER_DEG,
    STEERING_MISALIGN_EXIT_DEG,
    TARGET_MARKER_SIZE_MM,
    TIP_SPEED_FILTER_ALPHA,
    TIP_WALL_CONTACT_ENTER_MM,
    TIP_WALL_CONTACT_EXIT_MM,
    VIRTUAL_SHEATH_BLEND_OUT_MM,
    VIRTUAL_SHEATH_RELEASE_S_MM,
    WALL_CONTACT_TIP_PROBE_NODES,
    WIRE_RADIUS_MM,
    WIRE_TOTAL_LENGTH_MM,
    NATIVE_WIRE_RADIUS_MM,
    NATIVE_WIRE_TOTAL_LENGTH_MM,
    MAGNETIC_HEAD_EDGES,
    MAGNETIC_FIELD_SMOOTHING_ALPHA,
    GUIDEWIRE_BACKEND,
)
from .geometry import _NearestSurface, _lumen_profile, _opening_radius
from .math_utils import _cumlen, _interp, _marker_points, _normalize, _parallel_transport, _quat_from_basis, _quat_from_z_to, _quat_rotate, _read, _writeable
from .references.guidewire_magnetic_control import DistalMagneticHeadModel, UniformMagneticFieldController
from .runtime import ensure_sofa

Sofa = ensure_sofa()

print('[controller] elasticrod sync v2 loaded 闁?direct visual node update')


class GuidewireControllerBase(Sofa.Core.Controller):
    """
    Python 闁硅矇鍐ㄧ厬闁革絻鍔嶇€垫粓宕ユ惔锝庝紓闁告帒妫旂悮閬嶅级闄囨禍瀵告嫻閿濆鎳犻柨?    1. `beam` fallback闁挎稒姘ㄩ幋椋庣磼椤撴繂鈻忛柣顫妽濡偊鎯冮崟顔剧闁告柣鍔岄鐔兼焻娴ｉ顐?+ 闁惧繑纰嶇€氭瑩妫勫Ο纰卞悁 + 闁煎灚鏌ㄩ崬鎾箮閺囩偛顨涚紒瀣暱閻ｉ箖宕犻弽鐢靛耿
    2. `elasticrod`闁挎稒鑹捐ぐ褔寮寸€涙ɑ鐓€闁告鍠撻弫鎾澄熼垾宕団偓鐑芥儍閸曨喚鐝堕柣锝呰嫰閹斥剝绂掗妶蹇曠濞戞挸绉撮崯鈧柟闈涚秺閸ｈ櫣鎲伴崱妤€鏅搁柡浣虹節缁楋絾鎷呭鍥╂瀭闁?
    濞戞挶鍊撻柌婊堝触鎼达綆浼傞柛蹇氫含閺併倝鎯冮崟顖氬姤闁告帒妫楄ぐ褔寮垫径搴ｆ閻犲洦娲栬ぐ鑼喆閸℃顕ч柕鍡曟祰閸掓稒绔熸担绋挎闁规亽鍔岄崹浠嬪箲椤旇　鍋撴担瑙勶級闊洦顨呴幏浼存儎閸涘﹥绨氶柛姘湰椤掔偤濡?    缁惧彞绀佸┃鈧柡鍌滄嚀閹粓濡存担瑙勪粯閺夆晜鍨抽懙鎴ｇ疀閸愵亜娈犳繛鍫濈仛閻擄繝骞嶆穱鎵佸亾娴ｄ警姊块柛?闁告梹绋撻悡鈺呭礆閸℃稑甯崇€规瓕灏欑划锟犲礂閵娾晛鍔ョ紒澶庮唺濮橈妇绱掑▎蹇撴枾闁?C++ 缂備礁瀚▎銏ゅΥ?    """

    def __init__(
        self,
        *args,
        root_node=None,
        constraint_solver=None,
        wire_mech=None,
        proximal_push_ff=None,
        tip_torque_ff=None,
        rod_model=None,
        native_mass=None,
        native_axial_assist_ff=None,
        native_axial_assist_indices=None,
        entry_push_indices=None,
        backend_name: str = 'beam',
        magnetic_force_field=None,
        centerline_points=None,
        insertion_direction=(0.0, 0.0, 1.0),
        push_force_target_speed_mm_s: float = 1.0,
        max_push_mm: float = 1e9,
        node_initial_path_s_mm=None,
        vessel_vertices=None,
        vessel_faces=None,
        vessel_surface_query_face_candidate_count: int = 384,
        enable_vessel_lumen_constraint: bool = ENABLE_LUMEN_SAFETY_PROJECTION,
        enable_virtual_sheath: bool = ENABLE_VIRTUAL_SHEATH,
        target_marker_mech=None,
        force_arrow_mech=None,
        force_arrow_anchor=None,
        navigation_mode: int = NAVIGATION_MODE,
        drive_node_count: int = PUSH_FORCE_NODE_COUNT,
        native_support_indices=None,
        native_drive_window_indices=None,
        tail_push_indices=None,
        external_support_length_mm: float = 0.0,
        external_support_radius_mm: float = 0.0,
        camera_object=None,
        camera_follow_offset=None,
        use_python_magnetic_fallback: bool = False,
        python_magnetic_field_strength: float = BEAM_PYTHON_MAGNETIC_FIELD_STRENGTH,
        python_magnetic_moment: float = BEAM_PYTHON_MAGNETIC_MOMENT,
        python_magnetic_force_gain: float = BEAM_PYTHON_MAGNETIC_FORCE_GAIN,
        native_virtual_sheath_target_mech=None,
        native_virtual_sheath_indices=None,
        native_virtual_sheath_offsets_mm=None,
        native_virtual_sheath_stiffnesses=None,
        enable_native_virtual_sheath: bool = ELASTICROD_ENABLE_VIRTUAL_SHEATH,
        enable_native_thrust_limit: bool = ELASTICROD_ENABLE_THRUST_LIMIT,
        native_thrust_force_n: float = ELASTICROD_THRUST_FORCE_N,
        physics_rod_state_mech=None,
        **kwargs,
    ):
        kwargs.setdefault('listening', True)
        super().__init__(*args, **kwargs)
        self.root_node = root_node
        self.constraint_solver = constraint_solver
        self.wire_mech = wire_mech
        self.proximal_push_ff = proximal_push_ff
        self._proximal_push_indices_data = proximal_push_ff.findData('indices') if proximal_push_ff is not None else None
        self.tip_torque_ff = tip_torque_ff
        self.rod_model = rod_model
        self.native_mass = native_mass
        self.native_axial_assist_ff = native_axial_assist_ff
        self._native_axial_assist_indices_data = (
            native_axial_assist_ff.findData('indices') if native_axial_assist_ff is not None else None
        )
        self.native_axial_assist_indices = [int(i) for i in (native_axial_assist_indices or [])]
        self.native_drive_window_indices = sorted({
            int(i) for i in (native_drive_window_indices or []) if int(i) >= 0
        })
        self.tail_push_indices = [int(i) for i in (tail_push_indices or entry_push_indices or [])]
        self.entry_push_indices = self.tail_push_indices.copy()
        self.external_support_length_mm = float(max(external_support_length_mm, 0.0))
        self.external_support_radius_mm = float(max(external_support_radius_mm, 0.0))
        self._requested_native_support_indices = [int(i) for i in (native_support_indices or [])]
        self.backend_name = str(backend_name)
        self.is_native_backend = self.backend_name == 'elasticrod' and self.rod_model is not None
        self.is_native_strict = self.is_native_backend and ELASTICROD_STABILIZATION_MODE == 'strict'
        self.is_native_safe = self.is_native_backend and ELASTICROD_STABILIZATION_MODE == 'safe'
        self.use_native_displacement_feed = (
            self.is_native_backend
            and self.is_native_safe
            and bool(ELASTICROD_ENABLE_DISPLACEMENT_PUSH)
        )
        self.use_native_entry_push_band = (
            self.is_native_safe
            or (
                (not self.is_native_strict)
                and bool(ELASTICROD_STRICT_ENTRY_PUSH_BAND_ENABLED)
                and not bool(ELASTICROD_STRICT_SIMPLE_TAIL_DRIVE)
            )
        )
        self.native_entry_push_band_length_mm = float(max(ELASTICROD_ENTRY_PUSH_BAND_LENGTH_MM, 0.5))
        self.native_entry_push_band_outside_offset_mm = float(max(ELASTICROD_ENTRY_PUSH_BAND_OUTSIDE_OFFSET_MM, 0.0))
        self.native_entry_push_band_min_node_count = int(max(ELASTICROD_ENTRY_PUSH_BAND_MIN_NODE_COUNT, 1))
        self._active_native_push_indices = [int(i) for i in (self.tail_push_indices or self.native_axial_assist_indices)]
        self._active_external_push_indices: list[int] = []
        self.enable_native_strict_postsolve_guard = bool(
            self.is_native_strict and ELASTICROD_ENABLE_STRICT_POSTSOLVE_GUARD
        )
        self.enable_native_strict_lumen_clamp = bool(
            self.is_native_strict and ELASTICROD_ENABLE_STRICT_LUMEN_CLAMP
        )
        self.native_strict_lumen_clamp_tolerance_mm = float(max(ELASTICROD_STRICT_LUMEN_CLAMP_TOLERANCE_MM, 0.0))
        self.native_strict_max_linear_speed_mm_s = float(max(ELASTICROD_STRICT_MAX_LINEAR_SPEED_MM_S, 0.0))
        self.is_beam_realtime = (not self.is_native_backend) and bool(BEAM_RUNTIME_IS_RECORDING)
        self.is_native_realtime = self.is_native_backend and ELASTICROD_RUNTIME_PROFILE == 'realtime_gui_10min'
        self._is_runsofa_process = 'runsofa' in Path(sys.executable).name.lower()
        self._gui_wallclock_launch = os.environ.get('GUIDEWIRE_ELASTICROD_GUI_WALLCLOCK', '').strip() == '1'
        self.use_native_gui_wallclock_control = (
            self.is_native_backend
            and self.is_native_realtime
            and bool(ELASTICROD_GUI_WALLCLOCK_CONTROL)
            and (self._is_runsofa_process or self._gui_wallclock_launch)
        )
        self.native_diagnostic_realtime = self.is_native_backend and ELASTICROD_DIAGNOSTIC_PROFILE == 'realtime'
        self.magnetic_force_field = magnetic_force_field
        self.centerline = np.asarray(centerline_points or [], dtype=float)
        self.centerline_cum = _cumlen(self.centerline[:, :3])
        self.path_len = float(self.centerline_cum[-1]) if self.centerline_cum.size else 0.0
        self.insertion_direction = _normalize(np.asarray(insertion_direction, dtype=float))
        if np.linalg.norm(self.insertion_direction) < 1e-12:
            self.insertion_direction = DEFAULT_INSERTION_DIR.copy()
        self.push_force_target_speed_mm_s = float(push_force_target_speed_mm_s)
        self.max_push_mm = float(max_push_mm)
        self.navigation_mode = 2 if int(navigation_mode) == 2 else 1
        self.enable_vessel_lumen_constraint = bool(enable_vessel_lumen_constraint)
        self.enable_virtual_sheath = bool(enable_virtual_sheath)
        self.enable_native_virtual_sheath = self.is_native_backend and bool(enable_native_virtual_sheath)
        self.enable_native_thrust_limit = self.is_native_backend and bool(enable_native_thrust_limit)
        if self.is_native_strict:
            self.enable_native_virtual_sheath = False
            self.enable_native_thrust_limit = False
        self.enable_native_axial_path_assist = (
            self.is_native_backend
            and (not self.is_native_strict)
            and bool(ELASTICROD_ENABLE_AXIAL_PATH_ASSIST)
            and self.native_axial_assist_ff is not None
            and len(self.native_axial_assist_indices) > 0
        )
        self.native_thrust_force_n = float(native_thrust_force_n)
        self.use_beam_safety_projection = (
            (not self.is_native_backend)
            and (self.enable_vessel_lumen_constraint or self.enable_virtual_sheath)
        )
        self.use_kinematic_beam_insertion = (not self.is_native_backend) and bool(BEAM_USE_KINEMATIC_INSERTION)
        self.vessel_vertices = np.asarray(vessel_vertices or [], dtype=float)
        self.vessel_faces = np.asarray(vessel_faces or [], dtype=int).reshape(-1, 3)
        self.vessel_surface_query_face_candidate_count = max(int(vessel_surface_query_face_candidate_count), 32)
        self.vessel_surface_query = (
            _NearestSurface(
                self.vessel_vertices[:, :3],
                self.vessel_faces[:, :3],
                face_candidate_count=self.vessel_surface_query_face_candidate_count,
            )
            if self.vessel_vertices.ndim == 2 and self.vessel_vertices.shape[0] > 0
            else None
        )

        self._pos = self.wire_mech.findData('position')
        self._rest = self.wire_mech.findData('rest_position')
        self._vel = self.wire_mech.findData('velocity')
        self._rod_state_pos = (
            physics_rod_state_mech.findData('position')
            if physics_rod_state_mech is not None else None
        )
        self._rod_state_vel = (
            physics_rod_state_mech.findData('velocity')
            if physics_rod_state_mech is not None else None
        )
        self._rod_state_free_pos = (
            physics_rod_state_mech.findData('free_position')
            if physics_rod_state_mech is not None else None
        )
        self._rod_state_free_vel = (
            physics_rod_state_mech.findData('free_velocity')
            if physics_rod_state_mech is not None else None
        )
        self._target_marker_pos = target_marker_mech.findData('position') if target_marker_mech is not None else None
        self._force_arrow_pos = force_arrow_mech.findData('position') if force_arrow_mech is not None else None
        self._camera_position = camera_object.findData('position') if camera_object is not None else None
        self._camera_lookat = camera_object.findData('lookAt') if camera_object is not None else None

        self._debug_target_point = magnetic_force_field.findData('debugTargetPoint') if magnetic_force_field is not None else None
        self._debug_lookahead_point = magnetic_force_field.findData('debugLookAheadPoint') if magnetic_force_field is not None else None
        self._debug_ba_vector = magnetic_force_field.findData('debugBaVector') if magnetic_force_field is not None else None
        self._debug_force_vector = magnetic_force_field.findData('debugForceVector') if magnetic_force_field is not None else None
        self._debug_torque_vector = magnetic_force_field.findData('debugTorqueVector') if magnetic_force_field is not None else None
        self._debug_magnetic_moment_vector = magnetic_force_field.findData('debugMagneticMomentVector') if magnetic_force_field is not None else None
        self._debug_torque_sin = magnetic_force_field.findData('debugTorqueSin') if magnetic_force_field is not None else None
        self._debug_assist_force_vector = magnetic_force_field.findData('debugAssistForceVector') if magnetic_force_field is not None else None
        self._debug_outward_assist_component = magnetic_force_field.findData('debugOutwardAssistComponentN') if magnetic_force_field is not None else None
        self._debug_distal_tangent_field_angle_deg = magnetic_force_field.findData('debugDistalTangentFieldAngleDeg') if magnetic_force_field is not None else None
        self._debug_upcoming_turn_deg = magnetic_force_field.findData('debugUpcomingTurnDeg') if magnetic_force_field is not None else None
        self._debug_bend_severity = magnetic_force_field.findData('debugBendSeverity') if magnetic_force_field is not None else None
        self._debug_scheduled_field_scale = magnetic_force_field.findData('debugScheduledFieldScale') if magnetic_force_field is not None else None
        self._debug_scheduled_field_scale_base = magnetic_force_field.findData('debugScheduledFieldScaleBase') if magnetic_force_field is not None else None
        self._debug_strict_steering_need_alpha = magnetic_force_field.findData('debugStrictSteeringNeedAlpha') if magnetic_force_field is not None else None
        self._debug_entry_release_alpha = magnetic_force_field.findData('debugEntryReleaseAlpha') if magnetic_force_field is not None else None
        self._debug_recentering_alpha = magnetic_force_field.findData('debugRecenteringAlpha') if magnetic_force_field is not None else None
        self._native_br_vector = magnetic_force_field.findData('brVector') if magnetic_force_field is not None else None
        self._native_external_field_scale = magnetic_force_field.findData('externalFieldScale') if magnetic_force_field is not None else None
        self._native_external_control_dt = magnetic_force_field.findData('externalControlDt') if magnetic_force_field is not None else None
        self._native_external_surface_clearance = magnetic_force_field.findData('externalSurfaceClearanceMm') if magnetic_force_field is not None else None
        self._native_external_surface_contact_active = magnetic_force_field.findData('externalSurfaceContactActive') if magnetic_force_field is not None else None
        self._native_nominal_br_vector = (
            np.asarray(self._native_br_vector.value, dtype=float).reshape(3).copy()
            if self._native_br_vector is not None else np.zeros(3, dtype=float)
        )
        self._native_commanded_insertion = rod_model.findData('commandedInsertion') if rod_model is not None else None
        self._native_commanded_twist = rod_model.findData('commandedTwist') if rod_model is not None else None
        self._native_insertion_direction = rod_model.findData('insertionDirection') if rod_model is not None else None
        self.native_strict_boundary_driver_enabled = (
            self.is_native_strict
            and (self._native_commanded_insertion is not None)
            and (not bool(ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER))
        )
        self.use_native_displacement_feed = bool(
            self.use_native_displacement_feed
            or (
                self.is_native_strict
                and bool(ELASTICROD_STRICT_SIMPLE_TAIL_DRIVE)
                and (not self.native_strict_boundary_driver_enabled)
            )
        )
        self._native_use_kinematic_support = rod_model.findData('useKinematicSupportBlock') if rod_model is not None else None
        self._native_push_node_count_data = rod_model.findData('pushNodeCount') if rod_model is not None else None
        self._native_strict_barrier_enabled_data = rod_model.findData('strictLumenBarrierEnabled') if rod_model is not None else None
        self._native_debug_stretch = rod_model.findData('debugStretch') if rod_model is not None else None
        self._native_debug_kappa = rod_model.findData('debugKappa') if rod_model is not None else None
        self._native_debug_twist = rod_model.findData('debugTwist') if rod_model is not None else None
        self._native_debug_edge_length = rod_model.findData('debugEdgeLengthMm') if rod_model is not None else None
        self._native_debug_abnormal_edge_index = rod_model.findData('debugAbnormalEdgeIndex') if rod_model is not None else None
        self._native_debug_abnormal_edge_length_mm = rod_model.findData('debugAbnormalEdgeLengthMm') if rod_model is not None else None
        self._native_debug_abnormal_edge_ref_length_mm = rod_model.findData('debugAbnormalEdgeRefLengthMm') if rod_model is not None else None
        self._native_debug_max_axial_boundary_error = rod_model.findData('debugMaxAxialBoundaryErrorMm') if rod_model is not None else None
        self._native_debug_max_lateral_boundary_error = rod_model.findData('debugMaxLateralBoundaryErrorMm') if rod_model is not None else None
        self._native_debug_max_internal_force = rod_model.findData('debugMaxInternalForceN') if rod_model is not None else None
        self._native_debug_max_stretch_force = rod_model.findData('debugMaxStretchForceN') if rod_model is not None else None
        self._native_debug_max_boundary_force = rod_model.findData('debugMaxBoundaryForceN') if rod_model is not None else None
        self._native_debug_max_boundary_torque = rod_model.findData('debugMaxBoundaryTorqueNm') if rod_model is not None else None
        self._native_debug_drive_reaction = rod_model.findData('debugDriveReactionN') if rod_model is not None else None
        self._native_debug_max_bend_residual = rod_model.findData('debugMaxBendResidual') if rod_model is not None else None
        self._native_debug_min_lumen_clearance_mm = rod_model.findData('debugMinLumenClearanceMm') if rod_model is not None else None
        self._native_debug_barrier_force_vector = rod_model.findData('debugBarrierForceVector') if rod_model is not None else None
        self._native_debug_barrier_active_node_count = rod_model.findData('debugBarrierActiveNodeCount') if rod_model is not None else None
        self._native_debug_max_head_stretch = rod_model.findData('debugMaxHeadStretch') if rod_model is not None else None
        self._native_virtual_sheath_target_pos = (
            native_virtual_sheath_target_mech.findData('position') if native_virtual_sheath_target_mech is not None else None
        )
        self._native_virtual_sheath_target_rest = (
            native_virtual_sheath_target_mech.findData('rest_position') if native_virtual_sheath_target_mech is not None else None
        )
        self._native_virtual_sheath_indices = [int(i) for i in ([] if native_virtual_sheath_indices is None else native_virtual_sheath_indices)]
        self._native_virtual_sheath_offsets_mm = np.asarray(
            [] if native_virtual_sheath_offsets_mm is None else native_virtual_sheath_offsets_mm,
            dtype=float,
        ).reshape(-1)
        self._native_virtual_sheath_stiffnesses = np.asarray(
            [] if native_virtual_sheath_stiffnesses is None else native_virtual_sheath_stiffnesses,
            dtype=float,
        ).reshape(-1)
        if (
            self._native_virtual_sheath_offsets_mm.size != len(self._native_virtual_sheath_indices)
            or self._native_virtual_sheath_stiffnesses.size != len(self._native_virtual_sheath_indices)
        ):
            count = min(
                len(self._native_virtual_sheath_indices),
                int(self._native_virtual_sheath_offsets_mm.size),
                int(self._native_virtual_sheath_stiffnesses.size),
            )
            self._native_virtual_sheath_indices = self._native_virtual_sheath_indices[:count]
            self._native_virtual_sheath_offsets_mm = self._native_virtual_sheath_offsets_mm[:count]
            self._native_virtual_sheath_stiffnesses = self._native_virtual_sheath_stiffnesses[:count]
        if (
            (self._native_virtual_sheath_target_pos is None or len(self._native_virtual_sheath_indices) == 0)
            and self._native_debug_drive_reaction is None
        ):
            self.enable_native_virtual_sheath = False
            self.enable_native_thrust_limit = False
        self._native_virtual_sheath_configured_arc_mm = float(ELASTICROD_SHEATH_LENGTH_MM)
        self._native_virtual_sheath_actual_arc_mm = float(self._native_virtual_sheath_offsets_mm[-1]) if self._native_virtual_sheath_offsets_mm.size else 0.0

        wire_pos = np.asarray(_read(self._pos), dtype=float)
        self.node_count = int(wire_pos.shape[0])
        self.mid_index = max(self.node_count // 2, 0)
        self.drive_count = int(min(max(1, drive_node_count), self.node_count))
        self.magnetic_head_edge_count = int(
            ELASTICROD_MAGNETIC_HEAD_EDGES if self.is_native_backend else MAGNETIC_HEAD_EDGES
        )
        self.distal_visual_node_count = int(
            ELASTICROD_DISTAL_VISUAL_NODE_COUNT if self.is_native_backend else DISTAL_VISUAL_NODE_COUNT
        )
        self.distal_indices = list(range(max(0, self.node_count - self.distal_visual_node_count), self.node_count))

        self.initial_wire_centers = wire_pos[:, :3].copy()
        self.node_initial_path_s_mm = np.asarray(node_initial_path_s_mm, dtype=float).reshape(-1)
        self.tip_path_s0 = float(max(self.node_initial_path_s_mm[-1], 0.0))
        self.estimated_push_mm = 0.0
        if self.is_native_backend and self.use_native_entry_push_band:
            self.drive_reference_indices = [int(i) for i in self._native_entry_push_indices()]
        elif self.is_native_backend and self.is_native_strict and bool(ELASTICROD_STRICT_SIMPLE_TAIL_DRIVE):
            # strict simple tail drive means only the explicit ConstantForceField stays on the tail.
            # The physical insertion progress reference must still follow the native entry/drive window,
            # otherwise the tail barely moving can falsely inflate backlog and self-throttle the tail push.
            self.drive_reference_indices = [
                int(i) for i in self.native_drive_window_indices if 0 <= int(i) < self.node_count
            ]
            if len(self.drive_reference_indices) == 0:
                self.drive_reference_indices = [int(i) for i in self.tail_push_indices if 0 <= int(i) < self.node_count]
        else:
            self.drive_reference_indices = [0] if self.is_native_backend else list(range(self.drive_count))
        if len(self.drive_reference_indices) == 0:
            self.drive_reference_indices = [0] if self.is_native_backend else list(range(self.drive_count))
        self.drive_reference_index = self.drive_reference_indices[-1]
        self.drive_reference_pos0 = wire_pos[self.drive_reference_index, :3].copy()
        total_length_mm = NATIVE_WIRE_TOTAL_LENGTH_MM if self.is_native_backend else WIRE_TOTAL_LENGTH_MM
        self.rest_spacing_mm = total_length_mm / max(self.node_count - 1, 1)
        if self.is_native_backend:
            if self.is_native_strict:
                self.native_support_indices = []
            elif self._requested_native_support_indices:
                self.native_support_indices = sorted({
                    int(i) for i in self._requested_native_support_indices if 0 <= int(i) < self.node_count
                })
            else:
                native_support_count = self.drive_count
                if self._native_push_node_count_data is not None:
                    try:
                        native_support_count = int(self._native_push_node_count_data.value)
                    except Exception:
                        native_support_count = self.drive_count
                self.native_support_indices = list(range(min(self.node_count, max(native_support_count, 2))))
            self.native_support_count = len(self.native_support_indices)
        else:
            self.native_support_count = self.drive_count
            self.native_support_indices = list(range(self.native_support_count))

        self.step_count = 0
        self.nominal_push_force_total = float(PUSH_FORCE_INITIAL_TOTAL)
        if self.is_native_backend and self.use_native_entry_push_band:
            self.nominal_push_force_total = max(self.nominal_push_force_total, 0.20)
        self.push_force_scale = 1.0
        self.push_force_calibrated = False
        self.push_force_calibration_time = 0.0
        self.commanded_push_mm = 0.0
        self.filtered_tip_forward_speed_mm_s = 0.0
        self.tip_forward_speed_mm_s = 0.0
        self.drive_push_mm = 0.0
        self.base_progress_mm = 0.0
        self.mid_progress_mm = 0.0
        self.tip_axial_progress_mm = 0.0
        self.tip_progress_raw_mm = 0.0
        self.tip_progress_mm = 0.0
        self.prev_tip_pos = wire_pos[-1, :3].copy()
        _, self.prev_tip_proj_s_mm = self._project_to_centerline(self.prev_tip_pos)
        self.wall_contact_active = False
        self.wall_contact_clearance_mm = float('inf')
        self.surface_wall_contact_clearance_mm = float('inf')
        self._strict_wall_contact_release_counter = 0
        self._strict_wall_contact_enter_step: int | None = None
        self._strict_wall_contact_release_step: int | None = None
        self.tip_contact_correction_mm = 0.0
        self._surface_probe_cache_step = -1
        self._surface_probe_cache: list[tuple[int, float, np.ndarray, np.ndarray]] = []
        self._surface_edge_probe_cache_step = -1
        self._surface_edge_probe_cache: list[tuple[int, float, np.ndarray, float, np.ndarray, np.ndarray]] = []
        self._surface_min_clearance_cache_step = -1
        self._surface_min_clearance_cache = float('inf')
        self._head_surface_clearance_cache_step = -1
        self._head_surface_clearance_cache = float('inf')
        self._head_wall_clearance_cache_step = -1
        self._head_wall_clearance_cache = float('inf')
        self._head_wall_clearance_exact_step = -1
        self._native_strict_min_lumen_clearance_cache_step = -1
        self._native_strict_min_lumen_clearance_cache = float('inf')
        self._native_strict_barrier_active_cache_step = -1
        self._native_strict_barrier_active_cache = False
        self._strict_gui_force_exact_surface_until_step = -1
        self._geometry_cache_step = -1
        self._geometry_cache_rigid: np.ndarray | None = None
        self._geometry_cache_points_mm: np.ndarray | None = None
        self._geometry_cache_tip_projection: tuple[np.ndarray, float] | None = None
        self._strict_external_push_cache_step = -1
        self._strict_external_push_cache: list[int] = []
        self._native_strict_support_exhausted = False
        self._native_visual_sync_targets: list[tuple[str, object, np.ndarray, object | None]] = []
        self.steering_misaligned = False
        self.steering_angle_deg = 0.0
        self.beam_compression_active = False
        self.beam_compression_mm = 0.0
        self.beam_stall_active = False

        fast_lumen_face_candidate_count = (
            self.vessel_surface_query_face_candidate_count
            if self.is_native_strict and self.use_native_gui_wallclock_control
            else 1024
        )
        self.fast_lumen_profile_mm = (
            _lumen_profile(
                self.centerline[:, :3],
                self.vessel_vertices[:, :3],
                self.vessel_faces[:, :3],
                face_candidate_count=fast_lumen_face_candidate_count,
            )
            if self.vessel_vertices.ndim == 2 and self.vessel_vertices.shape[0] >= 8
            else np.zeros(0, dtype=float)
        )
        self.use_fast_lumen = self.fast_lumen_profile_mm.shape[0] == self.centerline.shape[0]
        self.entry_point = self.centerline[0, :3].copy()
        self.entry_radius_mm = (
            _opening_radius(self.vessel_vertices[:, :3], self.entry_point, self.insertion_direction)
            if self.vessel_vertices.ndim == 2 and self.vessel_vertices.shape[0] > 0
            else 2.4
        )
        self.pre_entry_guard_trigger_mm = max(CONTACT_ALARM_DISTANCE_MM + 0.5 * WIRE_RADIUS_MM, 0.75)
        if self.vessel_vertices.ndim == 2 and self.vessel_vertices.shape[0] > 0:
            verts = np.asarray(self.vessel_vertices[:, :3], dtype=float)
            rel = verts - self.entry_point.reshape(1, 3)
            axial = rel @ self.insertion_direction.reshape(3, 1)
            radial = rel - np.outer(axial.reshape(-1), self.insertion_direction)
            axial_pad = max(2.0 * self.rest_spacing_mm, 4.0)
            radial_limit = max(self.entry_radius_mm + 3.0 * WIRE_RADIUS_MM, self.entry_radius_mm + 1.0)
            mask = (
                (axial.reshape(-1) >= -max(float(BEAM_PRE_ENTRY_ACCESS_GUIDE_MM), 0.0) - axial_pad)
                & (axial.reshape(-1) <= axial_pad)
                & (np.linalg.norm(radial, axis=1) <= radial_limit)
            )
            selected = verts[mask]
            if selected.shape[0] < 128:
                stride = max(1, int(np.ceil(verts.shape[0] / 12000.0)))
                selected = verts[::stride]
            self.pre_entry_guard_vertices = selected
        else:
            self.pre_entry_guard_vertices = np.zeros((0, 3), dtype=float)
        self.force_arrow_anchor = np.asarray(
            force_arrow_anchor if force_arrow_anchor is not None else (self.entry_point + np.array([-12.0, 18.0, 0.0], dtype=float)),
            dtype=float,
        ).reshape(3)
        self.camera_follow_offset = np.asarray(
            camera_follow_offset if camera_follow_offset is not None else [24.0, -110.0, 34.0],
            dtype=float,
        ).reshape(3)

        self._force_diag_printed = False
        self._constraint_diag_printed = False
        self._push_calibration_started = False
        self._push_calibration_final_logged = False
        self.use_python_magnetic_fallback = bool(use_python_magnetic_fallback)
        self.python_magnetic_force_gain = float(python_magnetic_force_gain)
        self._fallback_target_point = self.entry_point.copy()
        self._fallback_ba_vector = self.insertion_direction.copy()
        self._fallback_force_vector = self.insertion_direction.copy()
        self._fallback_nav_s_mm = float(self.tip_path_s0)
        self._fallback_head_model = None
        self._fallback_field_controller = None
        self._fallback_torque_weights = np.zeros(0, dtype=float)
        self.sim_time_s = 0.0
        self._native_control_dt_s = 0.0
        self._native_control_time_mode = 'solver'
        self._native_gui_wallclock_last_s: float | None = None
        self._diagnostic_prev_centers = wire_pos[:, :3].copy()
        self._native_safe_last_stable_pos = np.array(_read(self._pos), dtype=float, copy=True) if self.is_native_backend else None
        self._native_safe_last_stable_vel = np.array(_read(self._vel), dtype=float, copy=True) if self.is_native_backend else None
        self._native_safe_last_stable_rod_pos = (
            np.array(_read(self._rod_state_pos), dtype=float, copy=True)
            if self.is_native_backend and self._rod_state_pos is not None else None
        )
        self._native_safe_last_stable_rod_vel = (
            np.array(_read(self._rod_state_vel), dtype=float, copy=True)
            if self.is_native_backend and self._rod_state_vel is not None else None
        )
        self._native_safe_last_stable_rod_free_pos = (
            np.array(_read(self._rod_state_free_pos), dtype=float, copy=True)
            if self.is_native_backend and self._rod_state_free_pos is not None else None
        )
        self._native_safe_last_stable_rod_free_vel = (
            np.array(_read(self._rod_state_free_vel), dtype=float, copy=True)
            if self.is_native_backend and self._rod_state_free_vel is not None else None
        )
        self._native_safe_recovery_cooldown = 0
        self._native_safe_displacement_push_last_mm = 0.0
        self._native_safe_displacement_push_logged = False
        self._native_safe_last_recovery_step = -1
        self._native_safe_last_recovery_kind = ''
        self._native_safe_distal_recovery_streak = 0
        self._native_first_contact_step: int | None = None
        self._native_max_post_contact_jump_mm = 0.0
        self._native_stall_logged = False
        self._native_last_severe_step = -1
        self._native_failfast_triggered = False
        self._native_last_strict_guard_step = -1
        self._native_last_barrier_deficit_log_step = -1
        self._native_last_barrier_deficit_mm = 0.0
        self._native_strict_hold_active_this_step = False
        self._native_runtime_band = 'free' if self.is_native_realtime else 'quality'
        if self.is_native_realtime:
            (
                self._native_runtime_dt_s,
                self._native_runtime_solver_max_iter,
                self._native_runtime_solver_tolerance,
                self._native_runtime_speed_scale,
            ) = self._native_realtime_band_settings('free')
        else:
            self._native_runtime_dt_s = float(ELASTICROD_DT_S)
            self._native_runtime_solver_max_iter = int(ELASTICROD_CONSTRAINT_SOLVER_MAX_ITER)
            self._native_runtime_solver_tolerance = float(ELASTICROD_CONSTRAINT_SOLVER_TOLERANCE)
            self._native_runtime_speed_scale = 1.0
        self._native_runtime_settings_applied = False
        self._native_runtime_last_visual_step = -1
        self._native_runtime_last_visual_band = None
        self._native_runtime_band_reason = 'initial'
        self._native_virtual_sheath_paused = False
        self._native_virtual_sheath_pause_reason = ''
        self._native_virtual_sheath_last_reaction_n = 0.0
        self._native_axial_assist_mode = ''
        self._push_scale_reason = 'initial'
        self._native_strict_driver_limited = False
        self._native_strict_driver_limit_reason = ''
        self.native_strict_barrier_enabled = bool(
            self.is_native_backend
            and ELASTICROD_STRICT_NATIVE_LUMEN_BARRIER
            and self._native_strict_barrier_enabled_data is not None
            and bool(self._native_strict_barrier_enabled_data.value)
        )
        self.native_kinematic_sheath_driver = bool(
            self._native_use_kinematic_support is not None and bool(self._native_use_kinematic_support.value)
        )
        if self.use_python_magnetic_fallback:
            head_edge_count = int(max(1, self.magnetic_head_edge_count))
            self._fallback_head_model = DistalMagneticHeadModel(
                moment_magnitude=float(python_magnetic_moment),
                head_edge_count=head_edge_count,
            )
            self._fallback_field_controller = UniformMagneticFieldController(
                strength=float(python_magnetic_field_strength),
                moment_magnitude=float(python_magnetic_moment),
                smoothing_alpha=float(np.clip(MAGNETIC_FIELD_SMOOTHING_ALPHA, 0.0, 1.0)),
                initial_direction=self.insertion_direction,
            )
            self._fallback_torque_weights = self._fallback_head_model.distribution_weights(len(self.distal_indices))
        if self._native_insertion_direction is not None:
            self._native_insertion_direction.value = self.insertion_direction.tolist()
        self._refresh_native_visual_sync_targets()
        self._update_target_marker(self.entry_point)
        self._write_native_virtual_sheath_targets()

    def _node_s(self, idx: int) -> float:
        return float(self.node_initial_path_s_mm[idx] + self.estimated_push_mm)

    def _contact_radius_mm(self) -> float:
        return float(NATIVE_WIRE_RADIUS_MM if self.is_native_backend else WIRE_RADIUS_MM)

    def _set_forcefield_indices(self, data, indices: list[int]) -> None:
        if data is None:
            return
        try:
            data.value = [int(i) for i in indices]
        except Exception:
            return

    def _set_active_native_push_indices(self, indices: list[int], *, update_reference: bool = True) -> list[int]:
        clean = [int(i) for i in indices if 0 <= int(i) < self.node_count]
        if len(clean) == 0:
            return []
        self._active_native_push_indices = clean
        if self.is_native_backend and update_reference:
            self.drive_count = len(clean)
            self.drive_reference_indices = clean.copy()
            self.drive_reference_index = clean[-1]
            self.drive_reference_pos0 = self.initial_wire_centers[self.drive_reference_index, :3].copy()
        return clean

    def _invalidate_geometry_cache(self) -> None:
        self._geometry_cache_step = -1
        self._geometry_cache_rigid = None
        self._geometry_cache_points_mm = None
        self._geometry_cache_tip_projection = None
        self._strict_external_push_cache_step = -1
        self._strict_external_push_cache = []
        self._head_wall_clearance_cache_step = -1

    def _current_rigid_state(self) -> np.ndarray:
        if self._geometry_cache_step == self.step_count and self._geometry_cache_rigid is not None:
            return self._geometry_cache_rigid
        rigid = np.asarray(_read(self._pos), dtype=float)
        self._geometry_cache_step = self.step_count
        self._geometry_cache_rigid = rigid
        self._geometry_cache_points_mm = None
        self._geometry_cache_tip_projection = None
        return rigid

    def _current_points_mm(self) -> np.ndarray:
        if self._geometry_cache_step == self.step_count and self._geometry_cache_points_mm is not None:
            return self._geometry_cache_points_mm
        if self.is_native_backend and self._rod_state_pos is not None:
            rod = np.asarray(_read(self._rod_state_pos), dtype=float)
            points_mm = 1000.0 * rod[:, :3] if rod.ndim == 2 and rod.shape[1] >= 3 else np.zeros((0, 3), dtype=float)
        else:
            rigid = self._current_rigid_state()
            points_mm = rigid[:, :3] if rigid.ndim == 2 and rigid.shape[1] >= 3 else np.zeros((0, 3), dtype=float)
        self._geometry_cache_step = self.step_count
        self._geometry_cache_points_mm = np.asarray(points_mm, dtype=float)
        return self._geometry_cache_points_mm

    def _tip_projection_to_centerline(self) -> Tuple[np.ndarray, float]:
        if self._geometry_cache_step == self.step_count and self._geometry_cache_tip_projection is not None:
            proj_point, proj_s = self._geometry_cache_tip_projection
            return np.asarray(proj_point, dtype=float).copy(), float(proj_s)
        rigid = self._current_rigid_state()
        if rigid.ndim != 2 or rigid.shape[0] == 0 or self.centerline.shape[0] == 0:
            projection = (self.entry_point.copy(), 0.0)
        else:
            tip_pos = np.asarray(rigid[-1, :3], dtype=float).reshape(3)
            projection = self._project_to_centerline(tip_pos)
        self._geometry_cache_step = self.step_count
        self._geometry_cache_tip_projection = (np.asarray(projection[0], dtype=float).copy(), float(projection[1]))
        proj_point, proj_s = self._geometry_cache_tip_projection
        return np.asarray(proj_point, dtype=float).copy(), float(proj_s)

    def _refresh_native_visual_sync_targets(self) -> None:
        self._native_visual_sync_targets = []
        if self.wire_mech is None:
            return
        gw_node = self.wire_mech.getContext()
        if gw_node is None:
            return
        for child_name in ('BodyVisual', 'MagneticHead'):
            try:
                child = gw_node.getChild(child_name)
                if child is None:
                    continue
                dofs = child.getObject('dofs')
                mapping_obj = child.getObject('RigidMapping')
                if dofs is None or mapping_obj is None:
                    continue
                vis_dofs_data = dofs.findData('position')
                idx_data = mapping_obj.findData('rigidIndexPerPoint')
                if vis_dofs_data is None or idx_data is None:
                    continue
                indices = np.asarray(idx_data.value, dtype=int).ravel()
                if indices.size == 0:
                    continue
                valid = (indices >= 0) & (indices < self.node_count)
                indices = indices[valid]
                if indices.size == 0:
                    continue
                ogl_pos = None
                ogl_child = child.getChild('Model') or child.getChild('Visual')
                if ogl_child is not None:
                    ogl_obj = ogl_child.getObject('vis')
                    if ogl_obj is not None:
                        ogl_pos = ogl_obj.findData('position')
                self._native_visual_sync_targets.append((child_name, vis_dofs_data, indices, ogl_pos))
            except Exception:
                continue

    def _native_entry_push_indices(self) -> list[int]:
        if not self.use_native_entry_push_band or self.node_count <= 0:
            return [int(i) for i in self._active_native_push_indices] if self._active_native_push_indices else list(range(self.drive_count))

        if self.is_native_safe:
            spacing_mm = float(getattr(
                self,
                'rest_spacing_mm',
                (NATIVE_WIRE_TOTAL_LENGTH_MM if self.is_native_backend else WIRE_TOTAL_LENGTH_MM) / max(self.node_count - 1, 1),
            ))
            back_mm = max(
                float(ELASTICROD_AXIAL_PATH_ASSIST_WINDOW_BACK_MM),
                max(2.0 * spacing_mm, 1.0),
            )
            front_mm = max(float(ELASTICROD_AXIAL_PATH_ASSIST_WINDOW_FRONT_MM), 0.0)
            min_count = int(max(ELASTICROD_AXIAL_PATH_ASSIST_WINDOW_MIN_NODE_COUNT, 2))
            s_min = -back_mm
            s_max = front_mm
            candidates = [i for i in range(self.node_count) if s_min - 1.0e-9 <= self._node_s(i) <= s_max + 1.0e-9]
            if len(candidates) >= min_count:
                return self._set_active_native_push_indices(candidates)

            target_s = 0.5 * (s_min + s_max)
            ranked = sorted(
                list(range(self.node_count)),
                key=lambda idx: (
                    abs(self._node_s(idx) - target_s),
                    abs(self._node_s(idx)),
                    -self._node_s(idx),
                    idx,
                ),
            )
            fallback = sorted(ranked[: min(min_count, len(ranked))])
            if fallback:
                return self._set_active_native_push_indices(fallback)
            return [int(i) for i in self._active_native_push_indices] if self._active_native_push_indices else list(range(self.drive_count))

        s_max = -self.native_entry_push_band_outside_offset_mm
        s_min = -(self.native_entry_push_band_outside_offset_mm + self.native_entry_push_band_length_mm)
        candidates = [i for i in range(self.node_count) if s_min - 1.0e-9 <= self._node_s(i) <= s_max + 1.0e-9]
        if len(candidates) >= self.native_entry_push_band_min_node_count:
            return self._set_active_native_push_indices(candidates)

        outside = [i for i in range(self.node_count) if self._node_s(i) <= s_max + 1.0e-9]
        ranked = sorted(
            outside if outside else list(range(min(self.node_count, max(self.native_entry_push_band_min_node_count, 1)))),
            key=lambda idx: (
                abs(self._node_s(idx) + self.native_entry_push_band_outside_offset_mm + 0.5 * self.native_entry_push_band_length_mm),
                abs(self._node_s(idx) - s_max),
                idx,
            ),
        )
        fallback = sorted(ranked[: min(self.native_entry_push_band_min_node_count, len(ranked))])
        if fallback:
            return self._set_active_native_push_indices(fallback)
        return [int(i) for i in self._active_native_push_indices] if self._active_native_push_indices else list(range(self.drive_count))

    def _current_node_centerline_s_mm(self) -> np.ndarray:
        points_mm = self._current_points_mm()
        count = min(int(self.node_count), int(points_mm.shape[0]))
        if count <= 0:
            return np.zeros(0, dtype=float)
        if self.centerline.shape[0] < 2:
            return np.asarray([self._node_s(i) for i in range(count)], dtype=float)

        proj_s = np.zeros(count, dtype=float)
        for idx in range(count):
            _, s = self._project_to_centerline(points_mm[idx, :3])
            proj_s[idx] = float(s)
        return proj_s

    def _native_axial_assist_targets(self, deficit_mm: float) -> tuple[list[tuple[int, float]], bool]:
        if self.is_native_safe:
            default_indices = (
                self._native_entry_push_indices()
                if self.use_native_entry_push_band
                else [int(i) for i in self.native_axial_assist_indices]
            )
            entry_front_limit_mm = max(0.25 * float(self.rest_spacing_mm), 0.75)
            trimmed_indices = [
                int(idx) for idx in default_indices
                if 0 <= int(idx) < self.node_count and float(self._node_s(int(idx))) <= entry_front_limit_mm + 1.0e-9
            ]
            if len(trimmed_indices) >= max(min(int(ELASTICROD_AXIAL_PATH_ASSIST_WINDOW_MIN_NODE_COUNT), 2), 1):
                default_indices = trimmed_indices
        else:
            default_indices = (
                self._native_entry_push_indices()
                if self.use_native_entry_push_band
                else [int(i) for i in self.native_axial_assist_indices]
            )
        current_proj_s = self._current_node_centerline_s_mm() if self.is_native_safe else np.zeros(0, dtype=float)
        default_targets = []
        for idx in default_indices:
            idx = int(idx)
            if not (0 <= idx < self.node_count):
                continue
            nominal_s = float(self._node_s(idx))
            if self.is_native_safe and idx < current_proj_s.size and np.isfinite(float(current_proj_s[idx])):
                path_s = max(float(current_proj_s[idx]), nominal_s, 0.0)
            else:
                path_s = max(nominal_s, 0.0)
            default_targets.append((idx, path_s))
        return default_targets, False

    def _native_safe_displacement_push_targets(self) -> list[tuple[int, float, float]]:
        if (not self.use_native_displacement_feed) or self.node_count <= 0:
            return []

        if self.is_native_strict:
            base_indices = [
                idx for idx in range(self.node_count)
                if float(self._node_s(idx)) <= 1.0e-9
            ]
            if not base_indices:
                base_indices = sorted({
                    int(i) for i in self._strict_hand_push_indices()
                    if 0 <= int(i) < self.node_count
                })
            if not base_indices:
                return []
            support_length_mm = max(
                float(self.external_support_length_mm),
                float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM),
                float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_LENGTH_MM),
            )
            transition_span_mm = max(1.5 * float(self.rest_spacing_mm), 3.0)
            support_span_mm = max(min(support_length_mm, transition_span_mm), 0.5 * float(self.rest_spacing_mm), 1.0)
            targets: list[tuple[int, float, float]] = []
            for idx in base_indices:
                support_depth_mm = max(-float(self._node_s(idx)), 0.0)
                if support_depth_mm > support_length_mm + 1.0e-9:
                    centering_alpha = 0.0
                else:
                    centering_alpha = 0.72 + 0.28 * float(np.clip(support_depth_mm / support_span_mm, 0.0, 1.0))
                targets.append((idx, 1.0, float(np.clip(centering_alpha, 0.0, 1.0))))

            anchor = max(base_indices)
            transition_count = max(int(np.ceil(transition_span_mm / max(float(self.rest_spacing_mm), 1.0e-6))), 2)
            transition_end = min(self.node_count, anchor + transition_count + 1)
            for idx in range(anchor + 1, transition_end):
                material_s_mm = max(float(self._node_s(idx)), 0.0)
                if material_s_mm > transition_span_mm + 1.0e-9:
                    break
                transition_u = float(np.clip(material_s_mm / max(transition_span_mm, 1.0e-6), 0.0, 1.0))
                alpha = float(np.clip(1.0 - 0.80 * transition_u, 0.20, 1.0))
                if alpha <= 1.0e-6:
                    continue
                # Once material has entered the lumen, the external tail feed
                # should stop dragging it back to the centerline. The previous
                # blend kept recentring several millimetres inside the vessel,
                # which fought magnetic steering and created hook-like kinks
                # near the first bend. Keep only a tiny entry collar to avoid a
                # numerical snag exactly at the ostium, then hand control to the
                # rod/contact physics.
                entry_collar_mm = max(0.75 * float(self.rest_spacing_mm), 0.8)
                if material_s_mm <= entry_collar_mm + 1.0e-9:
                    collar_u = float(np.clip(material_s_mm / max(entry_collar_mm, 1.0e-6), 0.0, 1.0))
                    centering_alpha = float(np.clip(0.10 * (1.0 - collar_u), 0.0, 0.10))
                else:
                    centering_alpha = 0.0
                targets.append((idx, alpha, centering_alpha))
            return targets

        def centering_weight(idx: int, alpha: float) -> float:
            return 0.0

        base_indices = sorted({
            int(i) for i in self._native_entry_push_indices()
            if 0 <= int(i) < self.node_count
        })
        if not base_indices:
            return []

        if self.node_initial_path_s_mm.size == 0:
            return [(idx, 1.0, centering_weight(idx, 1.0)) for idx in base_indices]

        count = min(self.node_count, int(self.node_initial_path_s_mm.size))
        nominal_s = self.node_initial_path_s_mm[:count] + float(self.commanded_push_mm)
        front_mm = max(float(ELASTICROD_AXIAL_PATH_ASSIST_WINDOW_FRONT_MM), 0.0)
        front_denom_mm = max(front_mm, 0.5 * float(self.rest_spacing_mm), 1.0e-6)
        targets: list[tuple[int, float, float]] = []
        for idx in base_indices:
            if idx >= count:
                continue
            node_s = float(nominal_s[idx])
            if node_s <= 0.0:
                alpha = 1.0
            else:
                alpha = float(np.clip(1.0 - node_s / front_denom_mm, 0.0, 1.0))
            if alpha <= 1.0e-6:
                continue
            targets.append((idx, alpha, centering_weight(idx, alpha)))
        return targets

    def _update_displacement_push(self, dt: float) -> None:
        if (
            (not self.use_native_displacement_feed)
            or dt <= 0.0
            or self._rod_state_pos is None
        ):
            return

        push_targets = self._native_safe_displacement_push_targets()
        if not push_targets:
            return

        push_speed_mm_s = max(float(ELASTICROD_DISPLACEMENT_PUSH_VELOCITY_MM_PER_S), 0.0)
        if push_speed_mm_s <= 1.0e-9:
            return

        scale = float(
            np.clip(
                self._native_startup_ramp_scale() * float(np.clip(self.push_force_scale, 0.0, self._max_push_scale_allowed())),
                0.0,
                self._max_push_scale_allowed(),
            )
        )
        if scale <= 1.0e-9:
            return

        drive_lead_cap_mm = max(1.5 * float(self.rest_spacing_mm), 4.0)
        progress_anchor_mm = max(
            float(self.tip_progress_raw_mm),
            min(float(self.drive_push_mm), float(self.tip_progress_raw_mm) + drive_lead_cap_mm),
            0.0,
        )
        backlog_mm = float(max(self.commanded_push_mm - progress_anchor_mm, 0.0))
        if backlog_mm <= 1.0e-6:
            return

        shift_mm = float(min(backlog_mm, push_speed_mm_s * scale * float(dt)))
        if self.is_native_strict:
            catchup_threshold_mm = max(1.5 * float(self.rest_spacing_mm), 1.0)
            catchup_mm = max(backlog_mm - catchup_threshold_mm, 0.0)
            if catchup_mm > 0.0:
                shift_mm = float(
                    min(
                        backlog_mm,
                        shift_mm + min(
                            0.35 * catchup_mm,
                            max(0.75 * float(self.rest_spacing_mm), 0.60),
                        ),
                    )
                )
        if shift_mm <= 1.0e-6:
            return

        if self.is_native_strict:
            fresh_contact_hold = False
            if (
                self.wall_contact_active
                and self._strict_wall_contact_enter_step is not None
                and self.step_count >= self._strict_wall_contact_enter_step
            ):
                fresh_contact_hold = (self.step_count - self._strict_wall_contact_enter_step) <= 12
            if fresh_contact_hold:
                self._native_strict_hold_active_this_step = True
                return
            if self.wall_contact_active:
                self._native_strict_hold_active_this_step = True
                return

            physical_gap_mm = float(self._native_strict_physical_contact_clearance_mm())
            if np.isfinite(physical_gap_mm):
                near_contact_alpha = float(np.clip((physical_gap_mm - 0.05) / 0.25, 0.0, 1.0))
                shift_mm *= 0.15 + 0.85 * near_contact_alpha

            head_stretch = float(self._native_strict_max_head_stretch())
            soft_limit, hard_limit = self._native_strict_head_stretch_limits()
            if self.wall_contact_active and head_stretch >= soft_limit:
                self._native_strict_hold_active_this_step = True
                return
            if hard_limit > soft_limit:
                stretch_alpha = float(np.clip((head_stretch - soft_limit) / (hard_limit - soft_limit), 0.0, 1.0))
                shift_mm *= 1.0 - 0.85 * stretch_alpha

            if shift_mm <= 1.0e-6:
                self._native_strict_hold_active_this_step = True
                return

        rod_pos = np.array(_read(self._rod_state_pos), dtype=float, copy=True)
        if rod_pos.ndim != 2 or rod_pos.shape[0] == 0 or rod_pos.shape[1] < 3:
            return
        rod_free_pos = (
            np.array(_read(self._rod_state_free_pos), dtype=float, copy=True)
            if self._rod_state_free_pos is not None
            else None
        )
        rod_vel = (
            np.array(_read(self._rod_state_vel), dtype=float, copy=True)
            if self._rod_state_vel is not None
            else None
        )
        rod_free_vel = (
            np.array(_read(self._rod_state_free_vel), dtype=float, copy=True)
            if self._rod_state_free_vel is not None
            else None
        )

        centers_mm = 1000.0 * rod_pos[:, :3].copy()
        axis = _normalize(np.asarray(self.insertion_direction, dtype=float).reshape(3))
        if np.linalg.norm(axis) < 1.0e-12:
            axis = DEFAULT_INSERTION_DIR.copy()
        entry = self.entry_point.reshape(3)
        support_length_mm = max(
            float(self.external_support_length_mm),
            float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM),
            float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_LENGTH_MM),
        )
        desired_vel_m_s = 1.0e-3 * (shift_mm / max(float(dt), 1.0e-9)) * axis

        moved = 0
        for idx, push_alpha, centering_alpha in push_targets:
            if idx < 0 or idx >= centers_mm.shape[0]:
                continue
            current = centers_mm[idx, :3]
            material_s_mm = float(self._node_s(idx))
            if (
                self.is_native_strict
                and material_s_mm > 0.0
                and self.centerline.shape[0] >= 2
                and self.path_len > 1.0e-9
            ):
                current_s_mm = max(material_s_mm, 0.0)
                target_s_mm = float(np.clip(current_s_mm + push_alpha * shift_mm, 0.0, self.path_len))
                tangent = self._centerline_tangent(current_s_mm)
                advected_point = current + push_alpha * shift_mm * tangent
                target_point, _, _ = self._nominal_centerline_frame(target_s_mm)
                new_point = (1.0 - centering_alpha) * advected_point + centering_alpha * target_point
            else:
                rel = current - entry
                axial_mm = float(np.dot(rel, axis))
                axis_point = entry + (axial_mm + push_alpha * shift_mm) * axis
                radial = rel - axial_mm * axis
                new_point = axis_point + (1.0 - centering_alpha) * radial
            if (
                self.is_native_strict
                and material_s_mm <= 0.0
                and material_s_mm >= -(support_length_mm + 1.0e-9)
            ):
                new_point = self._strict_project_inside_external_support(new_point)
            if float(np.linalg.norm(new_point - current)) <= 1.0e-9:
                continue

            centers_mm[idx, :3] = new_point
            rod_pos[idx, :3] = 1.0e-3 * new_point
            if rod_free_pos is not None and idx < rod_free_pos.shape[0] and rod_free_pos.shape[1] >= 3:
                rod_free_pos[idx, :3] = 1.0e-3 * new_point
            if rod_vel is not None and idx < rod_vel.shape[0] and rod_vel.shape[1] >= 3:
                rod_vel[idx, :3] = push_alpha * desired_vel_m_s
                if rod_vel.shape[1] > 3:
                    rod_vel[idx, 3:] *= max(0.0, 1.0 - centering_alpha)
            if rod_free_vel is not None and idx < rod_free_vel.shape[0] and rod_free_vel.shape[1] >= 3:
                rod_free_vel[idx, :3] = push_alpha * desired_vel_m_s
                if rod_free_vel.shape[1] > 3:
                    rod_free_vel[idx, 3:] *= max(0.0, 1.0 - centering_alpha)
            moved += 1

        if moved <= 0:
            return

        with _writeable(self._rod_state_pos) as out_pos:
            out_pos[:] = rod_pos
        if self._rod_state_free_pos is not None and rod_free_pos is not None:
            with _writeable(self._rod_state_free_pos) as out_free_pos:
                out_free_pos[:] = rod_free_pos
        if self._rod_state_vel is not None and rod_vel is not None:
            with _writeable(self._rod_state_vel) as out_vel:
                out_vel[:] = rod_vel
        if self._rod_state_free_vel is not None and rod_free_vel is not None:
            with _writeable(self._rod_state_free_vel) as out_free_vel:
                out_free_vel[:] = rod_free_vel

        self._native_safe_displacement_push_last_mm = shift_mm
        self._sync_native_rod_to_display()
        self._invalidate_surface_probe_cache()
        if not self._native_safe_displacement_push_logged:
            push_nodes = [idx for idx, _, _ in push_targets]
            print(
                (
                    f'[INFO] [elasticrod-safe] displacement push enabled: '
                    f'nodes={push_nodes}, speed={push_speed_mm_s:.3f} mm/s, '
                    f'entryBand=[-{ELASTICROD_AXIAL_PATH_ASSIST_WINDOW_BACK_MM:.1f}, +{ELASTICROD_AXIAL_PATH_ASSIST_WINDOW_FRONT_MM:.1f}] mm'
                    if self.is_native_safe else
                    f'[INFO] [elasticrod-strict] direct tail feed enabled: '
                    f'nodes={push_nodes}, speed={push_speed_mm_s:.3f} mm/s, '
                    f'externalSupportLength={self.external_support_length_mm:.3f} mm'
                )
            )
            self._native_safe_displacement_push_logged = True

    def _strict_hand_push_indices(self) -> list[int]:
        if not (self.is_native_backend and self.is_native_strict):
            if self.use_native_entry_push_band:
                dynamic = self._native_entry_push_indices()
                if dynamic:
                    return dynamic
            configured = [int(i) for i in self.tail_push_indices if 0 <= int(i) < self.node_count]
            if configured:
                return configured
            count = min(int(ELASTICROD_STRICT_HAND_PUSH_NODE_COUNT), self.node_count)
            return list(range(max(count, 0)))
        if self.native_strict_boundary_driver_enabled:
            return []
        return self._strict_external_push_indices()

    def _entry_axis_coordinates(self, centers_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(centers_mm, dtype=float).reshape(-1, 3)
        rel = points - self.entry_point.reshape(1, 3)
        axial = rel @ self.insertion_direction.reshape(3, 1)
        radial = rel - axial.reshape(-1, 1) * self.insertion_direction.reshape(1, 3)
        return axial.reshape(-1), np.linalg.norm(radial, axis=1)

    def _strict_external_support_clearance_mm(self, point: np.ndarray) -> float:
        if not (self.is_native_strict and self.is_native_backend):
            return float('inf')
        support_length_mm = max(
            float(self.external_support_length_mm),
            float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM),
            float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_LENGTH_MM),
        )
        support_radius_mm = max(float(self.external_support_radius_mm), 0.0)
        if support_length_mm <= 1.0e-9 or support_radius_mm <= 1.0e-9:
            return float('inf')
        p = np.asarray(point, dtype=float).reshape(3)
        rel = p - self.entry_point.reshape(3)
        axial_mm = float(np.dot(rel, self.insertion_direction))
        if axial_mm >= 0.0 or axial_mm < -support_length_mm:
            return float('inf')
        radial_vec = rel - axial_mm * self.insertion_direction
        radial_mm = float(np.linalg.norm(radial_vec))
        allowed_mm = max(
            support_radius_mm - self._contact_radius_mm() - max(self.native_strict_lumen_clamp_tolerance_mm, 0.05),
            0.0,
        )
        return float(allowed_mm - radial_mm)

    def _strict_project_inside_external_support(self, point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=float).reshape(3)
        support_length_mm = max(
            float(self.external_support_length_mm),
            float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM),
            float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_LENGTH_MM),
        )
        support_radius_mm = max(float(self.external_support_radius_mm), 0.0)
        if support_length_mm <= 1.0e-9 or support_radius_mm <= 1.0e-9:
            return p
        rel = p - self.entry_point.reshape(3)
        axial_mm = float(np.dot(rel, self.insertion_direction))
        axial_clamped_mm = float(np.clip(axial_mm, -support_length_mm, 0.0))
        axis_point = self.entry_point.reshape(3) + axial_clamped_mm * self.insertion_direction
        radial_vec = p - axis_point
        radial_mm = float(np.linalg.norm(radial_vec))
        allowed_mm = max(
            support_radius_mm - self._contact_radius_mm() - max(self.native_strict_lumen_clamp_tolerance_mm, 0.05),
            0.0,
        )
        if radial_mm <= allowed_mm + 1.0e-9:
            return axis_point.copy() if radial_mm < 1.0e-12 and allowed_mm <= 1.0e-9 else p
        if radial_mm <= 1.0e-12:
            return axis_point.copy()
        return axis_point + (allowed_mm / radial_mm) * radial_vec

    def _strict_external_push_indices(self) -> list[int]:
        if (not self.is_native_backend) or (not self.is_native_strict) or self.node_count <= 0:
            return []
        if self._strict_external_push_cache_step == self.step_count:
            return [int(i) for i in self._strict_external_push_cache]
        centers = self._current_points_mm()
        if centers.ndim != 2 or centers.shape[0] == 0:
            return []
        centers = centers[: self.node_count, :3]
        axial_mm, radial_mm = self._entry_axis_coordinates(centers)
        support_length_mm = max(
            float(self.external_support_length_mm),
            float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM),
            float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_LENGTH_MM),
        )
        support_radius_mm = max(float(self.external_support_radius_mm), 0.0)
        max_nodes = max(
            1,
            min(int(ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT), self.node_count),
        )

        outside_mask = axial_mm < -1.0e-9
        inside_support_mask = outside_mask & (axial_mm >= -support_length_mm - 1.0e-9)
        if support_radius_mm > 0.0:
            inside_support_mask &= radial_mm <= support_radius_mm + 1.0e-9

        eligible = np.flatnonzero(inside_support_mask)
        selected: list[int] = []
        if eligible.size > 0:
            anchor = int(eligible[np.argmax(axial_mm[eligible])])
            selected = [anchor]
            prev = anchor - 1
            while len(selected) < max_nodes and prev >= 0 and bool(outside_mask[prev]):
                selected.insert(0, prev)
                prev -= 1
        elif np.any(outside_mask):
            outside = np.flatnonzero(outside_mask)
            anchor = int(outside[np.argmax(axial_mm[outside])])
            selected = [anchor]
            prev = anchor - 1
            while len(selected) < max_nodes and prev >= 0 and bool(outside_mask[prev]):
                selected.insert(0, prev)
                prev -= 1

        selected = sorted(selected[:max_nodes])
        self._active_external_push_indices = selected.copy()
        self._active_native_push_indices = selected.copy()
        self._strict_external_push_cache_step = self.step_count
        self._strict_external_push_cache = selected.copy()
        return selected

    def _native_strict_support_stats(self) -> tuple[int, int]:
        if not (self.is_native_backend and self.is_native_strict and self.native_strict_boundary_driver_enabled):
            return 0, 0
        if self.node_initial_path_s_mm.size == 0 or self.node_count <= 0:
            return 0, 0

        count = min(int(self.node_count), int(self.node_initial_path_s_mm.size))
        nominal_s = self.node_initial_path_s_mm[:count] + float(self.commanded_push_mm)
        support_length_mm = max(float(ELASTICROD_STRICT_SUPPORT_WINDOW_LENGTH_MM), 0.0)
        if support_length_mm <= 1.0e-9:
            configured_drive_nodes = int(max(
                int(ELASTICROD_AXIAL_DRIVE_NODE_COUNT),
                int(ELASTICROD_PUSH_NODE_COUNT),
                1,
            ))
            return 0, min(count, configured_drive_nodes)
        support_mask = (
            (nominal_s >= -support_length_mm - 1.0e-9)
            & (nominal_s <= 0.0 + 1.0e-9)
        )

        drive_length_mm = max(float(ELASTICROD_STRICT_DRIVE_WINDOW_LENGTH_MM), 0.0)
        outside_offset_mm = max(float(ELASTICROD_STRICT_DRIVE_WINDOW_OUTSIDE_OFFSET_MM), 0.0)
        drive_max = -outside_offset_mm
        drive_min = -(outside_offset_mm + drive_length_mm)
        min_count = max(int(ELASTICROD_STRICT_DRIVE_WINDOW_MIN_NODE_COUNT), 1)

        if bool(ELASTICROD_USE_KINEMATIC_SHEATH_DRIVER):
            drive_count = int(np.count_nonzero(support_mask))
            return int(np.count_nonzero(support_mask)), drive_count

        candidate_mask = nominal_s >= -support_length_mm - 1.0e-9
        outside_candidates = np.flatnonzero(candidate_mask & (nominal_s <= drive_max + 1.0e-9))
        inside_band = outside_candidates[
            (nominal_s[outside_candidates] >= drive_min - 1.0e-9)
            & (nominal_s[outside_candidates] <= drive_max + 1.0e-9)
        ]
        if inside_band.size >= min_count:
            drive_count = int(inside_band.size)
        elif outside_candidates.size > 0:
            drive_count = int(min(min_count, outside_candidates.size))
        else:
            drive_count = 0
        return int(np.count_nonzero(support_mask)), drive_count

    def _native_strict_boundary_driver_has_material(self) -> bool:
        _, drive_count = self._native_strict_support_stats()
        return drive_count > 0

    def _native_strict_wiring_warning(self) -> str:
        if not (self.is_native_backend and self.is_native_strict):
            return ''
        if bool(ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER):
            return ''
        native_drive_nodes = [int(i) for i in self.native_drive_window_indices if 0 <= int(i) < self.node_count]
        tail_push_nodes = [int(i) for i in self._strict_hand_push_indices() if 0 <= int(i) < self.node_count]
        if len(native_drive_nodes) == 0 or len(tail_push_nodes) == 0:
            return ''
        native_drive_set = set(native_drive_nodes)
        tail_push_set = set(tail_push_nodes)
        if native_drive_set == tail_push_set or native_drive_set.issubset(tail_push_set):
            return (
                'native drive window overlaps tail push nodes: '
                f'nativeDriveWindowNodes={native_drive_nodes} tailPushNodes={tail_push_nodes}'
            )
        if max(native_drive_nodes) <= max(tail_push_nodes):
            return (
                'native drive window collapsed onto tail-side indices: '
                f'nativeDriveWindowNodes={native_drive_nodes} tailPushNodes={tail_push_nodes}'
            )
        return ''

    def _sheath_blend_alpha(self, nominal_s: float) -> float:
        """
        闁惧繑纰嶇€氭瑩妫勫Ο纰卞悁濞戞挸绉撮崯鈧柛?`s=0` 濠㈣泛瀚悘娑㈡⒒閹绢喖娅為柡鈧幘鍛闁兼澘鏈Σ鎼佸捶閵娿儱寮抽柛娆欑到閹鎯冮崟顏嗩伇濞戞搩浜為悡顓熸交閸ャ劍璇為悽顖ょ畱閸炴挳鏌呴幇顓у妱闂侇偀鍋撻柛锔戒航閳?
        閺夆晜鐟﹂悧閬嶅矗椤栨瑤绨伴梺顒€鐏濋崢銈嗙▔閳ь剛鈧潧婀卞ù澶愭焽閺勫繐螡闁绘劗鎳撻崹鐗堢附閸婄喐纭堕弶鈺佹搐閸欏棝宕ｉ敐鍡橆槯闁挎稑濂旂粩瀛樼▔椤忓洨绠烽悶姘煎亜閻ｎ剟宕楅妸鈺傛暁闁革负鍔庡ú璺ㄧ棯婢跺摜鐟愰柕?        闁告瑱缂氱粩瀛樼▔椤忓嫬鍤掔紓浣哥箰閻ｎ剟宕楅妸銊ユ闁汇垺鍞荤槐婵囩鎼淬倐鍋撶仦鎯╅柛妤佹礃椤斿矂姊归崹顔碱唺闁活剦鍓熷Λ鍧楀箯婢跺﹤鐓傞柡浣规緲閳ь剙绉崇花?rest length闁?        """
        if nominal_s <= VIRTUAL_SHEATH_RELEASE_S_MM:
            return 1.0
        blend = max(float(VIRTUAL_SHEATH_BLEND_OUT_MM), 1e-9)
        if nominal_s >= VIRTUAL_SHEATH_RELEASE_S_MM + blend:
            return 0.0
        return float(1.0 - (nominal_s - VIRTUAL_SHEATH_RELEASE_S_MM) / blend)

    def _project_to_centerline(self, point: np.ndarray) -> Tuple[np.ndarray, float]:
        p = np.asarray(point, dtype=float).reshape(3)
        best_q = self.centerline[0, :3].copy()
        best_s = 0.0
        best_d2 = float('inf')
        for i in range(self.centerline.shape[0] - 1):
            a = self.centerline[i, :3]
            b = self.centerline[i + 1, :3]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            u = 0.0 if ab2 < 1e-12 else float(np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0))
            q = a + u * ab
            d2 = float(np.dot(p - q, p - q))
            if d2 < best_d2:
                best_q = q
                best_s = float(self.centerline_cum[i] + u * float(self.centerline_cum[i + 1] - self.centerline_cum[i]))
                best_d2 = d2
        return best_q, best_s

    def _forward_centerline_target(
        self,
        point: np.ndarray,
        min_forward_mm: float | None = None,
        forward_dir: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, float]:
        p = np.asarray(point, dtype=float).reshape(3)
        proj_point, proj_s = self._project_to_centerline(p)
        if self.centerline.shape[0] == 0:
            return proj_point, proj_s

        forward_dir_vec = _normalize(
            np.asarray(forward_dir, dtype=float).reshape(3) if forward_dir is not None else self.insertion_direction
        )
        if np.linalg.norm(forward_dir_vec) < 1.0e-12:
            forward_dir_vec = self.insertion_direction.copy()

        min_forward = max(float(min_forward_mm) if min_forward_mm is not None else 0.0, max(0.15 * self.rest_spacing_mm, 0.25))
        best_any = (proj_point.copy(), float(proj_s), float(np.dot(proj_point - p, proj_point - p)))
        best_forward: tuple[np.ndarray, float, float, float] | None = None

        for i in range(max(self.centerline.shape[0] - 1, 0)):
            a = self.centerline[i, :3]
            b = self.centerline[i + 1, :3]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            u = 0.0 if ab2 < 1.0e-12 else float(np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0))
            q = a + u * ab
            q_s = float(self.centerline_cum[i] + u * float(self.centerline_cum[i + 1] - self.centerline_cum[i]))
            delta = q - p
            d2 = float(np.dot(delta, delta))
            forward_mm = float(np.dot(delta, forward_dir_vec))
            if d2 < best_any[2]:
                best_any = (q.copy(), q_s, d2)
            if q_s + 1.0e-9 < proj_s:
                continue
            if forward_mm + 1.0e-9 < min_forward:
                continue
            if best_forward is None or d2 < best_forward[2] - 1.0e-9 or (
                abs(d2 - best_forward[2]) <= 1.0e-9 and forward_mm < best_forward[3]
            ):
                best_forward = (q.copy(), q_s, d2, forward_mm)

        if best_forward is not None:
            return best_forward[0], best_forward[1]
        target_s = float(np.clip(proj_s + min_forward, 0.0, self.path_len))
        return _interp(self.centerline[:, :3], self.centerline_cum, target_s), target_s

    def _nominal_centerline_frame(self, nominal_s: float) -> Tuple[np.ndarray, float, float]:
        """
        闁诡儸鍡楀幋濞村吋锚鐎垫煡鏁?        閻庣敻鈧稓鑹炬慨锝呯箰閹舵岸濡存担鍦Ж闁煎搫鍊婚崑锝夋儍閸曨喖骞囬柛鎰噹閻ｃ劑宕楅妸锕€顫岀憸鐗堝敾缁辨繃绋夊鍛櫃闁?O(N_centerline) 闁汇劌瀚〒鑸垫交閹寸姴浠柟鍏肩矌閸屻劑鏁?        闁兼澘鏈Σ鎼佹儎鐎涙ê澶嶅ù锝堟硶閺併倗鎷犻妷銊ノ濋柣鎰贡濞堟垿宕ュ鍕枀鐎殿啫鍥ㄦ瘣 `nominal_s` 闁革负鍔嬮懙鎴ｇ疀閸愵亜娈犲☉鎾筹攻瑜板啴宕愮粭琛″亾?
        閺夆晜鐟╅崳鐑芥儍閸曨喕鎹嶉悹鎰剁到瑜把囧及椤栨せ鍋撳鍫熜╃紒灞界仢椤ュ棝宕楀鍐亢闁炽儲绻愮槐婵囩▔瀹ュ棙笑缂侇喖澧介垾妯尖偓浣冨閸╁懘鏁嶇仦鑺ョ婵縿鍊撴繛鍥偨閵娿儲鍊冲☉鏂款槸婵剟姊归懗顖滅濞磋壈澹堥崘缁樺緞閻斿懙鏃傗偓瑙勭啲缁?        濞戞梻鍠曢崗姗€寮伴幑鎰暱闂傚嫬绉崇紞?Python 闁硅矇鍐ㄧ厬闁革絻鍔庡▓鎴︽倻椤撶姴浠€殿喒鍋撻梺搴撳亾闁?        """
        proj_s = float(np.clip(nominal_s, 0.0, self.path_len))
        q = _interp(self.centerline[:, :3], self.centerline_cum, proj_s)
        radius = float(np.interp(proj_s, self.centerline_cum, self.fast_lumen_profile_mm))
        return q, proj_s, radius

    def _centerline_tangent(self, nominal_s: float) -> np.ndarray:
        if self.centerline.shape[0] < 2 or self.path_len <= 1.0e-9 or nominal_s <= 0.0:
            return self.insertion_direction.copy()
        ds = max(0.5 * self.rest_spacing_mm, 0.5)
        s0 = float(np.clip(nominal_s - ds, 0.0, self.path_len))
        s1 = float(np.clip(nominal_s + ds, 0.0, self.path_len))
        if s1 <= s0 + 1.0e-9:
            return self.insertion_direction.copy()
        p0 = _interp(self.centerline[:, :3], self.centerline_cum, s0)
        p1 = _interp(self.centerline[:, :3], self.centerline_cum, s1)
        tan = _normalize(p1 - p0)
        return tan if np.linalg.norm(tan) > 1.0e-12 else self.insertion_direction.copy()

    def _update_estimated_push_mm(self) -> float:
        rigid = self._current_rigid_state()
        if rigid.shape[0] == 0:
            return self.estimated_push_mm

        drive_delta = rigid[self.drive_reference_index, :3] - self.drive_reference_pos0
        measured_drive = float(np.dot(drive_delta, self.insertion_direction))
        measured_drive = float(np.clip(measured_drive, 0.0, self.max_push_mm))
        self.drive_push_mm = max(self.drive_push_mm, measured_drive)

        _, tip_proj_s = self._tip_projection_to_centerline()
        measured_tip = float(np.clip(tip_proj_s - self.tip_path_s0, 0.0, self.max_push_mm))
        self.tip_progress_raw_mm = measured_tip
        self.tip_progress_mm = max(self.tip_progress_mm, measured_tip)
        if rigid.shape == self.initial_wire_centers.shape:
            self.base_progress_mm = float(np.dot(rigid[0, :3] - self.initial_wire_centers[0, :3], self.insertion_direction))
            self.mid_progress_mm = float(np.dot(
                rigid[self.mid_index, :3] - self.initial_wire_centers[self.mid_index, :3],
                self.insertion_direction,
            ))
            self.tip_axial_progress_mm = float(np.dot(
                rigid[-1, :3] - self.initial_wire_centers[-1, :3],
                self.insertion_direction,
            ))

        if self.is_native_backend:
            if self.is_native_strict:
                measured_push_mm = float(max(
                    self.drive_push_mm,
                    self.base_progress_mm,
                    self.tip_progress_raw_mm,
                    0.0,
                ))
                lead_cap_mm = float(max(0.25, 0.5 * self.rest_spacing_mm))
                self.estimated_push_mm = float(min(self.commanded_push_mm, measured_push_mm + lead_cap_mm))
                return self.estimated_push_mm
            self.estimated_push_mm = float(self.commanded_push_mm)
        elif self.use_kinematic_beam_insertion:
            self.estimated_push_mm = max(self.commanded_push_mm, self.drive_push_mm)
        else:
            self.estimated_push_mm = max(self.estimated_push_mm, self.drive_push_mm, self.tip_progress_mm)
        return self.estimated_push_mm

    def _native_strict_barrier_contact_gate(self) -> float:
        if not self.is_native_strict:
            return 0.0
        wall_gap_mm = float(self._native_strict_actual_wall_gap_mm())
        physical_gap_mm = float(self._native_strict_physical_contact_clearance_mm())
        min_clearance_mm = float(self._native_strict_min_lumen_clearance_mm())
        barrier_nodes = self._native_strict_barrier_active_node_count()
        gates = [float(self._native_strict_hard_wall_contact())]
        if np.isfinite(wall_gap_mm):
            gates.append(float(np.clip((0.24 - wall_gap_mm) / 0.18, 0.0, 1.0)))
        if np.isfinite(physical_gap_mm):
            gates.append(float(np.clip((0.24 - physical_gap_mm) / 0.18, 0.0, 1.0)))
        if barrier_nodes > 0 and np.isfinite(min_clearance_mm):
            gates.append(float(np.clip((0.06 - min_clearance_mm) / 0.12, 0.0, 1.0)))
        return float(max(gates))

    def _native_strict_hard_wall_contact(self) -> bool:
        if not self.is_native_strict:
            return bool(self.wall_contact_active)
        hard_gap_mm = max(float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM) - 0.04, 0.24)
        wall_gap_mm = float(self._native_strict_actual_wall_gap_mm())
        physical_gap_mm = float(self._native_strict_physical_contact_clearance_mm())
        clearance_mm = float(self.wall_contact_clearance_mm)
        min_clearance_mm = float(self._native_strict_min_lumen_clearance_mm())
        barrier_nodes = self._native_strict_barrier_active_node_count()
        if np.isfinite(clearance_mm) and clearance_mm <= hard_gap_mm:
            return True
        if np.isfinite(wall_gap_mm) and wall_gap_mm <= hard_gap_mm:
            return True
        if np.isfinite(physical_gap_mm) and physical_gap_mm <= hard_gap_mm:
            return True
        return bool(
            barrier_nodes > int(ELASTICROD_STRICT_GUI_GUIDED_CONTACT_MAX_BARRIER_NODES)
            and np.isfinite(min_clearance_mm)
            and min_clearance_mm <= max(0.16, hard_gap_mm - 0.08)
        )

    def _native_strict_guided_wall_follow_contact(self) -> bool:
        if (not self.is_native_strict) or (not self.wall_contact_active):
            return False
        barrier_nodes = self._native_strict_barrier_active_node_count()
        if barrier_nodes > int(ELASTICROD_STRICT_GUI_GUIDED_CONTACT_MAX_BARRIER_NODES):
            return False

        clearance_mm = float(self.wall_contact_clearance_mm)
        wall_gap_mm = float(self._native_strict_actual_wall_gap_mm())
        physical_gap_mm = float(self._native_strict_physical_contact_clearance_mm())
        head_stretch = float(self._native_strict_max_head_stretch())
        head_stretch_soft_limit, _ = self._native_strict_head_stretch_limits()
        support_stretch = self._native_debug_array(self._native_debug_stretch)
        max_stretch = float(np.max(np.abs(support_stretch))) if support_stretch.size else 0.0

        guided_clearance_floor_mm = max(
            float(ELASTICROD_STRICT_GUI_LIGHT_CONTACT_CLEARANCE_MM),
            float(ELASTICROD_STRICT_GUI_GUIDED_CONTACT_CLEARANCE_MM) + 0.10,
            float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM) + 0.02,
        )
        guided_gap_floor_mm = max(
            float(ELASTICROD_STRICT_GUI_LIGHT_CONTACT_WALL_GAP_MM),
            float(ELASTICROD_STRICT_GUI_GUIDED_CONTACT_WALL_GAP_MM) + 0.16,
            float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM) - 0.02,
        )

        if np.isfinite(clearance_mm) and clearance_mm < guided_clearance_floor_mm:
            return False
        if np.isfinite(wall_gap_mm) and wall_gap_mm < guided_gap_floor_mm:
            return False
        if np.isfinite(physical_gap_mm) and physical_gap_mm < guided_gap_floor_mm:
            return False
        if head_stretch > max(1.5 * head_stretch_soft_limit, 0.015):
            return False
        if max_stretch > 0.02:
            return False
        return True

    def _advance_commanded_push(self, dt: float) -> None:
        if dt <= 0.0:
            return
        if self.is_native_safe and ELASTICROD_ENABLE_SAFE_RECOVERY and self._native_safe_recovery_cooldown > 0:
            self._native_safe_recovery_cooldown = max(self._native_safe_recovery_cooldown - 1, 0)
        if self.is_native_strict and self.native_strict_boundary_driver_enabled:
            if not self._native_strict_boundary_driver_has_material():
                if not self._native_strict_support_exhausted:
                    support_count, drive_count = self._native_strict_support_stats()
                    print(
                        f'[INFO] [elasticrod-strict] insertion paused at step={self.step_count} '
                        f'because no nodes remain in the external drive corridor: '
                        f'supportCorridorOccupancy={support_count} driveWindowOccupancy={drive_count} '
                        f'commandedPush={self.commanded_push_mm:.3f} mm'
                    )
                    self._native_strict_support_exhausted = True
                return
            self._native_strict_support_exhausted = False
        if self.is_native_backend and self._native_thrust_limit_blocks_advance():
            return
        if self.is_native_strict:
            scale = float(
                np.clip(
                    self._native_startup_ramp_scale()
                    * float(np.clip(self.push_force_scale, 0.0, self._max_push_scale_allowed())),
                    0.0,
                    self._max_push_scale_allowed(),
                )
            )
            scale *= self._native_strict_driver_follow_scale()
            scale *= self._native_strict_guided_feed_boost()
            scale = float(min(scale, self._max_push_scale_allowed()))
            progress_anchor_mm = max(
                float(self.tip_progress_raw_mm),
                float(self.drive_push_mm) - 0.30,
                0.0,
            )
            backlog_mm = max(float(self.commanded_push_mm) - progress_anchor_mm, 0.0)
            if self.wall_contact_active:
                stalled_tip_speed = max(float(self.filtered_tip_forward_speed_mm_s), 0.0)
                stall_gate = float(np.clip((0.80 - stalled_tip_speed) / 0.80, 0.0, 1.0))
                backlog_gate = float(np.clip((backlog_mm - 0.35) / 0.90, 0.0, 1.0))
                head_stretch = float(self._native_strict_max_head_stretch())
                soft_limit, hard_limit = self._native_strict_head_stretch_limits()
                stretch_gate = 0.0
                if hard_limit > soft_limit:
                    stretch_gate = float(np.clip((head_stretch - soft_limit) / (hard_limit - soft_limit), 0.0, 1.0))
                settle_gate = max(stretch_gate, stall_gate * backlog_gate)
                if settle_gate >= 0.995:
                    return
                scale *= float((1.0 - settle_gate) + settle_gate * 0.05)
        else:
            scale = float(np.clip(self.push_force_scale, 0.0, self._max_push_scale_allowed()))
        if self.is_native_backend and (not self.is_native_strict):
            scale *= self._native_startup_ramp_scale()
        if self.is_native_safe:
            progress_anchor_mm = max(
                float(self.tip_progress_raw_mm),
                float(self.drive_push_mm) - 0.50,
                0.0,
            )
            backlog_mm = max(float(self.commanded_push_mm) - progress_anchor_mm, 0.0)
            if backlog_mm > 1.20:
                backlog_gate = float(np.clip((backlog_mm - 1.20) / 1.40, 0.0, 1.0))
                backlog_scale_floor = 0.65
                if self.wall_contact_active:
                    backlog_scale_floor = 0.50
                if self._native_safe_recovery_cooldown > 0:
                    backlog_scale_floor = min(backlog_scale_floor, 0.45)
                scale *= float((1.0 - backlog_gate) + backlog_gate * backlog_scale_floor)
            max_head_stretch = max(
                float(self._native_debug_scalar(self._native_debug_max_head_stretch, default=0.0)),
                0.0,
            )
            stretch_profile = self._native_debug_array(self._native_debug_stretch)
            max_stretch = float(np.max(np.abs(stretch_profile))) if stretch_profile.size else 0.0
            barrier_nodes = max(self._native_strict_barrier_active_node_count(), 0)
            safe_kink_gate = 0.0
            if self.wall_contact_active or barrier_nodes > 0:
                if np.isfinite(self.wall_contact_clearance_mm):
                    safe_kink_gate = max(
                        safe_kink_gate,
                        float(np.clip((0.18 - float(self.wall_contact_clearance_mm)) / 0.12, 0.0, 1.0)),
                    )
                safe_kink_gate = max(
                    safe_kink_gate,
                    float(np.clip((max_head_stretch - 6.0e-3) / 1.2e-2, 0.0, 1.0)),
                    float(np.clip((max_stretch - 2.5e-2) / 7.5e-2, 0.0, 1.0)),
                    float(np.clip((barrier_nodes - 3.0) / 4.0, 0.0, 1.0)),
                )
            if safe_kink_gate > 0.0:
                safe_kink_floor = 0.42
                scale *= float((1.0 - safe_kink_gate) + safe_kink_gate * safe_kink_floor)
        command_dt = float(dt)
        if self.use_native_gui_wallclock_control:
            command_dt = min(
                command_dt,
                max(self._native_gui_wallclock_insertion_dt_limit(), float(self._native_runtime_dt_s), 0.0),
            )
        push_speed_mm_s = float(self.push_force_target_speed_mm_s)
        if self.use_native_gui_wallclock_control:
            push_speed_mm_s *= float(max(ELASTICROD_GUI_WALLCLOCK_PUSH_SPEED_SCALE, 1.0))
        delta_push_mm = push_speed_mm_s * scale * command_dt
        if self.use_native_gui_wallclock_control:
            delta_push_mm = min(delta_push_mm, float(max(ELASTICROD_STRICT_GUI_MAX_INSERTION_STEP_MM, 0.0)))
        self.commanded_push_mm = float(np.clip(self.commanded_push_mm + delta_push_mm, 0.0, self.max_push_mm))

    def _native_strict_guided_feed_boost(self) -> float:
        if not (self.is_native_strict and self.use_native_gui_wallclock_control):
            return 1.0

        progress_mm = max(
            float(self.commanded_push_mm),
            float(self.drive_push_mm),
            float(self.tip_progress_raw_mm),
            0.0,
        )
        if progress_mm < float(ELASTICROD_STRICT_FEED_BOOST_START_MM):
            return 1.0

        head_stretch = float(self._native_strict_max_head_stretch())
        if head_stretch > float(ELASTICROD_STRICT_FEED_BOOST_HEAD_STRETCH_LIMIT):
            return 1.0

        support_stretch = self._native_debug_array(self._native_debug_stretch)
        max_stretch = float(np.max(np.abs(support_stretch))) if support_stretch.size else 0.0
        if max_stretch > 0.02:
            return 1.0

        bend_severity = float(np.clip(self._native_strict_bend_severity(), 0.0, 1.0))
        barrier_nodes = int(self._native_strict_barrier_active_node_count())
        boost = 1.0

        if self.wall_contact_active:
            physical_gap_mm = float(self._native_strict_physical_contact_clearance_mm())
            gap_gate = (
                float(np.clip((physical_gap_mm + 0.02) / 0.24, 0.0, 1.0))
                if np.isfinite(physical_gap_mm)
                else 1.0
            )
            barrier_limit = max(int(ELASTICROD_STRICT_GUI_GUIDED_CONTACT_MAX_BARRIER_NODES), 1)
            barrier_gate = float(np.clip((barrier_limit + 1 - barrier_nodes) / (barrier_limit + 1), 0.0, 1.0))
            if gap_gate > 0.0:
                boost = max(
                    boost,
                    1.0 + (float(ELASTICROD_STRICT_FEED_BOOST_CONTACT) - 1.0) * gap_gate * max(barrier_gate, 0.35),
                )

        bend_gate = max(
            float(np.clip((bend_severity - 0.30) / 0.45, 0.0, 1.0)),
            float(np.clip(barrier_nodes / 3.0, 0.0, 1.0)),
        )
        if bend_gate > 0.0:
            boost = max(boost, 1.0 + (float(ELASTICROD_STRICT_FEED_BOOST_BEND) - 1.0) * bend_gate)

        return float(np.clip(boost, 1.0, self._max_push_scale_allowed()))

    def _max_push_scale_allowed(self) -> float:
        if self.is_native_realtime:
            return float(
                max(
                    ELASTICROD_REALTIME_SPEED_SCALE_FREE,
                    ELASTICROD_REALTIME_SPEED_SCALE_TRANSITION,
                    ELASTICROD_REALTIME_SPEED_SCALE_CONTACT,
                )
            )
        return 1.0

    def _native_startup_ramp_scale(self) -> float:
        if not self.is_native_backend:
            return 1.0
        ramp_time = max(float(ELASTICROD_ACTIVE_STARTUP_RAMP_TIME_S), 0.0)
        if self.use_native_gui_wallclock_control:
            ramp_time = max(
                float(ELASTICROD_GUI_WALLCLOCK_STARTUP_RAMP_S),
                0.0,
            )
        if ramp_time <= 1e-9:
            return 1.0
        alpha = float(np.clip(self.sim_time_s / ramp_time, 0.0, 1.0))
        return float(0.5 - 0.5 * np.cos(np.pi * alpha))

    def _native_strict_magnetic_release_scale(self) -> float:
        if not self.is_native_strict:
            return 1.0
        straight_mm = max(float(ELASTICROD_STRICT_INITIAL_STRAIGHT_PUSH_MM), 0.0)
        release_mm = max(float(ELASTICROD_STRICT_MAGNETIC_RELEASE_SPAN_MM), 0.0)
        # Do not hard-disable the native field before the straight entry window
        # is cleared. A zero here resets the native magnetic ramp/filter state,
        # so when the head finally reaches the first bend the field has to cold
        # start from nearly zero again and the tip looks "stuck". Keep a small
        # preload alive instead; before release the strict field direction still
        # follows the entry axis, so this warms the controller without pulling
        # the head sideways ahead of the bend.
        preload_scale = 0.18
        lead_cap_mm = max(2.0 * float(self.rest_spacing_mm), 1.0)
        # Let the field follow real material feed, but do not allow the proximal
        # tail-drive to release steering far ahead of the distal head. That early
        # release was creating entry-region loops before the tip had actually
        # reached the first turn.
        progress_mm = max(
            float(self.tip_progress_raw_mm),
            min(float(self.drive_push_mm), float(self.tip_progress_raw_mm) + lead_cap_mm),
            min(float(self.base_progress_mm), float(self.tip_progress_raw_mm) + lead_cap_mm),
            0.0,
        )
        base_scale = 1.0
        if progress_mm <= straight_mm:
            base_scale = preload_scale
        elif release_mm <= 1.0e-9:
            base_scale = 1.0
        else:
            u = float(np.clip((progress_mm - straight_mm) / release_mm, 0.0, 1.0))
            release_alpha = float(u * u * (3.0 - 2.0 * u))
            base_scale = float(preload_scale + (1.0 - preload_scale) * release_alpha)
        return float(np.clip(base_scale, 0.0, 1.0))

    def _native_strict_field_damping_scale(self) -> float:
        if not self.is_native_strict:
            return 1.0
        head_stretch = float(self._native_strict_max_head_stretch())
        soft_limit, hard_limit = self._native_strict_head_stretch_limits()
        stretch_gate = 0.0
        engage_limit = max(0.55 * soft_limit, 0.010)
        if hard_limit > engage_limit:
            stretch_gate = float(np.clip((head_stretch - engage_limit) / (hard_limit - engage_limit), 0.0, 1.0))

        contact_gate = 0.0
        wall_gap_mm = float(self._native_strict_actual_wall_gap_mm())
        if self.wall_contact_active and np.isfinite(wall_gap_mm):
            contact_gate = float(np.clip((0.32 - wall_gap_mm) / 0.18, 0.0, 1.0))

        fresh_contact_gate = 0.0
        if (
            self.wall_contact_active
            and self._strict_wall_contact_enter_step is not None
            and self.step_count >= self._strict_wall_contact_enter_step
        ):
            settle_steps = 12.0
            fresh_contact_gate = float(
                np.clip(
                    1.0 - (self.step_count - self._strict_wall_contact_enter_step) / settle_steps,
                    0.0,
                    1.0,
                )
            )

        stalled_contact_gate = 0.0
        if self.wall_contact_active:
            stalled_tip_speed = max(float(self.filtered_tip_forward_speed_mm_s), 0.0)
            progress_anchor_mm = max(float(self.tip_progress_mm), float(self.drive_push_mm) - 0.30, 0.0)
            backlog_mm = max(float(self.commanded_push_mm) - progress_anchor_mm, 0.0)
            stalled_contact_gate = float(
                np.clip((0.60 - stalled_tip_speed) / 0.60, 0.0, 1.0)
                * np.clip((backlog_mm - 0.40) / 0.80, 0.0, 1.0)
            )

        damping_gate = max(stretch_gate, 0.85 * contact_gate, 0.95 * fresh_contact_gate, stalled_contact_gate)
        damping_floor = 0.05 if stalled_contact_gate > 0.0 else 0.12
        return float((1.0 - damping_gate) + damping_gate * damping_floor)

    def _native_control_dt(self, solver_dt: float) -> float:
        solver_dt = max(float(solver_dt), 0.0)
        if not self.use_native_gui_wallclock_control:
            self._native_control_time_mode = 'solver'
            self._native_control_dt_s = solver_dt
            return solver_dt

        now_s = time.perf_counter()
        if self._native_gui_wallclock_last_s is None:
            control_dt = solver_dt
        else:
            wall_dt = max(now_s - self._native_gui_wallclock_last_s, 0.0)
            max_dt = max(float(ELASTICROD_GUI_WALLCLOCK_DT_MAX_S), solver_dt)
            control_dt = min(max(wall_dt, solver_dt), max_dt)
        self._native_gui_wallclock_last_s = now_s
        self._native_control_time_mode = 'wallclock'
        self._native_control_dt_s = float(control_dt)
        return float(control_dt)

    def _native_gui_wallclock_insertion_dt_limit(self) -> float:
        limit_s = max(float(ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_MAX_S), 0.0)
        runtime_dt_s = max(float(self._native_runtime_dt_s), 0.0)
        if runtime_dt_s <= 0.0:
            return limit_s
        band = str(self._native_runtime_band or 'free')
        always_push_strict = self.is_native_strict and bool(ELASTICROD_STRICT_ALWAYS_PUSH_FORWARD)
        recent_contact_release_steps = (
            int(self.step_count - self._strict_wall_contact_release_step)
            if (
                self.is_native_strict
                and self._strict_wall_contact_release_step is not None
                and self.step_count >= self._strict_wall_contact_release_step
            )
            else 10**9
        )
        recent_contact_release = bool(
            self.is_native_strict
            and recent_contact_release_steps <= int(max(ELASTICROD_STRICT_RUNTIME_RELEASE_TRANSITION_HOLD_STEPS, 0))
        )
        if always_push_strict and band != 'contact' and (not self.wall_contact_active):
            # Keep strict "always push" aggressive through free/transition bands
            # so the head still feeds into the bend. Right after wall release,
            # however, the distal section is still ringing; keep pushing, but
            # let the guarded dt path below cap the feed step until the head is
            # quiet again.
            head_stretch_now = float(self._native_strict_max_head_stretch())
            support_stretch_now = self._native_debug_array(self._native_debug_stretch)
            max_stretch_now = float(np.max(np.abs(support_stretch_now))) if support_stretch_now.size else 0.0
            if (
                (band == 'free')
                and (not recent_contact_release)
                and head_stretch_now <= max(float(ELASTICROD_STRICT_FEED_BOOST_HEAD_STRETCH_LIMIT), 0.012)
                and max_stretch_now <= 0.020
            ):
                return limit_s

        if band == 'contact':
            band_scale = float(max(ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_CONTACT_SCALE, 1.0))
        elif band == 'transition':
            band_scale = float(max(ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_TRANSITION_SCALE, 1.0))
        else:
            band_scale = float(max(ELASTICROD_GUI_WALLCLOCK_INSERTION_DT_FREE_SCALE, 1.0))

        base_limit_s = float(min(limit_s, band_scale * runtime_dt_s))
        if (not self.is_native_strict) or band == 'free':
            return base_limit_s

        barrier_active = self._native_strict_barrier_active()
        barrier_nodes = self._native_strict_barrier_active_node_count()
        barrier_contact_gate = self._native_strict_barrier_contact_gate()
        head_stretch = self._native_strict_max_head_stretch()
        head_stretch_soft_limit, _ = self._native_strict_head_stretch_limits()
        bend_severity = self._native_strict_bend_severity()
        upcoming_turn_deg = self._native_strict_upcoming_turn_deg()
        turn_gate = float(np.clip((upcoming_turn_deg - 10.0) / 20.0, 0.0, 1.0))
        contact_clearance_mm = float(self.wall_contact_clearance_mm)
        if not np.isfinite(contact_clearance_mm):
            contact_clearance_mm = float(self._native_strict_actual_wall_gap_mm())
        wall_gap_mm = float(self._native_strict_actual_wall_gap_mm())
        clearance_gate = (
            float(np.clip((contact_clearance_mm - 0.16) / 0.20, 0.0, 1.0))
            if np.isfinite(contact_clearance_mm)
            else 1.0
        )
        stretch_ref = max(1.35 * head_stretch_soft_limit, 0.03)
        head_safe_gate = float(np.clip((stretch_ref - head_stretch) / max(stretch_ref, 1.0e-9), 0.0, 1.0))
        barrier_gate = 0.0
        penetration_gate = 0.0
        if barrier_nodes > 0 and barrier_contact_gate > 0.0:
            barrier_gate = max(barrier_gate, float(np.clip((float(barrier_nodes) - 1.0) / 3.0, 0.0, 1.0)))
        if np.isfinite(wall_gap_mm):
            if wall_gap_mm <= 0.0:
                penetration_gate = float(np.clip((-wall_gap_mm) / 0.18, 0.0, 1.0))
                barrier_gate = max(
                    barrier_gate,
                    float(np.clip((0.05 - wall_gap_mm) / 0.23, 0.35, 1.0)),
                )
        if barrier_active and barrier_nodes <= 0 and barrier_contact_gate > 0.0 and np.isfinite(contact_clearance_mm):
            barrier_gate = max(barrier_gate, float(np.clip((0.12 - contact_clearance_mm) / 0.12, 0.0, 0.6)))
        base_relaxed_gate = head_safe_gate * (1.0 - barrier_gate)
        relaxed_gate = base_relaxed_gate

        if band == 'transition':
            relaxed_gate *= max(0.35, bend_severity, turn_gate)
            relaxed_limit_s = float(
                min(limit_s, max(base_limit_s, float(ELASTICROD_STRICT_GUI_LIGHT_CONTACT_INSERTION_DT_S)))
            )
        else:
            contact_follow_gate = max(0.45, bend_severity, turn_gate)
            if (
                np.isfinite(wall_gap_mm)
                and wall_gap_mm > 0.0
                and barrier_nodes <= int(ELASTICROD_STRICT_GUI_GUIDED_CONTACT_MAX_BARRIER_NODES)
            ):
                wall_follow_gap_gate = float(np.clip((wall_gap_mm - 0.06) / 0.20, 0.0, 1.0))
                wall_follow_clearance_gate = (
                    float(np.clip((contact_clearance_mm - 0.11) / 0.22, 0.0, 1.0))
                    if np.isfinite(contact_clearance_mm)
                    else 1.0
                )
                wall_follow_gate = max(
                    wall_follow_gap_gate,
                    wall_follow_clearance_gate,
                    0.55 if self.wall_contact_active else 0.35,
                )
                relaxed_gate = base_relaxed_gate * contact_follow_gate * wall_follow_gate * (1.0 - 0.55 * penetration_gate)
            else:
                relaxed_gate = base_relaxed_gate * contact_follow_gate * max(clearance_gate, 0.25) * (1.0 - 0.55 * penetration_gate)
            relaxed_limit_s = float(
                min(limit_s, max(base_limit_s, float(ELASTICROD_STRICT_GUI_GUIDED_CONTACT_INSERTION_DT_S)))
            )

        result_limit_s = float((1.0 - relaxed_gate) * base_limit_s + relaxed_gate * relaxed_limit_s)
        if always_push_strict and band == 'contact':
            contact_metric_candidates = []
            if np.isfinite(contact_clearance_mm):
                contact_metric_candidates.append(float(contact_clearance_mm))
            if np.isfinite(wall_gap_mm):
                contact_metric_candidates.append(float(wall_gap_mm))
            contact_metric_mm = min(contact_metric_candidates) if contact_metric_candidates else float('inf')
            guarded_contact_dt_s = float(max(1.45 * runtime_dt_s, runtime_dt_s))
            if np.isfinite(contact_metric_mm):
                # Keep advancing in strict always-push mode, but progressively
                # shorten the GUI insertion command step toward a guarded
                # contact-band dt as the tip approaches or slightly crosses the
                # wall. The post-solve strict guard now catches tiny residual
                # wall violations, so we no longer need to collapse all the way
                # back to the raw solver dt, which was making the wire look
                # visually stalled in the first bend.
                deep_contact_gate = float(np.clip((0.34 - contact_metric_mm) / 0.28, 0.0, 1.0))
                result_limit_s = float(
                    (1.0 - deep_contact_gate) * result_limit_s
                    + deep_contact_gate * guarded_contact_dt_s
                )
            if self.wall_contact_active and self._strict_wall_contact_enter_step is not None:
                contact_duration_steps = max(int(self.step_count - self._strict_wall_contact_enter_step + 1), 0)
                prolonged_contact_gate = float(np.clip((contact_duration_steps - 18.0) / 18.0, 0.0, 1.0))
                if prolonged_contact_gate > 0.0:
                    result_limit_s = float(
                        (1.0 - prolonged_contact_gate) * result_limit_s
                        + prolonged_contact_gate * guarded_contact_dt_s
                    )

        if recent_contact_release:
            release_cap_s = float(max(ELASTICROD_STRICT_GUI_GUIDED_CONTACT_INSERTION_DT_S, 0.0))
            stretch_gate = float(np.clip((head_stretch - 0.010) / 0.020, 0.0, 1.0))
            stretch_cap_s = float(
                (1.0 - stretch_gate) * release_cap_s
                + stretch_gate * max(float(ELASTICROD_REALTIME_DT_CONTACT_S), 0.0025)
            )
            if stretch_cap_s > 0.0:
                result_limit_s = float(min(result_limit_s, stretch_cap_s))

        return result_limit_s

    def _update_tip_speed(self, tip_pos: np.ndarray, dt: float) -> None:
        if dt <= 1e-12:
            return
        _, tip_proj_s = self._tip_projection_to_centerline()
        raw_speed = float((tip_proj_s - self.prev_tip_proj_s_mm) / dt)
        self.prev_tip_proj_s_mm = tip_proj_s
        self.tip_forward_speed_mm_s = raw_speed
        alpha = float(np.clip(TIP_SPEED_FILTER_ALPHA, 0.0, 1.0))
        self.filtered_tip_forward_speed_mm_s = (1.0 - alpha) * self.filtered_tip_forward_speed_mm_s + alpha * raw_speed

    def _tip_probe_indices(self) -> list[int]:
        count = WALL_CONTACT_TIP_PROBE_NODES
        if self.is_native_strict:
            count = max(count, self.magnetic_head_edge_count + 2)
        count = min(count, len(self.distal_indices))
        return self.distal_indices[-count:] if count > 0 else [self.node_count - 1]

    def _virtual_sheath_point(self, idx: int, point: np.ndarray | None = None) -> np.ndarray:
        """
        闁哄唲鍛暭 beam 鐟滅増娲栭惈鍡涙焻閺勫繒甯嗛梺鎻掔焿缁辨繈妫勫Ο纰卞悁缂佹拝闄勫顐︽倷闁稓鐟濋柡鍕靛灙閳ь剚绮岄崹鍨叏鐎ｎ偅娈婚柡宥囨嚀椤曡鲸绋夊┑鍡氬煂闁绘鍩栭弳锝嗘媴閹炬娊鎸紒澶屽劦閳ь剚绻愮槐?        闁兼澘鏈Σ鍛婄▔閵夛妇澹愰柟绋款槴閳ь剚绮岄崣鍡涘矗閿濆洤浠?+ 閻犲洢鍎存俊顓㈡倷閻熺増鍊冲☉鏂款槸婵剟姊?* 闁规亽鍔忕换妯绘姜鐎涙ɑ鐓欓柛姘灍閳ь剚绻冨鐢稿绩閸撗呮瀭闁?
        閺夆晜鐟﹂悧杈╂偖椤愶妇浜濈紒鐙呯磿鐎规娊寮堕悢鐑樼暠閺夆晜鍨归顒勬嚍閸屾粌浠┑顔碱儑缁捇宕楁潏鈺佹疇闁挎稑鑻崵顓㈠矗閿濆拋妲卞☉鎾崇С缁变即宕堕悩杈闁告帗绻傞～鎰板即閼碱剙娈犳繛鍫濐儑閺嗏偓闁兼澘鑻懜浼村箣閹邦厼顫戦悷娆愬竾閳?        """
        nominal_s = self._node_s(idx)
        return self.entry_point + nominal_s * self.insertion_direction

    def _use_pre_entry_access_guide(self, nominal_s: float) -> bool:
        """
        闁告瑯浜滈顔炬偘閳ь剛绮婚垾鍐插汲闁告瑱绲介ˇ濠氭儍閸曨喚绠紒鏃戝灣閻擃厼鈻撻棃娑欏剻闁活潿鍔夐埀顒佺矎椤旀牠姊婚鑳巢鐎点倕艌閳ь剚绻愮槐?        闁烩晩鍠氬▓鎴炵▔瀹ュ棙笑缂備綀鍛暰闁革负鍔忛、鍛不閳ュ啿鏁堕柡鈧娑欏閻庣敻妫跨粭锝夋晬瀹€鍐ｅ亾鐏炵偓笑濞ｅ洦绻嗛惁澶嬫媴閹炬剚妯嗛悗鐢告？缁楋綁宕ｉ鍥у幋濞寸姴娴峰﹢锛勨偓鍦仜閸欏棝宕ｉ敐鍫㈢闁稿繈鍎荤槐?        濞戞挸绉崇槐鐗堢鎼淬値鏀ㄧ紒鐘偓鎰佹▎濠㈤€涙閺呭爼妫冮姀銈嗗磳閺夆晜绋戦獮鎾诲Υ?        """
        if self.is_native_backend or self.enable_virtual_sheath:
            return False
        guide_len = max(float(BEAM_PRE_ENTRY_ACCESS_GUIDE_MM), 0.0)
        return (-guide_len <= float(nominal_s) < 0.0)

    def _pre_entry_access_point(self, nominal_s: float) -> np.ndarray:
        return self.entry_point + float(nominal_s) * self.insertion_direction

    def _strict_native_surface_guard_eligible(self, point: np.ndarray | None, nominal_s: float) -> bool:
        if nominal_s >= VIRTUAL_SHEATH_RELEASE_S_MM:
            return True
        if (not self.is_native_strict) or point is None or self.centerline.shape[0] == 0:
            return False

        p = np.asarray(point, dtype=float).reshape(3)
        support_length_mm = max(float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM), 0.0)
        axial_mm = float(np.dot(p - self.entry_point, self.insertion_direction))
        if axial_mm < -(support_length_mm + 1.0e-9):
            return False

        q, proj_s = self._project_to_centerline(p)
        if not np.isfinite(proj_s):
            return False
        proj_s = float(np.clip(proj_s, 0.0, self.path_len))
        if self.use_fast_lumen and self.fast_lumen_profile_mm.shape[0] == self.centerline.shape[0]:
            radius_mm = float(np.interp(proj_s, self.centerline_cum, self.fast_lumen_profile_mm))
        else:
            radius_mm = float(self.entry_radius_mm)
        radial_mm = float(np.linalg.norm(p - q))
        guard_margin_mm = max(0.50, float(self._contact_radius_mm() + ELASTICROD_STRICT_BARRIER_ACTIVATION_MARGIN_MM))
        if radial_mm > radius_mm + guard_margin_mm:
            return False
        if axial_mm >= 0.0:
            return True

        proximal_guard_mm = support_length_mm + max(0.5 * self.rest_spacing_mm, 0.5)
        return proj_s <= proximal_guard_mm + 1.0e-9

    def _point_wall_clearance(self, point: np.ndarray, nominal_s: float, exact_projection: bool = False) -> float:
        if self._use_pre_entry_access_guide(nominal_s):
            axis_point = self._pre_entry_access_point(nominal_s)
            return -float(np.linalg.norm(np.asarray(point, dtype=float).reshape(3) - axis_point))
        strict_surface_guard = self._strict_native_surface_guard_eligible(point, nominal_s)
        if nominal_s < VIRTUAL_SHEATH_RELEASE_S_MM and not strict_surface_guard:
            return float('inf')
        if self.is_native_strict and strict_surface_guard and nominal_s < VIRTUAL_SHEATH_RELEASE_S_MM:
            support_clearance = self._strict_external_support_clearance_mm(point)
            if np.isfinite(support_clearance):
                return float(support_clearance)
        profile_clearance = float('inf')
        if (nominal_s >= 0.0 or strict_surface_guard) and self.use_fast_lumen:
            if exact_projection or nominal_s < VIRTUAL_SHEATH_RELEASE_S_MM:
                q, proj_s = self._project_to_centerline(point)
                radius = float(np.interp(proj_s, self.centerline_cum, self.fast_lumen_profile_mm))
            else:
                q, proj_s, radius = self._nominal_centerline_frame(nominal_s)
            radial = float(np.linalg.norm(np.asarray(point, dtype=float).reshape(3) - q))
            profile_clearance = max(radius - LUMEN_CLEARANCE_MM, 0.0) - radial
        if self.is_native_safe and nominal_s >= VIRTUAL_SHEATH_RELEASE_S_MM:
            exact_surface = self._point_surface_clearance_sample(
                point,
                nominal_s,
                exact_projection=True,
            )
            if exact_surface is not None and np.isfinite(float(exact_surface[0])):
                if np.isfinite(profile_clearance):
                    return float(min(profile_clearance, float(exact_surface[0])))
                return float(exact_surface[0])
        return float(profile_clearance)

    def _invalidate_surface_probe_cache(self) -> None:
        self._surface_probe_cache_step = -1
        self._surface_probe_cache = []
        self._surface_edge_probe_cache_step = -1
        self._surface_edge_probe_cache = []
        self._surface_min_clearance_cache_step = -1
        self._head_surface_clearance_cache_step = -1
        self._head_wall_clearance_cache_step = -1

    def _surface_monitor_low_budget(self) -> bool:
        return bool((not self.is_native_backend) or (self.is_native_strict and self.use_native_gui_wallclock_control))

    def _strict_surface_probe_indices(self, points_mm: np.ndarray | None = None) -> list[int]:
        if self.node_count <= 0:
            return []

        if points_mm is not None and points_mm.ndim == 2 and points_mm.shape[0] >= self.node_count:
            released = [
                i
                for i in range(self.node_count)
                if self._strict_native_surface_guard_eligible(points_mm[i, :3], self._node_s(i))
            ]
        else:
            released = [i for i in range(self.node_count) if self._node_s(i) >= VIRTUAL_SHEATH_RELEASE_S_MM]
        if not released:
            return []
        if (
            points_mm is not None
            and points_mm.ndim == 2
            and points_mm.shape[0] >= self.node_count
            and self._strict_surface_fullscan_required()
        ):
            return released

        low_budget = self._surface_monitor_low_budget()
        contact_distance_mm = float(ELASTICROD_CONTACT_DISTANCE_MM) if self.is_native_strict else float(CONTACT_DISTANCE_MM)
        near_contact = (
            self.wall_contact_active
            or self._native_strict_barrier_active_node_count() > 0
            or self.surface_wall_contact_clearance_mm <= max(1.5, 6.0 * contact_distance_mm)
        )
        if (not self.is_native_backend) and low_budget:
            dense_tip_count = max(WALL_CONTACT_TIP_PROBE_NODES + self.magnetic_head_edge_count, 8)
            light_tip_count = max(WALL_CONTACT_TIP_PROBE_NODES + 2, 5)
        elif low_budget:
            dense_tip_count = max(2 * WALL_CONTACT_TIP_PROBE_NODES + self.magnetic_head_edge_count, 8)
            light_tip_count = max(self.magnetic_head_edge_count + 1, 5)
        else:
            dense_tip_count = max(4 * WALL_CONTACT_TIP_PROBE_NODES, 20)
            light_tip_count = max(WALL_CONTACT_TIP_PROBE_NODES + self.magnetic_head_edge_count + 2, 8)
        tip_probe_count = min(dense_tip_count if near_contact else light_tip_count, len(released))
        tip_dense = released[-tip_probe_count:]
        sampled = set(tip_dense)
        body = released[:-tip_probe_count]
        if not body:
            return sorted(sampled)

        # Tail nodes are always checked densely; the longer body is sampled
        # sparsely, then densified again wherever the cheap lumen estimate says
        # we are already close to a wall.
        if (not self.is_native_backend) and low_budget:
            coarse_stride = 8 if near_contact else 18
        else:
            coarse_stride = (4 if near_contact else 12) if low_budget else (1 if near_contact else 6)
        phase = self.step_count % coarse_stride
        sampled.update(body[phase::coarse_stride])
        sampled.add(body[0])
        sampled.add(body[-1])
        sampled.add(body[len(body) // 2])

        if points_mm is not None and points_mm.ndim == 2 and points_mm.shape[0] >= self.node_count:
            near_wall_margin_mm = max(
                0.75,
                3.0 * contact_distance_mm,
                2.0 * float(ELASTICROD_STRICT_BARRIER_ACTIVATION_MARGIN_MM),
            )
            clearance_probe_indices = body if near_contact else body[phase::coarse_stride]
            for idx in clearance_probe_indices:
                clearance = self._point_wall_clearance(points_mm[idx, :3], self._node_s(idx), exact_projection=False)
                if clearance > near_wall_margin_mm:
                    continue
                sampled.add(idx)
                if idx > 0 and self._strict_native_surface_guard_eligible(points_mm[idx - 1, :3], self._node_s(idx - 1)):
                    sampled.add(idx - 1)
                if idx + 1 < self.node_count and self._strict_native_surface_guard_eligible(points_mm[idx + 1, :3], self._node_s(idx + 1)):
                    sampled.add(idx + 1)

        return sorted(sampled)

    def _strict_surface_probe_edge_indices(self, points_mm: np.ndarray | None = None) -> list[int]:
        if self.node_count <= 1:
            return []

        if points_mm is not None and points_mm.ndim == 2 and points_mm.shape[0] >= self.node_count:
            released = []
            for i in range(self.node_count - 1):
                midpoint = 0.5 * (points_mm[i, :3] + points_mm[i + 1, :3])
                nominal_s = 0.5 * (self._node_s(i) + self._node_s(i + 1))
                if self._strict_native_surface_guard_eligible(midpoint, nominal_s):
                    released.append(i)
        else:
            released = [
                i
                for i in range(self.node_count - 1)
                if min(self._node_s(i), self._node_s(i + 1)) >= VIRTUAL_SHEATH_RELEASE_S_MM
            ]
        if not released:
            return []
        if (
            points_mm is not None
            and points_mm.ndim == 2
            and points_mm.shape[0] >= self.node_count
            and self._strict_surface_fullscan_required()
        ):
            return released

        low_budget = self._surface_monitor_low_budget()
        contact_distance_mm = float(ELASTICROD_CONTACT_DISTANCE_MM) if self.is_native_strict else float(CONTACT_DISTANCE_MM)
        near_contact = (
            self._native_strict_barrier_active_node_count() > 0
            or self.surface_wall_contact_clearance_mm <= max(0.75, 3.0 * contact_distance_mm)
        )
        if (not self.is_native_backend) and low_budget:
            dense_tip_count = max(WALL_CONTACT_TIP_PROBE_NODES + self.magnetic_head_edge_count, 8)
            light_tip_count = max(WALL_CONTACT_TIP_PROBE_NODES + 1, 5)
        elif low_budget:
            dense_tip_count = max(2 * WALL_CONTACT_TIP_PROBE_NODES + self.magnetic_head_edge_count, 8)
            light_tip_count = max(self.magnetic_head_edge_count, 5)
        else:
            dense_tip_count = max(4 * WALL_CONTACT_TIP_PROBE_NODES, 20)
            light_tip_count = max(WALL_CONTACT_TIP_PROBE_NODES + self.magnetic_head_edge_count + 1, 8)
        tip_probe_count = min(dense_tip_count if near_contact else light_tip_count, len(released))
        tip_dense = released[-tip_probe_count:]
        sampled = set(tip_dense)
        body = released[:-tip_probe_count]
        if not body:
            return sorted(sampled)

        if (not self.is_native_backend) and low_budget:
            coarse_stride = 10 if near_contact else 20
        else:
            coarse_stride = (5 if near_contact else 14) if low_budget else (2 if near_contact else 8)
        phase = (self.step_count + 1) % coarse_stride
        sampled.update(body[phase::coarse_stride])
        sampled.add(body[0])
        sampled.add(body[-1])
        sampled.add(body[len(body) // 2])

        if points_mm is not None and points_mm.ndim == 2 and points_mm.shape[0] >= self.node_count:
            near_wall_margin_mm = max(
                0.75,
                3.0 * contact_distance_mm,
                2.0 * float(ELASTICROD_STRICT_BARRIER_ACTIVATION_MARGIN_MM),
            )
            clearance_probe_edges = body if near_contact else body[phase::coarse_stride]
            for edge_idx in clearance_probe_edges:
                midpoint = 0.5 * (points_mm[edge_idx, :3] + points_mm[edge_idx + 1, :3])
                nominal_s = 0.5 * (self._node_s(edge_idx) + self._node_s(edge_idx + 1))
                clearance = self._point_wall_clearance(midpoint, nominal_s, exact_projection=False)
                if clearance > near_wall_margin_mm:
                    continue
                sampled.add(edge_idx)
                if edge_idx > 0:
                    prev_midpoint = 0.5 * (points_mm[edge_idx - 1, :3] + points_mm[edge_idx, :3])
                    prev_s = 0.5 * (self._node_s(edge_idx - 1) + self._node_s(edge_idx))
                    if self._strict_native_surface_guard_eligible(prev_midpoint, prev_s):
                        sampled.add(edge_idx - 1)
                if edge_idx + 1 < self.node_count - 1:
                    next_midpoint = 0.5 * (points_mm[edge_idx + 1, :3] + points_mm[edge_idx + 2, :3])
                    next_s = 0.5 * (self._node_s(edge_idx + 1) + self._node_s(edge_idx + 2))
                    if self._strict_native_surface_guard_eligible(next_midpoint, next_s):
                        sampled.add(edge_idx + 1)

        return sorted(sampled)

    def _surface_query_closest_point(self, point: np.ndarray) -> tuple[float, np.ndarray] | None:
        if self.vessel_surface_query is None:
            return None
        p = np.asarray(point, dtype=float).reshape(3)
        distance, closest, _ = self.vessel_surface_query.query(p)
        if not np.isfinite(distance):
            return None
        return float(distance), np.asarray(closest, dtype=float).reshape(3)

    def _point_surface_clearance_sample(
        self,
        point: np.ndarray,
        nominal_s: float,
        surface_query: tuple[float, np.ndarray] | None = None,
        exact_projection: bool = False,
    ) -> tuple[float, np.ndarray, np.ndarray] | None:
        strict_surface_guard = self._strict_native_surface_guard_eligible(point, nominal_s)
        if self.vessel_surface_query is None or (nominal_s < VIRTUAL_SHEATH_RELEASE_S_MM and not strict_surface_guard):
            return None

        p = np.asarray(point, dtype=float).reshape(3)
        query_result = surface_query if surface_query is not None else self._surface_query_closest_point(p)
        if query_result is None:
            return None
        _, closest = query_result

        if exact_projection or not self.use_fast_lumen or nominal_s < VIRTUAL_SHEATH_RELEASE_S_MM:
            q, proj_s = self._project_to_centerline(p)
        else:
            q, proj_s, _ = self._nominal_centerline_frame(nominal_s)

        inward = np.asarray(q, dtype=float).reshape(3) - np.asarray(closest, dtype=float).reshape(3)
        inward_norm = float(np.linalg.norm(inward))
        if inward_norm <= 1.0e-9:
            inward = p - np.asarray(closest, dtype=float).reshape(3)
            inward_norm = float(np.linalg.norm(inward))
        if inward_norm <= 1.0e-9:
            inward = self._centerline_tangent(proj_s)
            inward_norm = float(np.linalg.norm(inward))
        inward = inward / max(inward_norm, 1.0e-12)

        signed_depth = float(np.dot(p - np.asarray(closest, dtype=float).reshape(3), inward))
        clearance = signed_depth - self._contact_radius_mm()
        return float(clearance), np.asarray(closest, dtype=float).reshape(3), inward

    def _surface_probe_requires_exact_projection(
        self,
        clearance_mm: float,
        *,
        near_contact: bool,
        tip_region: bool,
    ) -> bool:
        if self._strict_surface_fullscan_required():
            return True
        if self._strict_gui_skip_exact_surface_monitor():
            return False
        if not self.use_fast_lumen:
            return True
        if not np.isfinite(clearance_mm):
            return False
        contact_distance_mm = float(ELASTICROD_CONTACT_DISTANCE_MM) if self.is_native_strict else float(CONTACT_DISTANCE_MM)
        activation_margin = max(float(ELASTICROD_STRICT_BARRIER_ACTIVATION_MARGIN_MM), 0.25)
        if self.is_native_strict and self.use_native_gui_wallclock_control:
            exact_margin = max(0.30, 1.25 * contact_distance_mm, 0.85 * activation_margin)
            tip_exact_margin = max(0.50, 1.75 * contact_distance_mm, 1.40 * activation_margin)
        else:
            exact_margin = max(0.50, 2.0 * contact_distance_mm, 1.5 * activation_margin)
            tip_exact_margin = max(0.90, 3.0 * contact_distance_mm, 2.5 * activation_margin)
        if tip_region and (near_contact or clearance_mm <= tip_exact_margin):
            return True
        if clearance_mm <= exact_margin:
            return True
        if near_contact and clearance_mm <= tip_exact_margin:
            return True
        return False

    def _strict_surface_monitor_near_contact(self) -> bool:
        contact_distance_mm = float(ELASTICROD_CONTACT_DISTANCE_MM) if self.is_native_strict else float(CONTACT_DISTANCE_MM)
        threshold_mm = max(
            0.75,
            3.0 * contact_distance_mm,
            float(ELASTICROD_STRICT_BARRIER_ACTIVATION_MARGIN_MM) + 0.10,
        )
        if self.wall_contact_active:
            return True
        if self._native_strict_barrier_active_node_count() > 0:
            return True
        if np.isfinite(self.surface_wall_contact_clearance_mm) and self.surface_wall_contact_clearance_mm <= threshold_mm:
            return True
        native_clearance = self._native_debug_scalar(self._native_debug_min_lumen_clearance_mm)
        if np.isfinite(native_clearance) and native_clearance <= threshold_mm:
            return True
        return False

    def _strict_surface_fullscan_required(self) -> bool:
        if not self.is_native_strict:
            return False
        barrier_nodes = self._native_strict_barrier_active_node_count()
        if barrier_nodes >= max(int(ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT) - 1, 4):
            return True
        native_clearance = self._native_debug_scalar(self._native_debug_min_lumen_clearance_mm, default=float('inf'))
        if np.isfinite(native_clearance) and native_clearance <= 0.30:
            return True
        return self.wall_contact_active

    def _strict_gui_skip_exact_surface_monitor(self) -> bool:
        if not (self.is_native_strict and self.use_native_gui_wallclock_control):
            return False
        if self._strict_surface_fullscan_required():
            return False
        barrier_nodes = self._native_strict_barrier_active_node_count()
        native_clearance = self._native_debug_scalar(self._native_debug_min_lumen_clearance_mm, default=float('inf'))
        tip_offset = self._tip_centerline_offset_mm()
        far_clearance_mm = float(ELASTICROD_STRICT_GUI_FAR_CLEARANCE_MM)
        far_offset_mm = float(ELASTICROD_STRICT_GUI_FAR_OFFSET_MM)
        offset_requires_exact = bool(
            tip_offset >= far_offset_mm
            and (
                self.wall_contact_active
                or barrier_nodes > 0
                or (
                    np.isfinite(native_clearance)
                    and native_clearance <= max(far_clearance_mm + 0.25, 1.25)
                )
            )
        )
        near_contact = bool(
            self.wall_contact_active
            or barrier_nodes > 0
            or (np.isfinite(native_clearance) and native_clearance <= far_clearance_mm)
            or offset_requires_exact
        )
        if near_contact:
            hold_steps = max(int(ELASTICROD_STRICT_GUI_EXACT_SURFACE_RECHECK_STEPS), 1)
            self._strict_gui_force_exact_surface_until_step = max(
                self._strict_gui_force_exact_surface_until_step,
                self.step_count + hold_steps,
            )
            return False
        return self.step_count > self._strict_gui_force_exact_surface_until_step

    def _cheap_surface_clearance_mm(self, *, tip_only: bool = False) -> float:
        points_mm = self._current_points_mm()
        clearances: list[float] = []
        if points_mm.ndim != 2 or points_mm.shape[0] == 0:
            return float('inf')

        if tip_only:
            probe_indices = [
                int(i) for i in self._tip_probe_indices()
                if 0 <= int(i) < self.node_count
            ]
        else:
            probe_indices = self._strict_surface_probe_indices(points_mm)

        for idx in probe_indices:
            nominal_s = self._node_s(idx)
            if not self._strict_native_surface_guard_eligible(points_mm[idx, :3], nominal_s):
                continue
            if nominal_s < VIRTUAL_SHEATH_RELEASE_S_MM:
                q, proj_s = self._project_to_centerline(points_mm[idx, :3])
                radius = float(np.interp(proj_s, self.centerline_cum, self.fast_lumen_profile_mm))
            else:
                q, _, radius = self._nominal_centerline_frame(nominal_s)
            clearances.append(float(radius - np.linalg.norm(points_mm[idx, :3] - q) - self._contact_radius_mm()))

        if tip_only:
            tip_edge_start = max(self.node_count - WALL_CONTACT_TIP_PROBE_NODES - 1, 0)
            if self.is_native_strict:
                tip_edge_start = max(self.node_count - (self.magnetic_head_edge_count + 2), 0)
            edge_indices = list(range(tip_edge_start, max(self.node_count - 1, 0)))
        else:
            edge_indices = self._strict_surface_probe_edge_indices(points_mm)

        for edge_idx in edge_indices:
            p0 = points_mm[edge_idx, :3]
            p1 = points_mm[edge_idx + 1, :3]
            point = 0.5 * (p0 + p1)
            nominal_s = 0.5 * (self._node_s(edge_idx) + self._node_s(edge_idx + 1))
            if not self._strict_native_surface_guard_eligible(point, nominal_s):
                continue
            if nominal_s < VIRTUAL_SHEATH_RELEASE_S_MM:
                q, proj_s = self._project_to_centerline(point)
                radius = float(np.interp(proj_s, self.centerline_cum, self.fast_lumen_profile_mm))
            else:
                q, _, radius = self._nominal_centerline_frame(nominal_s)
            clearances.append(float(radius - np.linalg.norm(point - q) - self._contact_radius_mm()))

        return float(min(clearances)) if clearances else float('inf')

    def _surface_probe_samples(self) -> list[tuple[int, float, np.ndarray, np.ndarray]]:
        if self._strict_gui_skip_exact_surface_monitor():
            self._surface_probe_cache_step = self.step_count
            self._surface_probe_cache = []
            return self._surface_probe_cache
        near_contact = self._strict_surface_monitor_near_contact()
        if not self.is_native_backend:
            refresh_stride = (
                max(int(BEAM_SURFACE_REFRESH_NEAR_STEPS), 1)
                if (near_contact or self.step_count <= 5)
                else max(int(BEAM_SURFACE_REFRESH_FAR_STEPS), 1)
            )
        elif self.is_native_strict and self.use_native_gui_wallclock_control:
            refresh_stride = (
                max(int(ELASTICROD_STRICT_GUI_SURFACE_REFRESH_NEAR_STEPS), 1)
                if (near_contact or self.step_count <= 5)
                else max(int(ELASTICROD_STRICT_GUI_SURFACE_REFRESH_FAR_STEPS), 1)
            )
        else:
            refresh_stride = 1 if (near_contact or self.step_count <= 5) else 6
        if self._surface_probe_cache_step >= 0 and (self.step_count - self._surface_probe_cache_step) < refresh_stride:
            return self._surface_probe_cache

        points_mm = self._current_points_mm()
        samples: list[tuple[int, float, np.ndarray, np.ndarray]] = []
        if points_mm.ndim == 2 and points_mm.shape[0] > 0:
            tip_probe = set(self._tip_probe_indices())
            for idx in self._strict_surface_probe_indices(points_mm):
                nominal_s = self._node_s(idx)
                query_result = self._surface_query_closest_point(points_mm[idx, :3])
                if query_result is None:
                    continue
                sample = self._point_surface_clearance_sample(
                    points_mm[idx, :3],
                    nominal_s,
                    surface_query=query_result,
                    exact_projection=False,
                )
                if sample is None:
                    continue
                if self._surface_probe_requires_exact_projection(
                    float(sample[0]),
                    near_contact=near_contact,
                    tip_region=idx in tip_probe,
                ):
                    exact_sample = self._point_surface_clearance_sample(
                        points_mm[idx, :3],
                        nominal_s,
                        surface_query=query_result,
                        exact_projection=True,
                    )
                    if exact_sample is not None:
                        sample = exact_sample
                clearance, closest, inward = sample
                samples.append((idx, float(clearance), closest, inward))

        self._surface_probe_cache_step = self.step_count
        self._surface_probe_cache = samples
        return samples

    def _surface_edge_sample_alphas(self, edge_idx: int, near_contact: bool) -> tuple[float, ...]:
        tip_edge_start = max(self.node_count - WALL_CONTACT_TIP_PROBE_NODES - 2, 0)
        if self.is_native_strict:
            tip_edge_start = max(self.node_count - (self.magnetic_head_edge_count + 3), 0)
        if near_contact and edge_idx >= tip_edge_start:
            if self.is_native_strict and self.use_native_gui_wallclock_control:
                # Near the first bend, two interior samples can miss a soft
                # magnetic head grazing the wall between nodes. Probe the full
                # head edge more densely so strict contact stays continuous.
                return (0.10, 0.25, 1.0 / 3.0, 0.50, 2.0 / 3.0, 0.75, 0.90)
            return (0.10, 0.25, 0.50, 0.75, 0.90)
        return (0.5,)

    def _surface_edge_probe_samples(self) -> list[tuple[int, float, np.ndarray, float, np.ndarray, np.ndarray]]:
        if self._strict_gui_skip_exact_surface_monitor():
            self._surface_edge_probe_cache_step = self.step_count
            self._surface_edge_probe_cache = []
            return self._surface_edge_probe_cache
        near_contact = self._strict_surface_monitor_near_contact()
        if not self.is_native_backend:
            refresh_stride = (
                max(int(BEAM_SURFACE_REFRESH_NEAR_STEPS), 1)
                if (near_contact or self.step_count <= 5)
                else max(int(BEAM_SURFACE_REFRESH_FAR_STEPS), 1)
            )
        elif self.is_native_strict and self.use_native_gui_wallclock_control:
            refresh_stride = (
                1
                if near_contact
                else (
                    max(int(ELASTICROD_STRICT_GUI_SURFACE_REFRESH_NEAR_STEPS), 1)
                    if self.step_count <= 5
                    else max(int(ELASTICROD_STRICT_GUI_SURFACE_REFRESH_FAR_STEPS), 1)
                )
            )
        else:
            refresh_stride = 1 if (near_contact or self.step_count <= 5) else 6
        if self._surface_edge_probe_cache_step >= 0 and (self.step_count - self._surface_edge_probe_cache_step) < refresh_stride:
            return self._surface_edge_probe_cache

        points_mm = self._current_points_mm()
        samples: list[tuple[int, float, np.ndarray, float, np.ndarray, np.ndarray]] = []
        if points_mm.ndim == 2 and points_mm.shape[0] >= 2:
            tip_edge_start = max(self.node_count - WALL_CONTACT_TIP_PROBE_NODES - 1, 0)
            if self.is_native_strict:
                tip_edge_start = max(self.node_count - (self.magnetic_head_edge_count + 2), 0)
            for edge_idx in self._strict_surface_probe_edge_indices(points_mm):
                p0 = points_mm[edge_idx, :3]
                p1 = points_mm[edge_idx + 1, :3]
                for alpha in self._surface_edge_sample_alphas(edge_idx, near_contact):
                    point = (1.0 - alpha) * p0 + alpha * p1
                    nominal_s = (1.0 - alpha) * self._node_s(edge_idx) + alpha * self._node_s(edge_idx + 1)
                    query_result = self._surface_query_closest_point(point)
                    if query_result is None:
                        continue
                    sample = self._point_surface_clearance_sample(
                        point,
                        nominal_s,
                        surface_query=query_result,
                        exact_projection=False,
                    )
                    if sample is None:
                        continue
                    if self._surface_probe_requires_exact_projection(
                        float(sample[0]),
                        near_contact=near_contact,
                        tip_region=edge_idx >= tip_edge_start,
                    ):
                        exact_sample = self._point_surface_clearance_sample(
                            point,
                            nominal_s,
                            surface_query=query_result,
                            exact_projection=True,
                        )
                        if exact_sample is not None:
                            sample = exact_sample
                    clearance, closest, inward = sample
                    samples.append((edge_idx, float(alpha), point, clearance, closest, inward))

        self._surface_edge_probe_cache_step = self.step_count
        self._surface_edge_probe_cache = samples
        return samples

    def _surface_min_clearance_mm(self) -> float:
        if self._surface_min_clearance_cache_step == self.step_count:
            return float(self._surface_min_clearance_cache)
        if self._strict_gui_skip_exact_surface_monitor():
            tip_only = bool(
                self.is_native_strict
                and self.use_native_gui_wallclock_control
                and (not self.native_strict_barrier_enabled)
            )
            if tip_only:
                head_surface_clearance = self._head_surface_clearance()
                if np.isfinite(head_surface_clearance):
                    clearance = float(head_surface_clearance)
                else:
                    clearance = float(self._head_wall_clearance())
            else:
                clearance = self._cheap_surface_clearance_mm(tip_only=tip_only)
            self.surface_wall_contact_clearance_mm = clearance
            self._surface_min_clearance_cache_step = self.step_count
            self._surface_min_clearance_cache = float(clearance)
            return clearance
        samples = self._surface_probe_samples()
        edge_samples = self._surface_edge_probe_samples()
        clearances = [sample[1] for sample in samples]
        clearances.extend(sample[3] for sample in edge_samples)
        clearance = float(min(clearances)) if clearances else float('inf')
        self.surface_wall_contact_clearance_mm = clearance
        self._surface_min_clearance_cache_step = self.step_count
        self._surface_min_clearance_cache = float(clearance)
        return clearance

    def _head_surface_clearance(self) -> float:
        if self._head_surface_clearance_cache_step == self.step_count:
            cached_clearance = float(self._head_surface_clearance_cache)
            if self.is_native_strict and (not np.isfinite(cached_clearance)):
                head_profile_clearance = float(self._head_wall_clearance())
                near_contact_band_mm = max(float(ELASTICROD_STRICT_TIP_WALL_CONTACT_ENTER_MM) + 0.06, 0.24)
                if np.isfinite(head_profile_clearance) and head_profile_clearance <= near_contact_band_mm:
                    pass
                else:
                    return cached_clearance
            else:
                return cached_clearance
        samples = self._surface_probe_samples()
        edge_samples = self._surface_edge_probe_samples()
        def _strict_exact_tip_surface_clearance_fallback() -> float:
            if (not self.is_native_strict) or self.vessel_surface_query is None:
                return float('inf')
            head_profile_clearance = float(self._head_wall_clearance())
            near_contact_band_mm = max(float(ELASTICROD_STRICT_TIP_WALL_CONTACT_ENTER_MM) + 0.06, 0.24)
            if (
                (not self.wall_contact_active)
                and np.isfinite(head_profile_clearance)
                and head_profile_clearance > near_contact_band_mm
            ):
                return float('inf')
            recent_trusted_clearance = float(self.wall_contact_clearance_mm)
            if (
                (not self.wall_contact_active)
                and np.isfinite(recent_trusted_clearance)
                and recent_trusted_clearance >= max(float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM) + 0.02, 0.50)
                and np.isfinite(head_profile_clearance)
                and head_profile_clearance >= -0.08
            ):
                return recent_trusted_clearance
            points_mm = self._current_points_mm()
            if points_mm.ndim != 2 or points_mm.shape[0] == 0:
                return float('inf')
            clearances: list[float] = []
            for idx in self._tip_probe_indices():
                if not (0 <= idx < points_mm.shape[0]):
                    continue
                nominal_s = self._node_s(idx)
                sample = self._point_surface_clearance_sample(
                    points_mm[idx, :3],
                    nominal_s,
                    exact_projection=True,
                )
                if sample is not None and np.isfinite(float(sample[0])):
                    clearances.append(float(sample[0]))
            tip_edge_start = max(self.node_count - WALL_CONTACT_TIP_PROBE_NODES - 1, 0)
            if self.is_native_strict:
                tip_edge_start = max(self.node_count - (self.magnetic_head_edge_count + 2), 0)
            for edge_idx in range(tip_edge_start, max(points_mm.shape[0] - 1, 0)):
                if edge_idx < 0 or edge_idx + 1 >= points_mm.shape[0]:
                    continue
                p0 = points_mm[edge_idx, :3]
                p1 = points_mm[edge_idx + 1, :3]
                for alpha in self._surface_edge_sample_alphas(edge_idx, True):
                    point = (1.0 - alpha) * p0 + alpha * p1
                    nominal_s = (1.0 - alpha) * self._node_s(edge_idx) + alpha * self._node_s(edge_idx + 1)
                    sample = self._point_surface_clearance_sample(
                        point,
                        nominal_s,
                        exact_projection=True,
                    )
                    if sample is not None and np.isfinite(float(sample[0])):
                        clearances.append(float(sample[0]))
            return float(min(clearances)) if clearances else float('inf')

        if not samples and not edge_samples:
            clearance = _strict_exact_tip_surface_clearance_fallback()
            self._head_surface_clearance_cache_step = self.step_count
            self._head_surface_clearance_cache = clearance
            return clearance
        tip_probe = set(self._tip_probe_indices())
        tip_clearances = [sample[1] for sample in samples if sample[0] in tip_probe]
        tip_edge_start = max(self.node_count - WALL_CONTACT_TIP_PROBE_NODES - 1, 0)
        if self.is_native_strict:
            tip_edge_start = max(self.node_count - (self.magnetic_head_edge_count + 2), 0)
        tip_clearances.extend(sample[3] for sample in edge_samples if sample[0] >= tip_edge_start)
        if tip_clearances:
            clearance = float(min(tip_clearances))
            self._head_surface_clearance_cache_step = self.step_count
            self._head_surface_clearance_cache = clearance
            return clearance
        fallback_clearance = _strict_exact_tip_surface_clearance_fallback()
        if np.isfinite(fallback_clearance):
            self._head_surface_clearance_cache_step = self.step_count
            self._head_surface_clearance_cache = float(fallback_clearance)
            return float(fallback_clearance)
        all_clearances = [sample[1] for sample in samples]
        all_clearances.extend(sample[3] for sample in edge_samples)
        clearance = float(min(all_clearances)) if all_clearances else float('inf')
        self._head_surface_clearance_cache_step = self.step_count
        self._head_surface_clearance_cache = clearance
        return clearance

    def _native_strict_head_surface_contact_is_trustworthy(
        self,
        head_surface_clearance_mm: float,
        *,
        native_gap_mm: float,
        head_profile_clearance_mm: float,
    ) -> bool:
        if not np.isfinite(head_surface_clearance_mm):
            return False
        if not (self.is_native_strict and self.use_native_gui_wallclock_control):
            return True

        barrier_nodes = self._native_strict_barrier_active_node_count()
        native_clearance_mm = self._native_debug_scalar(
            self._native_debug_min_lumen_clearance_mm,
            default=float('inf'),
        )
        contact_distance_mm = float(ELASTICROD_CONTACT_DISTANCE_MM)
        trust_gap_mm = max(
            float(ELASTICROD_STRICT_GUI_GUIDED_PRECONTACT_WALL_GAP_MM),
            float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM),
            0.30,
        )
        trust_clearance_mm = max(
            float(ELASTICROD_STRICT_GUI_GUIDED_PRECONTACT_CLEARANCE_MM),
            float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM),
            0.30,
        )
        agreement_margin_mm = max(0.35, 3.0 * contact_distance_mm)

        # If the native lumen barrier is inactive and both native clearance
        # signals are comfortably positive, a lone negative exact-surface sample
        # in GUI wall-clock mode is almost always the transient mesh-projection
        # glitch that was causing false wall contact toggles.
        if (
            barrier_nodes <= 0
            and np.isfinite(native_gap_mm)
            and native_gap_mm >= trust_gap_mm
            and (
                (not np.isfinite(native_clearance_mm))
                or native_clearance_mm >= max(trust_clearance_mm - float(ELASTICROD_STRICT_BARRIER_SAFETY_MARGIN_MM), 0.0)
            )
        ):
            if head_surface_clearance_mm < -0.05:
                return False
            if (
                np.isfinite(head_profile_clearance_mm)
                and head_profile_clearance_mm >= trust_clearance_mm
                and head_surface_clearance_mm < 0.5 * trust_clearance_mm
            ):
                return False

        if np.isfinite(head_profile_clearance_mm):
            if (
                head_surface_clearance_mm < -0.05
                and head_profile_clearance_mm >= trust_clearance_mm
                and (
                    (not np.isfinite(native_gap_mm))
                    or native_gap_mm >= trust_gap_mm
                )
            ):
                return False
            if abs(head_surface_clearance_mm - head_profile_clearance_mm) <= agreement_margin_mm:
                return True

        if np.isfinite(native_gap_mm):
            if abs(head_surface_clearance_mm - native_gap_mm) <= agreement_margin_mm:
                return True
            if head_surface_clearance_mm >= native_gap_mm - agreement_margin_mm:
                return True

        if barrier_nodes > 0:
            return True
        if np.isfinite(native_clearance_mm) and native_clearance_mm <= trust_clearance_mm:
            return True
        return head_surface_clearance_mm >= 0.0

    def _native_strict_false_profile_contact_clearance_mm(
        self,
        *,
        head_profile_clearance_mm: float,
        head_surface_clearance_mm: float,
    ) -> float | None:
        if not self.is_native_strict:
            return None
        if not (
            np.isfinite(head_profile_clearance_mm)
            and np.isfinite(head_surface_clearance_mm)
        ):
            return None
        if head_profile_clearance_mm > max(float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM), 0.10):
            return None
        if head_surface_clearance_mm < max(float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM) + 0.02, 0.32):
            return None
        disagreement_margin_mm = max(0.75, 2.0 * float(self._contact_radius_mm()))
        if head_surface_clearance_mm < head_profile_clearance_mm + disagreement_margin_mm:
            return None
        if not self._native_strict_head_surface_contact_is_trustworthy(
            float(head_surface_clearance_mm),
            native_gap_mm=float(head_profile_clearance_mm),
            head_profile_clearance_mm=float(head_profile_clearance_mm),
        ):
            return None
        barrier_nodes = self._native_strict_barrier_active_node_count()
        if barrier_nodes > int(ELASTICROD_STRICT_GUI_GUIDED_CONTACT_MAX_BARRIER_NODES):
            return None
        head_stretch = float(self._native_strict_max_head_stretch())
        soft_limit, _ = self._native_strict_head_stretch_limits()
        if head_stretch > max(1.35 * soft_limit, 0.020):
            return None
        return float(head_surface_clearance_mm)

    def _native_strict_physical_contact_clearance_mm(self) -> float:
        if not self.is_native_strict:
            return float('inf')
        clearances: list[float] = []
        head_profile_clearance = self._head_wall_clearance()
        head_surface_clearance = self._head_surface_clearance()
        false_profile_override = self._native_strict_false_profile_contact_clearance_mm(
            head_profile_clearance_mm=float(head_profile_clearance),
            head_surface_clearance_mm=float(head_surface_clearance),
        )
        if false_profile_override is not None:
            return float(false_profile_override)
        native_gap = self._native_strict_actual_wall_gap_mm()
        if np.isfinite(native_gap):
            clearances.append(float(native_gap))
        surface_trustworthy = self._native_strict_head_surface_contact_is_trustworthy(
            float(head_surface_clearance),
            native_gap_mm=float(native_gap),
            head_profile_clearance_mm=float(head_profile_clearance),
        )
        contact_distance_mm = float(ELASTICROD_CONTACT_DISTANCE_MM)
        disagreement_margin_mm = max(0.35, 3.0 * contact_distance_mm)
        profile_is_false_contact = bool(
            surface_trustworthy
            and np.isfinite(head_surface_clearance)
            and np.isfinite(head_profile_clearance)
            and head_surface_clearance >= max(0.30, head_profile_clearance + disagreement_margin_mm)
            and ((not np.isfinite(native_gap)) or native_gap >= max(0.30, contact_distance_mm))
        )
        if (
            (not profile_is_false_contact)
            and surface_trustworthy
            and np.isfinite(head_surface_clearance)
            and np.isfinite(head_profile_clearance)
            and head_profile_clearance < head_surface_clearance - max(0.18, 0.75 * contact_distance_mm)
            and head_surface_clearance >= max(0.18, 0.95 * contact_distance_mm)
            and ((not np.isfinite(native_gap)) or native_gap >= max(float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM), 0.30))
        ):
            # In tight bends the centerline-derived radius profile can be more
            # conservative than the real STL lumen. Do not let that surrogate
            # clearance create a hard contact if the exact surface query and
            # native wall gap both say the head is still safely inside.
            profile_is_false_contact = True
        if np.isfinite(head_profile_clearance) and not profile_is_false_contact:
            clearances.append(float(head_profile_clearance))
        if surface_trustworthy:
            clearances.append(float(head_surface_clearance))
        return float(min(clearances)) if clearances else float('inf')

    def _project_inside_surface(self, point: np.ndarray, nominal_s: float, exact_projection: bool = False) -> np.ndarray:
        sample = self._point_surface_clearance_sample(point, nominal_s, exact_projection=exact_projection)
        if sample is None:
            return np.asarray(point, dtype=float).reshape(3)
        _, closest, inward = sample
        target_depth = float(self._contact_radius_mm() + max(self.native_strict_lumen_clamp_tolerance_mm, 0.05))
        return np.asarray(closest, dtype=float).reshape(3) + target_depth * np.asarray(inward, dtype=float).reshape(3)

    def _should_log_native_strict_guard(
        self,
        *,
        corrected_nodes: int = 0,
        corrected_edge_samples: int = 0,
        clipped_nodes: int = 0,
        min_clearance_mm: float = float('inf'),
    ) -> bool:
        if self._native_last_strict_guard_step == self.step_count:
            return False
        if self.step_count <= 5:
            return True
        if corrected_nodes > 0 or corrected_edge_samples > 0 or clipped_nodes > 0:
            return True
        near_wall_band_mm = max(float(ELASTICROD_STRICT_BARRIER_ACTIVATION_MARGIN_MM), 0.25)
        if np.isfinite(min_clearance_mm) and min_clearance_mm <= near_wall_band_mm:
            interval = 100 if self.use_native_gui_wallclock_control else 20
            return (self.step_count % interval) == 0
        return DEBUG_PRINT_EVERY > 0 and (self.step_count % DEBUG_PRINT_EVERY) == 0

    def _strict_surface_exact_monitor_required(self, native_clearance_mm: float | None = None) -> bool:
        if not self.is_native_strict:
            return True
        clearance = (
            float(native_clearance_mm)
            if native_clearance_mm is not None and np.isfinite(native_clearance_mm)
            else self._native_debug_scalar(self._native_debug_min_lumen_clearance_mm, default=float('inf'))
        )
        activation_margin = max(float(ELASTICROD_STRICT_BARRIER_ACTIVATION_MARGIN_MM), 0.25)
        exact_band_mm = max(0.25, 0.35 * activation_margin)
        if not self.native_strict_barrier_enabled:
            if not self.use_native_gui_wallclock_control:
                return True
            if not np.isfinite(clearance):
                return bool(self.wall_contact_active)
            return bool(self.wall_contact_active or clearance <= exact_band_mm)
        if not np.isfinite(clearance):
            return True
        return bool(
            self.wall_contact_active
            or self._native_strict_barrier_active_node_count() > 0
            or clearance <= exact_band_mm
        )

    def _pre_entry_surface_clearance(self, point: np.ndarray) -> float:
        """
        閺夆晜鐟╅崳鐑藉矗椤忓嫪绮甸柛蹇嬪劚瑜版稒寰勯弽顐ｇ暠閺夆晜鍨靛┃鈧悗鐟扮墕瀹曡偐鎲撮敃鈧ぐ鍌炲礆閵堝棗绁﹂柨娑樺缁楀骞嶉幐搴☆€撶紒顔煎⒔閳?signed-distance 闁汇劌瀚禍瀵告嫻閿濆啠鍋?
        闁烩晩鍠楅悥锝咁嚗閸繂寰斿ù锝嗘惈缁?        - SOFA 闁规亽鍎磋闂佺偓宕樼粈瀣嫻閿濆洦鍩傛慨婵撶悼濞堟垿骞掗妷銊愭洖效閸屾粳鎺楁晬?        - 閺夆晜鐟╅崳閿嬶紣濠靛棭妯嗛柣顏勵儌閳ь剚绮岄崣鍡涘矗閿濆顎嶉弶鈺傚灦閻撳洭寮堕垾鑼Ъ濠㈣埖鐗炵粩鐔封枔閸偅笑闁告熬绠戦崙锛勭磼韫囨挸娈ゅ☉鏂挎唉閸掓盯宕氶幏宀婃敤缂佺姭鈧剚妯嗗閫涜閳ь剚绻愮槐?        - 濞戞挴鍋撻柡鍐跨畳閸掓稑顕ュΔ鈧妵濠冩交閹搭垳绀夐悘蹇氶哺婵℃悂鏌囬敐鍕伇閻忓繐绻戦宀勫箯婢跺﹥绀€闁活亞鍠庨悿鍕礂閵夈劎鐔呴弶鐐差嚟閸ゅ酣鏁嶅畝鍏鎳為崒婊冧化缂佺嫏鍕垫⒕闁哄被鍎茬槐锟犲箳婢跺本鐣遍弶鍫ｎ潐椤斿瞼绮氶幐骞犱線濡?        """
        verts = np.asarray(self.pre_entry_guard_vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[0] == 0:
            return float('inf')
        p = np.asarray(point, dtype=float).reshape(3)
        return float(np.min(np.linalg.norm(verts - p.reshape(1, 3), axis=1)) - self._contact_radius_mm())

    def _head_wall_clearance(self) -> float:
        if self._head_wall_clearance_cache_step == self.step_count:
            return float(self._head_wall_clearance_cache)
        rigid = self._current_rigid_state()
        if rigid.shape[0] == 0:
            return float('inf')
        if self.is_beam_realtime:
            near_contact = self.wall_contact_active or self.step_count <= 5
            refresh_stride = (
                max(int(BEAM_HEAD_CLEARANCE_REFRESH_NEAR_STEPS), 1)
                if near_contact
                else max(int(BEAM_HEAD_CLEARANCE_REFRESH_FAR_STEPS), 1)
            )
            if self._head_wall_clearance_exact_step >= 0 and (self.step_count - self._head_wall_clearance_exact_step) < refresh_stride:
                self._head_wall_clearance_cache_step = self.step_count
                return float(self._head_wall_clearance_cache)
        if self.is_native_strict and self.use_native_gui_wallclock_control:
            near_contact = (
                self.wall_contact_active
                or self._native_strict_barrier_active_node_count() > 0
                or self.step_count <= 5
            )
            refresh_stride = 1 if near_contact else max(int(ELASTICROD_STRICT_GUI_SURFACE_REFRESH_FAR_STEPS), 1)
            if self._head_wall_clearance_exact_step >= 0 and (self.step_count - self._head_wall_clearance_exact_step) < refresh_stride:
                self._head_wall_clearance_cache_step = self.step_count
                return float(self._head_wall_clearance_cache)
        exact_projection = True
        if self.is_native_strict and self.use_native_gui_wallclock_control:
            exact_projection = bool(self._strict_surface_exact_monitor_required())
        clearances = [
            self._point_wall_clearance(rigid[i, :3], self._node_s(i), exact_projection=exact_projection)
            for i in self._tip_probe_indices()
        ]
        clearance = float(min(clearances)) if clearances else float('inf')
        self._head_wall_clearance_cache_step = self.step_count
        if exact_projection:
            self._head_wall_clearance_exact_step = self.step_count
        self._head_wall_clearance_cache = clearance
        return clearance

    def _tip_centerline_offset_mm(self) -> float:
        tip_pos, _ = self._tip_pose()
        if tip_pos.shape[0] == 0 or self.centerline.shape[0] == 0:
            return 0.0
        proj_point, _ = self._tip_projection_to_centerline()
        return float(np.linalg.norm(tip_pos - proj_point))

    def _beam_drive_target_point(self, idx: int) -> np.ndarray:
        base = self.initial_wire_centers[idx]
        return base + float(self.commanded_push_mm) * self.insertion_direction

    def _write_beam_drive_rest_target(self, idx: int, target: np.ndarray, quat: np.ndarray) -> None:
        if self._rest is None:
            return
        rest = np.array(_read(self._rest), dtype=float, copy=True)
        if idx < 0 or idx >= rest.shape[0]:
            return
        rest[idx, :3] = np.asarray(target, dtype=float).reshape(3)
        rest[idx, 3:7] = np.asarray(quat, dtype=float).reshape(4)
        with _writeable(self._rest) as rest_out:
            rest_out[:] = rest

    def _constrain_point(self, point: np.ndarray, nominal_s: float, exact_projection: bool = False) -> np.ndarray:
        p = np.asarray(point, dtype=float).reshape(3)
        if self._use_pre_entry_access_guide(nominal_s):
            return self._pre_entry_access_point(nominal_s)
        if self.is_native_strict and nominal_s < VIRTUAL_SHEATH_RELEASE_S_MM:
            support_clearance = self._strict_external_support_clearance_mm(p)
            if np.isfinite(support_clearance):
                return self._strict_project_inside_external_support(p)
        if nominal_s < VIRTUAL_SHEATH_RELEASE_S_MM:
            return p
        if self.is_native_safe and self.vessel_surface_query is not None:
            surface_sample = self._point_surface_clearance_sample(
                p,
                nominal_s,
                exact_projection=True,
            )
            if surface_sample is not None and float(surface_sample[0]) < -LUMEN_CONSTRAINT_TOLERANCE_MM:
                return self._project_inside_surface(p, nominal_s, exact_projection=True)
        if nominal_s >= 0.0 and self.use_fast_lumen:
            if exact_projection:
                q, proj_s = self._project_to_centerline(p)
                radius = float(np.interp(proj_s, self.centerline_cum, self.fast_lumen_profile_mm))
            else:
                q, proj_s, radius = self._nominal_centerline_frame(nominal_s)
            radial_vec = p - q
            radial = float(np.linalg.norm(radial_vec))
            allowed = max(radius - LUMEN_CLEARANCE_MM, 0.0)
            if radial <= allowed + 1e-9 or radial < 1e-12:
                return q.copy() if radial < 1e-12 and allowed <= 1e-9 else p
            return q + (allowed / radial) * radial_vec
        return p

    def _constrain_wire(self) -> None:
        """
        濞戞挸顦惇浼存嚂瀹€鍐厬闁告帒妫楃槐鎴︽晬?        1. 闁活亞鍠愰婊堟儍閸曨垱濮滅紒灞芥健閳ь剙绻楃换鏇㈠及椤栫偞娴?CollisionNode + 缂佹拝闄勫顐⑿ч崒婢帡宕抽…鎺斿耿
        2. 閺夆晜鐟╅崳鐑芥儍閸曨喖骞囬柛鎰噺婵洩銇愰崡鐐叉锭闁革负鍔忔俊顓㈡倷閻熸澘鍤掔紓浣哥箲濡叉垿寮伴幆褍姣夐柤鍨⒐濡炲倿宕楀鍐亢闁瑰嘲顦ú鏍晬?        3. 濞达絾鎸搁ˇ璇测枔娴ｅ啫顏熼柛姘煎灣閺併倝鎼瑰顓炵彲闂傚顭囬鎼佹晬鐏炶棄娑ч梺澶哥劍閾嗛柛姘灱閸ゆ粓鎮介崡鐐差唺闁挎稑濂旂粭澶屽枈閹峰矈鏀ㄧ紒鐘偓鍐叉暥闁煎搫鍊婚崑锝夊Υ?        """
        if not self.enable_vessel_lumen_constraint:
            self.tip_contact_correction_mm = 0.0
            return

        x = np.array(_read(self._pos), dtype=float, copy=True)
        v = np.array(_read(self._vel), dtype=float, copy=True)
        centers = x[:, :3].copy()
        centers_before = centers.copy()
        corrected: set[int] = set()
        orientation_update: set[int] = set()
        lumen_nodes = 0
        sheath_nodes = 0
        sheath_quat = _quat_from_z_to(self.insertion_direction)
        rest = None
        rest_dirty = False
        safe_exact_guard_start = (
            max(0, centers.shape[0] - (self.magnetic_head_edge_count + 12))
            if self.is_native_safe
            else centers.shape[0]
        )
        safe_exact_surface_margin_mm = 0.12

        blend_nodes = 0

        def _mark_orientation_update(index: int) -> None:
            if centers.shape[0] <= 0:
                return
            lo = max(0, int(index) - 1)
            hi = min(int(centers.shape[0]) - 1, int(index) + 1)
            for idx in range(lo, hi + 1):
                orientation_update.add(idx)

        for i in range(centers.shape[0]):
            if self.use_kinematic_beam_insertion and i < self.drive_count:
                drive_target = self._beam_drive_target_point(i)
                if np.linalg.norm(drive_target - centers[i]) > 1e-9:
                    centers[i] = drive_target
                    corrected.add(i)
                    _mark_orientation_update(i)
                v[i, :3] = self.push_force_target_speed_mm_s * self.insertion_direction
                v[i, 3:] = 0.0
                x[i, 3:7] = sheath_quat
                if self._rest is not None:
                    if rest is None:
                        rest = np.array(_read(self._rest), dtype=float, copy=True)
                    if 0 <= i < rest.shape[0]:
                        rest[i, :3] = drive_target
                        rest[i, 3:7] = sheath_quat
                        rest_dirty = True
                continue

            nominal_s = self._node_s(i)
            if self._use_pre_entry_access_guide(nominal_s):
                access_fixed = self._pre_entry_access_point(nominal_s)
                if np.linalg.norm(access_fixed - centers[i]) > 1e-9:
                    centers[i] = access_fixed
                    corrected.add(i)
                    _mark_orientation_update(i)
                axial_speed = float(np.dot(v[i, :3], self.insertion_direction))
                v[i, :3] = axial_speed * self.insertion_direction
                v[i, 3:] = 0.0
                x[i, 3:7] = sheath_quat
                continue
            if self.enable_virtual_sheath and nominal_s < VIRTUAL_SHEATH_RELEASE_S_MM:
                sheath_nodes += 1
                sheath_fixed = self._virtual_sheath_point(i, centers[i])
                if np.linalg.norm(sheath_fixed - centers[i]) > 1e-9:
                    centers[i] = sheath_fixed
                    corrected.add(i)
                    _mark_orientation_update(i)
                axial_speed = float(np.dot(v[i, :3], self.insertion_direction))
                v[i, :3] = axial_speed * self.insertion_direction
                v[i, 3:] = 0.0
                x[i, 3:7] = sheath_quat
                continue

            lumen_nodes += 1
            clearance = self._point_wall_clearance(centers[i], nominal_s)
            fixed = None
            if self.is_native_safe and i >= safe_exact_guard_start:
                surface_sample = self._point_surface_clearance_sample(
                    centers[i],
                    nominal_s,
                    exact_projection=True,
                )
                if surface_sample is not None:
                    surface_clearance, closest, inward = surface_sample
                    if float(surface_clearance) < safe_exact_surface_margin_mm:
                        target_depth_mm = float(self._contact_radius_mm() + max(safe_exact_surface_margin_mm, 0.08))
                        fixed = (
                            np.asarray(closest, dtype=float).reshape(3)
                            + target_depth_mm * np.asarray(inward, dtype=float).reshape(3)
                        )
            if fixed is None and clearance >= -LUMEN_CONSTRAINT_TOLERANCE_MM:
                fixed = centers[i]
            elif fixed is None:
                if self.is_native_safe:
                    surface_sample = self._point_surface_clearance_sample(
                        centers[i],
                        nominal_s,
                        exact_projection=True,
                    )
                    if surface_sample is not None and float(surface_sample[0]) < -LUMEN_CONSTRAINT_TOLERANCE_MM:
                        fixed = self._project_inside_surface(centers[i], nominal_s, exact_projection=True)
                    else:
                        fixed = self._constrain_point(centers[i], nominal_s)
                else:
                    fixed = self._constrain_point(centers[i], nominal_s)

            sheath_alpha = self._sheath_blend_alpha(nominal_s) if self.enable_virtual_sheath else 0.0
            if sheath_alpha > 0.0:
                blend_nodes += 1
                sheath_fixed = self._virtual_sheath_point(i, fixed)
                fixed = (1.0 - sheath_alpha) * fixed + sheath_alpha * sheath_fixed
                axial_speed = float(np.dot(v[i, :3], self.insertion_direction))
                v[i, :3] = (1.0 - sheath_alpha) * v[i, :3] + sheath_alpha * axial_speed * self.insertion_direction
                v[i, 3:] *= max(0.0, 1.0 - sheath_alpha)

            if np.linalg.norm(fixed - centers[i]) > 1e-9:
                centers[i] = fixed
                corrected.add(i)
                _mark_orientation_update(i)

        if self.use_kinematic_beam_insertion:
            min_seg_len = MIN_SEGMENT_LENGTH_RATIO * self.rest_spacing_mm
            max_seg_len = MAX_SEGMENT_LENGTH_RATIO * self.rest_spacing_mm
            relax_iters = 3
            relax_start_index = max(1, self.drive_count)
        else:
            # 闁绘せ鏅濋幃濠囧箳閵娿儱顫旀俊顖椻偓宕囩濞戞挸顑戠槐婵嬪矗椤忓啰绠介柣锝嗙懄閻庮剛绮╅婊勨挄闁诡兛娴囩粩鐔兼⒐婢跺憡鍙忓璺虹▌缁辨繈鏌嗛崹顔煎赋閻庣懓顦崣蹇涘箮閺囩偛顨涢柟璺猴攻閺嗭綁寮介柅娑氼偩
            # 閺夆晜鍨抽幎鈧柟瀛樺姀閳ь剚绮嶉惁鈥愁潰閵夈儱绻侀柛鎺撴构缁楀宕ｉ娆忓殤闂傗偓鎼存挴鍋撳┑鍫熺暠閺夆晜鍔曟慨鈺冣偓娑崇畵閹藉ジ寮舵幊閳?            min_seg_len = 0.25 * self.rest_spacing_mm
            min_seg_len = 0.25 * self.rest_spacing_mm
            max_seg_len = 2.50 * self.rest_spacing_mm
            relax_iters = 1
            relax_start_index = 1

        for _ in range(relax_iters):
            for i in range(relax_start_index, centers.shape[0]):
                prev = centers[i - 1]
                vec = centers[i] - prev
                dist = float(np.linalg.norm(vec))
                if min_seg_len <= dist <= max_seg_len:
                    continue
                if dist > 1e-12:
                    direction = vec / dist
                else:
                    direction = self.insertion_direction.copy()
                desired_len = min_seg_len if dist < min_seg_len else max_seg_len
                candidate = prev + direction * desired_len
                nominal_s = self._node_s(i)
                if self._use_pre_entry_access_guide(nominal_s):
                    centers[i] = self._pre_entry_access_point(nominal_s)
                elif self.enable_virtual_sheath and nominal_s < VIRTUAL_SHEATH_RELEASE_S_MM:
                    centers[i] = self._virtual_sheath_point(i, candidate)
                else:
                    candidate = self._constrain_point(candidate, nominal_s)
                    sheath_alpha = self._sheath_blend_alpha(nominal_s) if self.enable_virtual_sheath else 0.0
                    if sheath_alpha > 0.0:
                        sheath_fixed = self._virtual_sheath_point(i, candidate)
                        candidate = (1.0 - sheath_alpha) * candidate + sheath_alpha * sheath_fixed
                    centers[i] = candidate
                if np.linalg.norm(centers[i] - centers_before[i]) > 1e-9:
                    corrected.add(i)
                    _mark_orientation_update(i)

        if self.is_native_safe and self.vessel_surface_query is not None and centers.shape[0] >= 2:
            safe_edge_start = max(self.node_count - (self.magnetic_head_edge_count + 10), 0)
            target_depth_mm = float(self._contact_radius_mm() + max(safe_exact_surface_margin_mm, 0.08))
            for _ in range(3):
                changed_edge = False
                for edge_idx in range(safe_edge_start, centers.shape[0] - 1):
                    p0 = centers[edge_idx, :3]
                    p1 = centers[edge_idx + 1, :3]
                    for alpha in (0.20, 0.40, 0.60, 0.80):
                        point = (1.0 - alpha) * p0 + alpha * p1
                        nominal_s = (1.0 - alpha) * self._node_s(edge_idx) + alpha * self._node_s(edge_idx + 1)
                        sample = self._point_surface_clearance_sample(
                            point,
                            nominal_s,
                            exact_projection=True,
                        )
                        if sample is None:
                            continue
                        clearance, closest, inward = sample
                        if float(clearance) >= safe_exact_surface_margin_mm:
                            continue
                        projected_point = np.asarray(closest, dtype=float).reshape(3) + target_depth_mm * np.asarray(inward, dtype=float).reshape(3)
                        displacement = projected_point - point
                        displacement_norm = float(np.linalg.norm(displacement))
                        if displacement_norm <= 1.0e-9:
                            continue
                        w0 = float(np.clip(1.0 - alpha, 0.0, 1.0))
                        w1 = float(np.clip(alpha, 0.0, 1.0))
                        scale = 1.0 / max(w0 * w0 + w1 * w1, 1.0e-9)
                        centers[edge_idx, :3] += scale * w0 * displacement
                        centers[edge_idx + 1, :3] += scale * w1 * displacement
                        corrected.add(edge_idx)
                        corrected.add(edge_idx + 1)
                        _mark_orientation_update(edge_idx)
                        _mark_orientation_update(edge_idx + 1)
                        changed_edge = True
                        p0 = centers[edge_idx, :3]
                        p1 = centers[edge_idx + 1, :3]
                if not changed_edge:
                    break

        if self.is_native_safe and centers.shape[0] >= 3:
            safe_head_start = max(1, centers.shape[0] - max(self.magnetic_head_edge_count + 3, 5))
            safe_head_min_seg_len = 0.96 * self.rest_spacing_mm
            safe_head_max_seg_len = 1.06 * self.rest_spacing_mm
            safe_turn_limit_deg = 58.0 if self.wall_contact_active else 52.0
            safe_velocity_damp = 0.25 if self.wall_contact_active else 0.45
            for _ in range(1 if self.wall_contact_active else 2):
                changed_head = False
                for i in range(safe_head_start, centers.shape[0]):
                    prev = centers[i - 1, :3]
                    current = centers[i, :3]
                    nominal_s = self._node_s(i)
                    vec = current - prev
                    dist = float(np.linalg.norm(vec))
                    centerline_dir = self._centerline_tangent(max(nominal_s, 0.0))
                    direction = _normalize(vec if dist > 1.0e-9 else centerline_dir)
                    if np.linalg.norm(direction) < 1.0e-12:
                        direction = self.insertion_direction.copy()

                    turn_deg = 0.0
                    if i > safe_head_start:
                        prev_vec = centers[i - 1, :3] - centers[i - 2, :3]
                        prev_norm = float(np.linalg.norm(prev_vec))
                        if prev_norm > 1.0e-9:
                            prev_dir = prev_vec / prev_norm
                            turn_deg = float(
                                np.degrees(
                                    np.arccos(
                                        np.clip(float(np.dot(prev_dir, direction)), -1.0, 1.0)
                                    )
                                )
                            )
                            if turn_deg > safe_turn_limit_deg:
                                blend = float(np.clip((turn_deg - safe_turn_limit_deg) / 35.0, 0.0, 1.0))
                                direction = _normalize(
                                    (1.0 - 0.52 * blend) * direction
                                    + 0.38 * blend * prev_dir
                                    + 0.30 * blend * centerline_dir
                                )
                                if np.linalg.norm(direction) < 1.0e-12:
                                    direction = prev_dir

                    needs_length_fix = (dist < safe_head_min_seg_len) or (dist > safe_head_max_seg_len)
                    needs_turn_fix = turn_deg > safe_turn_limit_deg
                    if not (needs_length_fix or needs_turn_fix):
                        continue

                    desired_len = float(
                        np.clip(
                            dist if dist > 1.0e-9 else self.rest_spacing_mm,
                            safe_head_min_seg_len,
                            safe_head_max_seg_len,
                        )
                    )
                    candidate = prev + desired_len * direction
                    candidate = self._constrain_point(candidate, nominal_s, exact_projection=True)
                    if np.linalg.norm(candidate - current) <= 1.0e-6:
                        continue
                    centers[i, :3] = candidate
                    corrected.add(i)
                    _mark_orientation_update(i)
                    changed_head = True
                if not changed_head:
                    break

            for i in range(safe_head_start, centers.shape[0]):
                if i not in corrected:
                    continue
                v[i, :3] *= safe_velocity_damp
                v[i, 3:] = 0.0

        pre_entry_guard_edges = 0
        if (not self.is_native_backend) and (not self.enable_virtual_sheath) and BEAM_PRE_ENTRY_ACCESS_GUIDE_MM > 0.0:
            guide_span = max(float(BEAM_PRE_ENTRY_ACCESS_GUIDE_MM), 0.0) + 2.0 * self.rest_spacing_mm
            for i in range(centers.shape[0] - 1):
                s0 = self._node_s(i)
                s1 = self._node_s(i + 1)
                if min(s0, s1) > self.rest_spacing_mm:
                    continue
                if max(s0, s1) < -guide_span:
                    continue
                midpoint = 0.5 * (centers[i] + centers[i + 1])
                if self._pre_entry_surface_clearance(midpoint) >= self.pre_entry_guard_trigger_mm:
                    continue
                snapped = False
                for j, nominal_s in ((i, s0), (i + 1, s1)):
                    if nominal_s >= 0.0:
                        continue
                    access_fixed = self._pre_entry_access_point(nominal_s)
                    if np.linalg.norm(access_fixed - centers[j]) > 1e-9:
                        centers[j] = access_fixed
                        corrected.add(j)
                        _mark_orientation_update(j)
                        snapped = True
                    axial_speed = float(np.dot(v[j, :3], self.insertion_direction))
                    v[j, :3] = axial_speed * self.insertion_direction
                    v[j, 3:] = 0.0
                    x[j, 3:7] = sheath_quat
                if snapped:
                    pre_entry_guard_edges += 1

        probe_indices = [i for i in self._tip_probe_indices() if self._node_s(i) >= VIRTUAL_SHEATH_RELEASE_S_MM]
        self.tip_contact_correction_mm = (
            0.0 if not probe_indices else float(max(np.linalg.norm(centers[i] - centers_before[i]) for i in probe_indices))
        )
        if not np.isfinite(self.tip_contact_correction_mm):
            self.tip_contact_correction_mm = float(WIRE_TOTAL_LENGTH_MM)
        else:
            self.tip_contact_correction_mm = float(np.clip(self.tip_contact_correction_mm, 0.0, WIRE_TOTAL_LENGTH_MM))

        for i in range(centers.shape[0]):
            x[i, :3] = centers[i]
            nominal_s = self._node_s(i)
            if self._use_pre_entry_access_guide(nominal_s) or (self.use_kinematic_beam_insertion and i < self.drive_count) or (
                self.enable_virtual_sheath and nominal_s < VIRTUAL_SHEATH_RELEASE_S_MM
            ):
                x[i, 3:7] = sheath_quat
            elif self.use_kinematic_beam_insertion or i in orientation_update:
                if i == centers.shape[0] - 1:
                    tan = _normalize(centers[i] - centers[i - 1])
                else:
                    tan = _normalize(centers[i + 1] - centers[max(i - 1, 0)])
                if np.linalg.norm(tan) < 1e-12:
                    tan = self.insertion_direction.copy()
                x[i, 3:7] = _quat_from_z_to(tan)
            if i in corrected and nominal_s >= VIRTUAL_SHEATH_RELEASE_S_MM:
                if self.use_kinematic_beam_insertion:
                    v[i, :] = 0.0
                else:
                    corr = centers[i] - centers_before[i]
                    corr_n = float(np.linalg.norm(corr))
                    if corr_n > 1e-9:
                        inward = corr / corr_n
                        inward_speed = float(np.dot(v[i, :3], inward))
                        if inward_speed < 0.0:
                            v[i, :3] -= inward_speed * inward
                    v[i, 3:] *= 0.5

        with _writeable(self._pos) as pos:
            pos[:] = x
        with _writeable(self._vel) as vel:
            vel[:] = v
        if self.is_native_backend and self._rod_state_pos is not None:
            n = min(int(x.shape[0]), int(np.asarray(_read(self._rod_state_pos)).shape[0]))
            if n > 0:
                rod_pos = np.array(_read(self._rod_state_pos), dtype=float, copy=True)
                if rod_pos.ndim == 2 and rod_pos.shape[1] >= 3:
                    rod_pos[:n, :3] = 1.0e-3 * x[:n, :3]
                    with _writeable(self._rod_state_pos) as out_rod_pos:
                        out_rod_pos[:] = rod_pos
                if self._rod_state_free_pos is not None:
                    rod_free_pos = np.array(_read(self._rod_state_free_pos), dtype=float, copy=True)
                    if rod_free_pos.ndim == 2 and rod_free_pos.shape[0] >= n and rod_free_pos.shape[1] >= 3:
                        rod_free_pos[:n, :3] = 1.0e-3 * x[:n, :3]
                        with _writeable(self._rod_state_free_pos) as out_rod_free_pos:
                            out_rod_free_pos[:] = rod_free_pos
                if self._rod_state_vel is not None:
                    rod_vel = np.array(_read(self._rod_state_vel), dtype=float, copy=True)
                    if rod_vel.ndim == 2 and rod_vel.shape[0] >= n and rod_vel.shape[1] >= 3:
                        rod_vel[:n, :3] = 1.0e-3 * v[:n, :3]
                        with _writeable(self._rod_state_vel) as out_rod_vel:
                            out_rod_vel[:] = rod_vel
                if self._rod_state_free_vel is not None:
                    rod_free_vel = np.array(_read(self._rod_state_free_vel), dtype=float, copy=True)
                    if rod_free_vel.ndim == 2 and rod_free_vel.shape[0] >= n and rod_free_vel.shape[1] >= 3:
                        rod_free_vel[:n, :3] = 1.0e-3 * v[:n, :3]
                        with _writeable(self._rod_state_free_vel) as out_rod_free_vel:
                            out_rod_free_vel[:] = rod_free_vel
        if rest_dirty and rest is not None:
            with _writeable(self._rest) as rest_out:
                rest_out[:] = rest
        self._invalidate_geometry_cache()
        self._invalidate_surface_probe_cache()

        if not self._constraint_diag_printed:
            print(
                f'[INFO] Constraint safety chain enabled: lumenProjection={self.enable_vessel_lumen_constraint}, '
                f'virtualSheath={self.enable_virtual_sheath}, contactDistance={CONTACT_DISTANCE_MM:.3f} mm, '
                f'lumenNodes={lumen_nodes}, virtualSheathNodes={sheath_nodes}, '
                f'preEntryGuide={BEAM_PRE_ENTRY_ACCESS_GUIDE_MM:.2f} mm, '
                f'preEntryGuardTrigger={self.pre_entry_guard_trigger_mm:.2f} mm, '
                f'releaseS={VIRTUAL_SHEATH_RELEASE_S_MM:.2f} mm, '
                f'blendOut={VIRTUAL_SHEATH_BLEND_OUT_MM:.2f} mm, '
                f'segmentClamp=[{MIN_SEGMENT_LENGTH_RATIO:.3f}, {MAX_SEGMENT_LENGTH_RATIO:.3f}], '
                f'blendNodes={blend_nodes}, preEntryGuardEdges={pre_entry_guard_edges}'
            )
            self._constraint_diag_printed = True

    def _apply_native_strict_postsolve_guard(self) -> tuple[int, int]:
        if (not self.enable_native_strict_postsolve_guard) or self._rod_state_pos is None:
            return 0, 0

        rod_pos = np.array(_read(self._rod_state_pos), dtype=float, copy=True)
        if rod_pos.ndim != 2 or rod_pos.shape[0] == 0:
            return 0, 0

        rod_vel = (
            np.array(_read(self._rod_state_vel), dtype=float, copy=True)
            if self._rod_state_vel is not None
            else None
        )
        rod_free_pos = (
            np.array(_read(self._rod_state_free_pos), dtype=float, copy=True)
            if self._rod_state_free_pos is not None
            else None
        )
        rod_free_vel = (
            np.array(_read(self._rod_state_free_vel), dtype=float, copy=True)
            if self._rod_state_free_vel is not None
            else None
        )

        centers_mm = 1000.0 * rod_pos[:, :3].copy()
        corrected = 0
        corrected_edge_samples = 0
        clipped = 0
        locally_clipped = 0
        hold_damped = 0
        min_clearance = float('inf')
        max_speed_before = 0.0
        clamp_tolerance_mm = float(self.native_strict_lumen_clamp_tolerance_mm)
        guard_enter_clearance_mm = 0.025 if self.is_native_strict else -clamp_tolerance_mm
        corrected_node_indices: set[int] = set()
        corrected_edge_indices: set[int] = set()
        surrogate_disagreement_margin_mm = max(0.75, 2.0 * float(self._contact_radius_mm()))
        exact_surface_trust_clearance_mm = 0.20
        prefer_surrogate_recovery = False
        physical_clearance = float('inf')
        physical_tip_recovery = False

        def _clip_linear_speed(arr: np.ndarray | None) -> int:
            nonlocal max_speed_before
            if (
                arr is None
                or arr.ndim != 2
                or arr.shape[0] == 0
                or arr.shape[1] < 3
                or self.native_strict_max_linear_speed_mm_s <= 0.0
            ):
                return 0
            speeds_mm_s = 1000.0 * np.linalg.norm(arr[:, :3], axis=1)
            if speeds_mm_s.size:
                max_speed_before = max(max_speed_before, float(np.max(speeds_mm_s)))
            mask = speeds_mm_s > self.native_strict_max_linear_speed_mm_s
            if self.is_native_strict:
                hand_indices = [idx for idx in self._strict_hand_push_indices() if idx < int(mask.shape[0])]
                if hand_indices:
                    mask[np.asarray(hand_indices, dtype=int)] = False
            if not np.any(mask):
                return 0
            scales = self.native_strict_max_linear_speed_mm_s / np.maximum(speeds_mm_s[mask], 1.0e-12)
            arr[mask, :3] *= scales.reshape(-1, 1)
            return int(np.count_nonzero(mask))

        def _clip_selected_linear_speed(arr: np.ndarray | None, indices: set[int], cap_mm_s: float) -> int:
            nonlocal max_speed_before
            if (
                arr is None
                or arr.ndim != 2
                or arr.shape[0] == 0
                or arr.shape[1] < 3
                or cap_mm_s <= 0.0
                or not indices
            ):
                return 0
            valid = sorted(idx for idx in indices if 0 <= idx < arr.shape[0])
            if not valid:
                return 0
            node_idx = np.asarray(valid, dtype=int)
            speeds_mm_s = 1000.0 * np.linalg.norm(arr[node_idx, :3], axis=1)
            if speeds_mm_s.size:
                max_speed_before = max(max_speed_before, float(np.max(speeds_mm_s)))
            mask = speeds_mm_s > cap_mm_s
            if not np.any(mask):
                return 0
            selected = node_idx[mask]
            scales = cap_mm_s / np.maximum(speeds_mm_s[mask], 1.0e-12)
            arr[selected, :3] *= scales.reshape(-1, 1)
            return int(np.count_nonzero(mask))

        def _damp_linear_velocity(arr: np.ndarray | None, factor: float) -> int:
            nonlocal max_speed_before
            if (
                arr is None
                or arr.ndim != 2
                or arr.shape[0] == 0
                or arr.shape[1] < 3
                or factor >= (1.0 - 1.0e-9)
                or factor <= 0.0
            ):
                return 0
            speeds_mm_s = 1000.0 * np.linalg.norm(arr[:, :3], axis=1)
            if speeds_mm_s.size:
                max_speed_before = max(max_speed_before, float(np.max(speeds_mm_s)))
            moving = speeds_mm_s > 1.0e-6
            if not np.any(moving):
                return 0
            arr[moving, :3] *= float(factor)
            return int(np.count_nonzero(moving))

        if self.native_strict_barrier_enabled:
            native_clearance = self._native_debug_scalar(
                self._native_debug_min_lumen_clearance_mm,
                default=float('inf'),
            )
            barrier_nodes = self._native_strict_barrier_active_node_count() if self.is_native_strict else 0
            if self.is_native_strict:
                physical_clearance = float(self._native_strict_physical_contact_clearance_mm())
                if np.isfinite(physical_clearance):
                    min_clearance = min(min_clearance, physical_clearance)
            surface_clearance = float('inf')
            if self._strict_surface_exact_monitor_required(native_clearance):
                surface_clearance = self._surface_min_clearance_mm()
            exact_surface_safe_override = bool(
                np.isfinite(native_clearance)
                and native_clearance < -clamp_tolerance_mm
                and np.isfinite(surface_clearance)
                and surface_clearance > exact_surface_trust_clearance_mm
            )
            if np.isfinite(native_clearance) and (not exact_surface_safe_override):
                min_clearance = min(min_clearance, float(native_clearance))
            if np.isfinite(surface_clearance):
                min_clearance = min(min_clearance, float(surface_clearance))
            prefer_surrogate_recovery = bool(
                np.isfinite(native_clearance)
                and native_clearance < -clamp_tolerance_mm
                and (not exact_surface_safe_override)
                and (
                    (not np.isfinite(surface_clearance))
                    or (
                        surface_clearance > exact_surface_trust_clearance_mm
                        and surface_clearance >= native_clearance + surrogate_disagreement_margin_mm
                    )
                )
            )
            emergency_recovery = bool(
                self.enable_native_strict_lumen_clamp
                and (
                    (
                        np.isfinite(native_clearance)
                        and native_clearance < guard_enter_clearance_mm
                        and (not exact_surface_safe_override)
                    )
                    or (np.isfinite(surface_clearance) and surface_clearance < guard_enter_clearance_mm)
                    or (np.isfinite(physical_clearance) and physical_clearance < guard_enter_clearance_mm)
                )
            )
            physical_tip_recovery = bool(
                self.enable_native_strict_lumen_clamp
                and np.isfinite(physical_clearance)
                and physical_clearance < guard_enter_clearance_mm
            )
            if emergency_recovery:
                exact_surface_clearance = self._surface_min_clearance_mm()
                if np.isfinite(exact_surface_clearance):
                    min_clearance = min(min_clearance, float(exact_surface_clearance))
            if min_clearance < -clamp_tolerance_mm:
                should_log_deficit = (
                    self._native_last_barrier_deficit_log_step < 0
                    or (self.step_count - self._native_last_barrier_deficit_log_step) >= 20
                    or min_clearance <= (self._native_last_barrier_deficit_mm - 0.05)
                )
                if should_log_deficit:
                    print(
                        f'[WARN] [elasticrod-strict-guard] clearance deficit observed under native barrier: '
                        f'step={self.step_count} clearance={min_clearance:.4f} mm '
                        f'holdActive={self._native_strict_hold_active_this_step}'
                    )
                    self._native_last_barrier_deficit_log_step = self.step_count
                    self._native_last_barrier_deficit_mm = float(min_clearance)
            else:
                self._native_last_barrier_deficit_log_step = -1
                self._native_last_barrier_deficit_mm = 0.0
            if (not self.enable_native_strict_lumen_clamp) or (not emergency_recovery):
                hold_damping_factor = 1.0
                if self.is_native_strict and self._native_virtual_sheath_paused:
                    support_stretch = self._native_debug_array(self._native_debug_stretch)
                    global_stretch = float(np.max(np.abs(support_stretch))) if support_stretch.size else 0.0
                    global_soft_limit, global_hard_limit = self._native_strict_global_stretch_limits()
                    clearance_danger_gate = (
                        float(np.clip((0.24 - min_clearance) / 0.18, 0.0, 1.0))
                        if np.isfinite(min_clearance)
                        else 0.0
                    )
                    barrier_danger_gate = float(
                        np.clip(
                            (barrier_nodes - max(int(ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT) - 2, 2)) / 3.0,
                            0.0,
                            1.0,
                        )
                    )
                    global_danger_gate = float(
                        np.clip(
                            (global_stretch - 0.70 * global_soft_limit)
                            / max(global_hard_limit - 0.70 * global_soft_limit, 1.0e-6),
                            0.0,
                            1.0,
                        )
                    )
                    hold_damping_gate = max(
                        min(clearance_danger_gate, max(barrier_danger_gate, 0.25)),
                        global_danger_gate,
                    )
                    if hold_damping_gate > 0.0:
                        hold_damping_factor = float(np.clip(1.0 - (0.06 + 0.10 * hold_damping_gate), 0.82, 0.96))
                hold_damped = max(
                    _damp_linear_velocity(rod_vel, hold_damping_factor),
                    _damp_linear_velocity(rod_free_vel, hold_damping_factor),
                )
                clipped = max(_clip_linear_speed(rod_vel), _clip_linear_speed(rod_free_vel))
                if clipped > 0 or hold_damped > 0:
                    if self._rod_state_vel is not None and rod_vel is not None:
                        with _writeable(self._rod_state_vel) as out_vel:
                            out_vel[:] = rod_vel
                    if self._rod_state_free_vel is not None and rod_free_vel is not None:
                        with _writeable(self._rod_state_free_vel) as out_free_vel:
                            out_free_vel[:] = rod_free_vel
                if self._should_log_native_strict_guard(min_clearance_mm=min_clearance):
                    min_clearance_value = min_clearance if np.isfinite(min_clearance) else float('nan')
                    print(
                        f'[INFO] [elasticrod-strict-guard] step={self.step_count} '
                        f'correctedNodes=0 correctedEdgeSamples=0 clippedVelNodes={clipped + hold_damped} '
                        f'minClearance={min_clearance_value:.4f} mm maxSpeedBefore={max_speed_before:.2f} mm/s '
                        f'nativeBarrier={self.native_strict_barrier_enabled}'
                    )
                    self._native_last_strict_guard_step = self.step_count
                return 0, clipped + hold_damped

        if physical_tip_recovery:
            tip_recovery_indices = sorted({
                *[int(idx) for idx in self._tip_probe_indices()],
                *range(max(self.node_count - (self.magnetic_head_edge_count + 2), 0), self.node_count),
            })
            for i in tip_recovery_indices:
                if not (0 <= i < centers_mm.shape[0]):
                    continue
                nominal_s = self._node_s(i)
                surface_sample = self._point_surface_clearance_sample(
                    centers_mm[i],
                    nominal_s,
                    exact_projection=True,
                )
                corrected_point = (
                    self._project_inside_surface(centers_mm[i], nominal_s, exact_projection=True)
                    if surface_sample is not None and float(surface_sample[0]) < guard_enter_clearance_mm
                    else self._constrain_point(centers_mm[i], nominal_s, exact_projection=True)
                )
                correction = corrected_point - centers_mm[i]
                correction_norm = float(np.linalg.norm(correction))
                if correction_norm <= 1.0e-9:
                    continue
                inward = correction / correction_norm
                outward = -inward
                centers_mm[i] = corrected_point
                rod_pos[i, :3] = 1.0e-3 * corrected_point
                if rod_free_pos is not None and i < rod_free_pos.shape[0]:
                    rod_free_pos[i, :3] = 1.0e-3 * corrected_point
                if rod_vel is not None and i < rod_vel.shape[0] and rod_vel.shape[1] >= 3:
                    outward_speed = float(np.dot(rod_vel[i, :3], outward))
                    if outward_speed > 0.0:
                        rod_vel[i, :3] -= outward_speed * outward
                if rod_free_vel is not None and i < rod_free_vel.shape[0] and rod_free_vel.shape[1] >= 3:
                    outward_speed = float(np.dot(rod_free_vel[i, :3], outward))
                    if outward_speed > 0.0:
                        rod_free_vel[i, :3] -= outward_speed * outward
                corrected += 1
                corrected_node_indices.add(i)
                for neighbor_idx in (i - 1, i + 1):
                    if 0 <= neighbor_idx < centers_mm.shape[0]:
                        corrected_node_indices.add(neighbor_idx)

        for i in range(centers_mm.shape[0]):
            nominal_s = self._node_s(i)
            if not self._strict_native_surface_guard_eligible(centers_mm[i], nominal_s):
                continue

            surrogate_clearance = self._point_wall_clearance(centers_mm[i], nominal_s, exact_projection=True)
            surface_sample = None
            if self.native_strict_barrier_enabled:
                surface_sample = self._point_surface_clearance_sample(
                    centers_mm[i],
                    nominal_s,
                    exact_projection=True,
                )
            elif (
                self.is_native_strict
                and np.isfinite(surrogate_clearance)
                and surrogate_clearance < max(guard_enter_clearance_mm, exact_surface_trust_clearance_mm)
            ):
                # Strict non-barrier mode still uses the centerline-radius
                # surrogate for cheap broad-phase wall checks. Near the first
                # bend that surrogate can stay pinned at ~0 even when the exact
                # STL distance is safely positive, which makes the tip look
                # glued to the wall and triggers a corrective "snap" every
                # frame. Verify suspicious near-contact nodes against the exact
                # surface before treating them as a true penetration.
                surface_sample = self._point_surface_clearance_sample(
                    centers_mm[i],
                    nominal_s,
                    exact_projection=True,
                )
            surface_clearance = float(surface_sample[0]) if surface_sample is not None else float('inf')
            exact_surface_safe_override = bool(
                self.is_native_strict
                and np.isfinite(surface_clearance)
                and surface_clearance > exact_surface_trust_clearance_mm
                and (
                    (not np.isfinite(surrogate_clearance))
                    or surface_clearance >= surrogate_clearance + surrogate_disagreement_margin_mm
                )
            )
            use_surrogate_recovery = bool(
                prefer_surrogate_recovery
                and np.isfinite(surrogate_clearance)
                and (
                    (not np.isfinite(surface_clearance))
                    or surface_clearance >= surrogate_clearance + surrogate_disagreement_margin_mm
                )
            )
            clearance = (
                float(surrogate_clearance)
                if use_surrogate_recovery
                else (
                    surface_clearance
                    if np.isfinite(surface_clearance)
                    else float(surrogate_clearance)
                )
            )
            if np.isfinite(clearance):
                min_clearance = min(min_clearance, clearance)
            if exact_surface_safe_override:
                continue
            if (not self.enable_native_strict_lumen_clamp) or clearance >= guard_enter_clearance_mm:
                continue

            corrected_point = (
                self._constrain_point(centers_mm[i], nominal_s, exact_projection=True)
                if use_surrogate_recovery or surface_sample is None or not self.native_strict_barrier_enabled
                else self._project_inside_surface(centers_mm[i], nominal_s, exact_projection=True)
            )
            correction = corrected_point - centers_mm[i]
            correction_norm = float(np.linalg.norm(correction))
            if correction_norm <= 1.0e-9:
                continue

            if surface_sample is not None and (not use_surrogate_recovery):
                normal = -np.asarray(surface_sample[2], dtype=float).reshape(3)
            else:
                rel = centers_mm[i] - self.entry_point.reshape(3)
                axial_mm = float(np.dot(rel, self.insertion_direction))
                if self.is_native_strict and nominal_s < VIRTUAL_SHEATH_RELEASE_S_MM and axial_mm < 0.0:
                    radial_vec = rel - axial_mm * self.insertion_direction
                    radial_norm = float(np.linalg.norm(radial_vec))
                    normal = radial_vec / radial_norm if radial_norm > 1.0e-9 else self.insertion_direction.copy()
                else:
                    q, proj_s = self._project_to_centerline(centers_mm[i])
                    radial_vec = centers_mm[i] - q
                    radial_norm = float(np.linalg.norm(radial_vec))
                    normal = radial_vec / radial_norm if radial_norm > 1.0e-9 else self._centerline_tangent(proj_s)

            centers_mm[i] = corrected_point
            rod_pos[i, :3] = 1.0e-3 * corrected_point
            if rod_free_pos is not None and i < rod_free_pos.shape[0]:
                rod_free_pos[i, :3] = 1.0e-3 * corrected_point
            if rod_vel is not None and i < rod_vel.shape[0] and rod_vel.shape[1] >= 3:
                outward_speed = float(np.dot(rod_vel[i, :3], normal))
                if outward_speed > 0.0:
                    rod_vel[i, :3] -= outward_speed * normal
            if rod_free_vel is not None and i < rod_free_vel.shape[0] and rod_free_vel.shape[1] >= 3:
                outward_speed = float(np.dot(rod_free_vel[i, :3], normal))
                if outward_speed > 0.0:
                    rod_free_vel[i, :3] -= outward_speed * normal
            corrected += 1
            corrected_node_indices.add(i)
            for neighbor_idx in (i - 1, i + 1):
                if 0 <= neighbor_idx < centers_mm.shape[0]:
                    corrected_node_indices.add(neighbor_idx)
            for edge_idx in (i - 1, i):
                if 0 <= edge_idx < (centers_mm.shape[0] - 1):
                    corrected_edge_indices.add(edge_idx)

        if self.enable_native_strict_lumen_clamp:
            target_depth = float(self._contact_radius_mm() + max(self.native_strict_lumen_clamp_tolerance_mm, 0.10))
            second_pass_target_depth = float(self._contact_radius_mm() + max(self.native_strict_lumen_clamp_tolerance_mm, 0.12))
            magnetic_edge_start = max(centers_mm.shape[0] - 1 - int(self.magnetic_head_edge_count), 0)
            head_edge_emergency_tol_mm = 2.0 * clamp_tolerance_mm

            def _apply_edge_sample_correction(
                edge_idx: int,
                alpha: float,
                point: np.ndarray,
                closest: np.ndarray,
                inward: np.ndarray,
                target_depth_mm: float,
            ) -> bool:
                projected_point = np.asarray(closest, dtype=float).reshape(3) + target_depth_mm * np.asarray(inward, dtype=float).reshape(3)
                displacement = projected_point - np.asarray(point, dtype=float).reshape(3)
                displacement_norm = float(np.linalg.norm(displacement))
                if displacement_norm <= 1.0e-9:
                    return False

                w0 = float(np.clip(1.0 - alpha, 0.0, 1.0))
                w1 = float(np.clip(alpha, 0.0, 1.0))
                correction_scale = static_correction_scale = 1.0 / max(w0 * w0 + w1 * w1, 1.0e-9)
                outward = -np.asarray(inward, dtype=float).reshape(3)
                any_applied = False
                for node_idx, weight in ((edge_idx, w0), (edge_idx + 1, w1)):
                    if weight <= 1.0e-9:
                        continue
                    centers_mm[node_idx] += static_correction_scale * weight * displacement
                    rod_pos[node_idx, :3] = 1.0e-3 * centers_mm[node_idx]
                    if rod_free_pos is not None and node_idx < rod_free_pos.shape[0]:
                        rod_free_pos[node_idx, :3] = 1.0e-3 * centers_mm[node_idx]
                    if rod_vel is not None and node_idx < rod_vel.shape[0] and rod_vel.shape[1] >= 3:
                        outward_speed = float(np.dot(rod_vel[node_idx, :3], outward))
                        if outward_speed > 0.0:
                            rod_vel[node_idx, :3] -= outward_speed * outward
                    if rod_free_vel is not None and node_idx < rod_free_vel.shape[0] and rod_free_vel.shape[1] >= 3:
                        outward_speed = float(np.dot(rod_free_vel[node_idx, :3], outward))
                        if outward_speed > 0.0:
                            rod_free_vel[node_idx, :3] -= outward_speed * outward
                    corrected_node_indices.add(node_idx)
                    any_applied = True
                if any_applied:
                    for neighbor_idx in (edge_idx - 1, edge_idx + 2):
                        if 0 <= neighbor_idx < centers_mm.shape[0]:
                            corrected_node_indices.add(neighbor_idx)
                    corrected_edge_indices.add(edge_idx)
                return any_applied

            for edge_idx, alpha, point, clearance, closest, inward in self._surface_edge_probe_samples():
                if edge_idx < 0 or edge_idx + 1 >= centers_mm.shape[0]:
                    continue
                nominal_s = (1.0 - alpha) * self._node_s(edge_idx) + alpha * self._node_s(edge_idx + 1)
                if not self._strict_native_surface_guard_eligible(point, nominal_s):
                    continue
                surrogate_clearance = self._point_wall_clearance(point, nominal_s, exact_projection=True)
                use_surrogate_recovery = bool(
                    prefer_surrogate_recovery
                    and np.isfinite(surrogate_clearance)
                    and clearance >= surrogate_clearance + surrogate_disagreement_margin_mm
                )
                effective_clearance = float(surrogate_clearance) if use_surrogate_recovery else float(clearance)
                if np.isfinite(effective_clearance):
                    min_clearance = min(min_clearance, effective_clearance)
                if effective_clearance >= guard_enter_clearance_mm:
                    continue
                if (
                    self.is_native_strict
                    and edge_idx >= magnetic_edge_start
                    and (not self.wall_contact_active)
                    and effective_clearance >= -head_edge_emergency_tol_mm
                ):
                    continue
                if use_surrogate_recovery:
                    corrected_point = self._constrain_point(point, nominal_s, exact_projection=True)
                    correction = corrected_point - np.asarray(point, dtype=float).reshape(3)
                    correction_norm = float(np.linalg.norm(correction))
                    if correction_norm <= 1.0e-9:
                        continue
                    inward = -correction / correction_norm
                    closest = corrected_point - target_depth * inward

                if _apply_edge_sample_correction(
                    edge_idx,
                    alpha,
                    point,
                    closest,
                    inward,
                    target_depth,
                ):
                    corrected_edge_samples += 1

            for edge_idx in sorted(corrected_edge_indices):
                if edge_idx < 0 or edge_idx + 1 >= centers_mm.shape[0]:
                    continue
                p0 = centers_mm[edge_idx, :3]
                p1 = centers_mm[edge_idx + 1, :3]
                for alpha in self._surface_edge_sample_alphas(edge_idx, True):
                    point = (1.0 - alpha) * p0 + alpha * p1
                    nominal_s = (1.0 - alpha) * self._node_s(edge_idx) + alpha * self._node_s(edge_idx + 1)
                    if not self._strict_native_surface_guard_eligible(point, nominal_s):
                        continue
                    sample = self._point_surface_clearance_sample(
                        point,
                        nominal_s,
                        exact_projection=True,
                    )
                    clearance = float('inf')
                    closest = np.zeros(3, dtype=float)
                    inward = np.zeros(3, dtype=float)
                    if sample is not None:
                        clearance, closest, inward = sample
                    surrogate_clearance = self._point_wall_clearance(point, nominal_s, exact_projection=True)
                    use_surrogate_recovery = bool(
                        prefer_surrogate_recovery
                        and np.isfinite(surrogate_clearance)
                        and ((not np.isfinite(clearance)) or clearance >= surrogate_clearance + surrogate_disagreement_margin_mm)
                    )
                    if use_surrogate_recovery:
                        clearance = float(surrogate_clearance)
                    if (not np.isfinite(clearance)) and sample is None:
                        continue
                    if np.isfinite(clearance):
                        min_clearance = min(min_clearance, float(clearance))
                    if clearance >= guard_enter_clearance_mm:
                        continue
                    if use_surrogate_recovery:
                        corrected_point = self._constrain_point(point, nominal_s, exact_projection=True)
                        correction = corrected_point - np.asarray(point, dtype=float).reshape(3)
                        correction_norm = float(np.linalg.norm(correction))
                        if correction_norm <= 1.0e-9:
                            continue
                        inward = -correction / correction_norm
                        closest = corrected_point - second_pass_target_depth * inward
                    if _apply_edge_sample_correction(
                        edge_idx,
                        alpha,
                        point,
                        closest,
                        inward,
                        second_pass_target_depth,
                    ):
                        corrected_edge_samples += 1

        local_contact_speed_cap_mm_s = max(20.0, 2.0 * float(self.push_force_target_speed_mm_s))
        locally_clipped = max(
            _clip_selected_linear_speed(rod_vel, corrected_node_indices, local_contact_speed_cap_mm_s),
            _clip_selected_linear_speed(rod_free_vel, corrected_node_indices, local_contact_speed_cap_mm_s),
        )

        clipped = max(_clip_linear_speed(rod_vel), _clip_linear_speed(rod_free_vel))

        if corrected <= 0 and corrected_edge_samples <= 0 and clipped <= 0 and locally_clipped <= 0 and hold_damped <= 0:
            return 0, 0

        with _writeable(self._rod_state_pos) as out_pos:
            out_pos[:] = rod_pos
        if self._rod_state_vel is not None and rod_vel is not None:
            with _writeable(self._rod_state_vel) as out_vel:
                out_vel[:] = rod_vel
        if self._rod_state_free_pos is not None and rod_free_pos is not None:
            with _writeable(self._rod_state_free_pos) as out_free_pos:
                out_free_pos[:] = rod_free_pos
        if self._rod_state_free_vel is not None and rod_free_vel is not None:
            with _writeable(self._rod_state_free_vel) as out_free_vel:
                out_free_vel[:] = rod_free_vel
        self._invalidate_geometry_cache()
        self._invalidate_surface_probe_cache()
        self._native_strict_min_lumen_clearance_cache_step = -1
        self._native_strict_barrier_active_cache_step = -1

        if self._should_log_native_strict_guard(
            corrected_nodes=corrected,
            corrected_edge_samples=corrected_edge_samples,
            clipped_nodes=(clipped + locally_clipped + hold_damped),
            min_clearance_mm=min_clearance,
        ):
            min_clearance_value = min_clearance if np.isfinite(min_clearance) else float('nan')
            print(
                f'[INFO] [elasticrod-strict-guard] step={self.step_count} '
                f'correctedNodes={corrected} correctedEdgeSamples={corrected_edge_samples} clippedVelNodes={clipped + locally_clipped + hold_damped} '
                f'minClearance={min_clearance_value:.4f} mm maxSpeedBefore={max_speed_before:.2f} mm/s'
            )
            self._native_last_strict_guard_step = self.step_count

        return corrected + corrected_edge_samples, clipped + locally_clipped

    def _update_wall_contact_state(self) -> float:
        clearance = self._head_wall_clearance()
        if self.is_native_strict:
            physical_clearance = self._native_strict_physical_contact_clearance_mm()
            if np.isfinite(physical_clearance):
                clearance = float(physical_clearance)
            enter_mm = float(ELASTICROD_STRICT_TIP_WALL_CONTACT_ENTER_MM)
            exit_mm = float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM)
        else:
            surface_clearance = self._head_surface_clearance()
            if np.isfinite(surface_clearance):
                clearance = min(clearance, surface_clearance)
            enter_mm = float(TIP_WALL_CONTACT_ENTER_MM)
            exit_mm = float(TIP_WALL_CONTACT_EXIT_MM)
        prev = self.wall_contact_active
        if self.wall_contact_active:
            if self.is_native_strict and clearance > exit_mm:
                self._strict_wall_contact_release_counter += 1
                release_hold_steps = max(int(ELASTICROD_STRICT_TIP_WALL_CONTACT_RELEASE_HOLD_STEPS), 0)
                strict_wall_gap_mm = float(self._native_strict_actual_wall_gap_mm())
                strict_barrier_nodes = self._native_strict_barrier_active_node_count()
                fast_release = bool(
                    strict_barrier_nodes <= int(ELASTICROD_STRICT_GUI_GUIDED_CONTACT_MAX_BARRIER_NODES)
                    and np.isfinite(clearance)
                    and clearance >= max(exit_mm + 0.02, 0.32)
                    and (not np.isfinite(strict_wall_gap_mm) or strict_wall_gap_mm >= max(exit_mm, 0.30))
                )
                if fast_release:
                    release_hold_steps = min(release_hold_steps, 1)
                self.wall_contact_active = self._strict_wall_contact_release_counter <= release_hold_steps
            elif self.is_native_backend and (not self.is_native_strict) and clearance > exit_mm:
                self._strict_wall_contact_release_counter += 1
                release_hold_steps = max(int(ELASTICROD_SAFE_TIP_WALL_CONTACT_RELEASE_HOLD_STEPS), 0)
                self.wall_contact_active = self._strict_wall_contact_release_counter <= release_hold_steps
            else:
                self._strict_wall_contact_release_counter = 0
                self.wall_contact_active = clearance <= exit_mm
        else:
            self._strict_wall_contact_release_counter = 0
            self.wall_contact_active = clearance <= enter_mm
        self.wall_contact_clearance_mm = clearance
        if prev != self.wall_contact_active:
            state = 'entered' if self.wall_contact_active else 'released'
            if self.is_native_strict:
                if self.wall_contact_active:
                    self._strict_wall_contact_enter_step = self.step_count
                    self._strict_wall_contact_release_step = None
                else:
                    self._strict_wall_contact_enter_step = None
                    self._strict_wall_contact_release_step = self.step_count
            message = f'[INFO] Tip wall contact {state}: clearance={clearance:.4f} mm'
            if self.is_native_backend:
                if self.wall_contact_active and self._native_first_contact_step is None:
                    self._native_first_contact_step = self.step_count
                if not self.wall_contact_active:
                    self._native_stall_logged = False
                tip_pos, _ = self._tip_pose()
                message += (
                    f', step={self.step_count}, time={self.sim_time_s:.4f} s, '
                    f'tip={np.round(tip_pos, 3).tolist()}, commandedPush={self.commanded_push_mm:.3f} mm'
                )
            print(message)
        return clearance

    def _update_push_force_calibration(self, dt: float) -> None:
        """
        闁规亽鍔忕换妯侯啅閼碱剛鐥呴柡鈧憴鍕亣闁诡厽甯掗悾楣冨礉濞戞瑥浠橀柛鎺曨啇缁辨繈宕堕悩鎰佸妰閺夆晜鐟╅崳鐑藉矗椤忓嫭韬悹褔鏀遍鐐烘儎鐎涙﹩鍞介柛瀣煯缁旀潙鈻庨埄鍐ｅ亾瑜忛埞鏍ㄦ姜閼恒儳鍨奸悗瑙勭啲缁?        闁烩晩鍠楅悥锝夊及椤栨稑惟閻忓繐澧介顒勫箒閹烘垹鏆伴柟鎭掑妼婵繒鎷崘銊ョ厒闁炽儲绮岄柦鈺呭锤閸モ晛顔?1 mm/s闁炽儲绺块埀?
        濞戞挴鍋撻柡鍐跨畱閸ゎ參鎮抽幏灞藉灡濠㈤€涢檷閳ь兛鐒﹀Σ鎴﹀及閹规劗顦伴柛姘灥濞叉牠鏌呴埀顒勬晬鐏炴儳鐏楅柤鏉挎噹閸戯紕绱掕箛鏇ㄧ€茬€殿喒鍋撻柡宥呮搐閻ｉ箖鎯勭€涙﹩鍞介柨娑樿嫰濮樸劎绮╃€ｎ亜鐓濋梺澶哥閻ｉ箖寮介崶鈹偤骞掗妸銉ヮ潝闁?        闁告艾娴烽悽濠氬矗椤忓嫬甯掗悹浣侯焾濠€?`100%` 濞?`闁告垵绻戠敮鐟靶掗弬鍓т紣` 濞戞柨顑夊Λ鍧楀礆閸ャ劌搴婇柨娑樼焸娴尖晠宕楀鍡椢╅柟鎭掑妼婵繒鎼炬繝鍛閻℃帒锕ら妵鍥Υ?        """
        if self.push_force_calibrated:
            return

        speed = max(self.filtered_tip_forward_speed_mm_s, 0.0)
        unsafe = self.wall_contact_active or self.filtered_tip_forward_speed_mm_s <= PUSH_FORCE_CALIBRATION_LOCK_NEG_SPEED_MM_S
        out_of_region = self.drive_push_mm >= PUSH_FORCE_CALIBRATION_REGION_MM
        timed_out = (
            self.push_force_calibration_time >= PUSH_FORCE_CALIBRATION_TIME_S
            and speed >= 0.5 * self.push_force_target_speed_mm_s
        )
        if unsafe or out_of_region or timed_out:
            self.push_force_calibrated = True
            if not self._push_calibration_final_logged:
                print(
                    f'[INFO] Push force calibration locked: nominalTotal={self.nominal_push_force_total:.5f}, '
                    f'tipSpeed={self.filtered_tip_forward_speed_mm_s:.3f} mm/s, wallContact={self.wall_contact_active}'
                )
                self._push_calibration_final_logged = True
            return

        if not self._push_calibration_started:
            print(
                f'[INFO] Push force calibration started: targetSpeed={self.push_force_target_speed_mm_s:.3f} mm/s, '
                f'region={PUSH_FORCE_CALIBRATION_REGION_MM:.1f} mm, initialNominal={self.nominal_push_force_total:.5f}'
            )
            self._push_calibration_started = True

        self.push_force_calibration_time += float(dt)
        if speed < 0.25 * self.push_force_target_speed_mm_s:
            self.nominal_push_force_total += PUSH_FORCE_CALIBRATION_RAMP_PER_S * float(dt)
        else:
            desired_force = self.nominal_push_force_total * self.push_force_target_speed_mm_s / max(speed, 0.25 * self.push_force_target_speed_mm_s)
            desired_force = float(np.clip(desired_force, 0.6 * self.nominal_push_force_total, 1.6 * self.nominal_push_force_total))
            alpha = float(np.clip(PUSH_FORCE_CALIBRATION_ALPHA, 0.0, 1.0))
            self.nominal_push_force_total = (1.0 - alpha) * self.nominal_push_force_total + alpha * desired_force
        self.nominal_push_force_total = float(np.clip(self.nominal_push_force_total, PUSH_FORCE_MIN_TOTAL, PUSH_FORCE_MAX_TOTAL))

        if DEBUG_PRINT_EVERY > 0 and self.step_count % DEBUG_PRINT_EVERY == 0:
            print(
                f'[push-calib] t={self.push_force_calibration_time:.2f}s nominalTotal={self.nominal_push_force_total:.5f} '
                f'tipSpeed={self.filtered_tip_forward_speed_mm_s:.3f} mm/s drivePush={self.drive_push_mm:.3f} mm'
            )

    def _native_realtime_target_band(self) -> str:
        if not self.is_native_realtime:
            return 'quality'

        current = str(self._native_runtime_band or 'free')
        clearance = float(self.wall_contact_clearance_mm)
        steering = float(abs(self.steering_angle_deg))
        wall_contact = bool(self.wall_contact_active)
        hard_wall_contact = self._native_strict_hard_wall_contact() if self.is_native_strict else False
        barrier_nodes = self._native_strict_barrier_active_node_count() if self.is_native_strict else 0
        barrier_contact_gate = self._native_strict_barrier_contact_gate() if self.is_native_strict else 0.0
        barrier_active = self.is_native_strict and (barrier_nodes > 0 or barrier_contact_gate > 0.0)
        preview_turn_deg = self._native_strict_upcoming_turn_deg() if self.is_native_strict else steering
        bend_severity = self._native_strict_bend_severity() if self.is_native_strict else 0.0
        progress_gate_mm = float(max(
            ELASTICROD_STRICT_RUNTIME_PROGRESS_GATE_MM,
            ELASTICROD_STRICT_INITIAL_STRAIGHT_PUSH_MM,
            ELASTICROD_STRICT_MAGNETIC_RELEASE_SPAN_MM,
        ))
        strict_progress_mm = float(max(
            self.commanded_push_mm,
            self.tip_progress_raw_mm,
            self.tip_progress_mm,
            0.0,
        ))
        steering_preview_enabled = (
            (not self.is_native_strict)
            or wall_contact
            or barrier_active
            or strict_progress_mm >= progress_gate_mm
        )
        def choose_band(band: str, reason: str) -> str:
            self._native_runtime_band_reason = reason
            return band

        steering_metric = preview_turn_deg if self.is_native_strict else steering
        steering_hold_metric = max(float(steering_metric), float(steering))
        strict_clearance = (
            self._native_strict_min_lumen_clearance_mm()
            if self.is_native_strict
            else float('inf')
        )
        strict_wall_gap = (
            self._native_strict_actual_wall_gap_mm()
            if self.is_native_strict
            else float('inf')
        )
        strict_physical_gap = (
            self._native_strict_physical_contact_clearance_mm()
            if self.is_native_strict
            else float('inf')
        )
        # The native lumen barrier is now allowed to protect the rod away from the
        # exact wall. Do not immediately drop the whole runtime into the slow
        # contact band on the first mild barrier activation; reserve `contact`
        # for genuine near-wall / exact-contact states.
        strict_contact_threshold_mm = min(
            max(float(ELASTICROD_REALTIME_CLEARANCE_CONTACT_MM), 0.18),
            0.25,
        )
        strict_transition_threshold_mm = max(float(ELASTICROD_REALTIME_CLEARANCE_TRANSITION_MM), 0.45)
        strict_contact_active = bool(
            self.is_native_strict
            and barrier_active
            and np.isfinite(strict_clearance)
            and (
                (
                    strict_clearance <= strict_contact_threshold_mm
                    and (
                        wall_contact
                        or barrier_contact_gate >= 1.0
                        or (np.isfinite(strict_wall_gap) and strict_wall_gap <= max(strict_contact_threshold_mm, 0.18))
                        or (np.isfinite(strict_physical_gap) and strict_physical_gap <= max(strict_contact_threshold_mm, 0.18))
                    )
                )
                or (
                    barrier_nodes >= max(2 * int(ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT), 8)
                    and barrier_contact_gate > 0.0
                )
            )
        )
        strict_transition_active = bool(
            self.is_native_strict
            and barrier_active
            and np.isfinite(strict_clearance)
            and (
                (
                    strict_clearance <= strict_transition_threshold_mm
                    and (
                        wall_contact
                        or barrier_contact_gate > 0.0
                        or (np.isfinite(strict_wall_gap) and strict_wall_gap <= max(strict_transition_threshold_mm, 0.30))
                        or (np.isfinite(strict_physical_gap) and strict_physical_gap <= max(strict_transition_threshold_mm, 0.30))
                    )
                )
                or (
                    barrier_nodes >= max(int(ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT), 4)
                    and barrier_contact_gate > 0.35
                )
            )
        )
        steering_transition_clearance = max(0.80, 2.0 * float(ELASTICROD_REALTIME_CLEARANCE_TRANSITION_MM))
        steering_contact_clearance = max(0.60, 2.0 * float(ELASTICROD_REALTIME_CLEARANCE_CONTACT_MM))
        steering_transition_active = (
            steering_preview_enabled
            and (
                steering_metric >= ELASTICROD_REALTIME_STEERING_ENTER_DEG
                or bend_severity >= float(ELASTICROD_STRICT_RUNTIME_BEND_SEVERITY_TRANSITION)
            )
            and clearance <= steering_transition_clearance
        )
        steering_contact_active = (
            self.is_native_strict
            and steering_preview_enabled
            and (
                steering_metric >= ELASTICROD_REALTIME_STEERING_CONTACT_DEG
                or bend_severity >= float(ELASTICROD_STRICT_RUNTIME_BEND_SEVERITY_CONTACT)
            )
            and clearance <= steering_contact_clearance
        )
        steering_precontact_active = (
            self.is_native_strict
            and steering_preview_enabled
            and (
                steering_metric >= max(ELASTICROD_REALTIME_STEERING_CONTACT_DEG - 5.0, ELASTICROD_REALTIME_STEERING_ENTER_DEG)
                or bend_severity >= float(ELASTICROD_STRICT_RUNTIME_BEND_SEVERITY_TRANSITION)
            )
            and clearance <= max(0.90, steering_transition_clearance)
        )

        support_stretch = self._native_debug_array(self._native_debug_stretch)
        proximal_window = np.abs(support_stretch[: self.native_support_count]) if self.native_support_count > 0 else np.zeros(0, dtype=float)
        proximal_stretch = float(np.max(proximal_window)) if proximal_window.size else 0.0
        max_stretch = float(np.max(np.abs(support_stretch))) if support_stretch.size else 0.0
        max_head_stretch = self._native_strict_max_head_stretch()
        max_twist = self._native_debug_max_abs(self._native_debug_twist)
        if not self.is_native_strict:
            safe_contact_clearance_mm = clearance if np.isfinite(clearance) else float('inf')
            if np.isfinite(self.surface_wall_contact_clearance_mm):
                safe_contact_clearance_mm = min(safe_contact_clearance_mm, float(self.surface_wall_contact_clearance_mm))
            hard_wall_contact = bool(
                wall_contact
                and (
                    (
                        np.isfinite(safe_contact_clearance_mm)
                        and safe_contact_clearance_mm <= 0.10
                        and barrier_nodes >= 2
                    )
                    or max_head_stretch >= 8.0e-3
                    or max_stretch >= 6.5e-2
                    or max_twist >= 9.0e-2
                )
            )
            safe_barrier_contact = bool(
                barrier_active
                and np.isfinite(safe_contact_clearance_mm)
                and safe_contact_clearance_mm <= max(float(ELASTICROD_REALTIME_CLEARANCE_CONTACT_MM) + 0.05, 0.32)
                and (
                    barrier_nodes >= max(int(ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT), 4)
                    or max_head_stretch >= 4.0e-3
                    or max_twist >= 6.0e-2
                )
            )
        else:
            safe_barrier_contact = False
        proximal_abnormal_edge = self._native_debug_int(self._native_debug_abnormal_edge_index, default=-1)
        proximal_axial_err = self._native_debug_scalar(self._native_debug_max_axial_boundary_error)
        _, head_stretch_hard_limit = self._native_strict_head_stretch_limits()
        proximal_anomaly = (
            (0 <= proximal_abnormal_edge < self.native_support_count)
            or proximal_stretch >= 0.12
            or proximal_axial_err >= max(0.40 * self.rest_spacing_mm, 0.50)
            or max_stretch >= 0.20
            or max_head_stretch >= head_stretch_hard_limit
            or max_twist >= 0.25
        )
        if proximal_anomaly:
            return choose_band('contact', 'proximalAnomaly')
        if (
            self.is_native_strict
            and self._native_virtual_sheath_paused
            and self._native_virtual_sheath_pause_reason in {'clearance', 'headStretch', 'kink'}
        ):
            return choose_band('contact', f'strictHold:{self._native_virtual_sheath_pause_reason}')

        head_stretch_soft_limit, _ = self._native_strict_head_stretch_limits()
        recent_contact_release_steps = (
            int(self.step_count - self._strict_wall_contact_release_step)
            if (
                self.is_native_strict
                and self._strict_wall_contact_release_step is not None
                and self.step_count >= self._strict_wall_contact_release_step
            )
            else 10**9
        )
        recent_contact_release = bool(
            self.is_native_strict
            and recent_contact_release_steps <= int(max(ELASTICROD_STRICT_RUNTIME_RELEASE_TRANSITION_HOLD_STEPS, 0))
        )
        recent_contact_damping = bool(
            self.is_native_strict
            and recent_contact_release_steps <= int(max(ELASTICROD_STRICT_RUNTIME_RELEASE_CONTACT_HOLD_STEPS, 0))
        )
        kink_preview_clearance_limit = max(steering_transition_clearance, 0.95)
        if (
            self.is_native_strict
            and (
                steering_metric >= max(float(ELASTICROD_REALTIME_STEERING_ENTER_DEG) - 10.0, 25.0)
                or bend_severity >= max(float(ELASTICROD_STRICT_RUNTIME_BEND_SEVERITY_TRANSITION), 0.20)
                or recent_contact_release
            )
        ):
            kink_preview_clearance_limit = max(kink_preview_clearance_limit, 1.30)
        strict_kink_preview = bool(
            self.is_native_strict
            and (
                clearance <= kink_preview_clearance_limit
                or (recent_contact_release and clearance <= 1.45)
            )
            and (
                max_head_stretch >= max(0.60 * head_stretch_soft_limit, 0.014)
                or max_stretch >= 0.018
                or max_twist >= 0.08
            )
        )
        strict_bend_contact_hold = bool(
            self.is_native_strict
            and strict_progress_mm >= 3.0
            and np.isfinite(clearance)
            and clearance <= 1.20
            and (
                max_head_stretch >= 0.014
                or max_stretch >= 0.018
                or (
                    clearance <= 0.55
                    and (
                        steering_hold_metric >= 28.0
                        or bend_severity >= max(float(ELASTICROD_STRICT_RUNTIME_BEND_SEVERITY_TRANSITION), 0.25)
                    )
                )
            )
        )
        strict_near_wall_bend_preview = bool(
            self.is_native_strict
            and np.isfinite(clearance)
            and clearance <= max(float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM) + 0.14, 0.44)
            and (
                steering_hold_metric >= max(float(ELASTICROD_REALTIME_STEERING_ENTER_DEG) - 8.0, 27.0)
                or bend_severity >= max(float(ELASTICROD_STRICT_RUNTIME_BEND_SEVERITY_TRANSITION), 0.20)
                or max_head_stretch >= 0.010
                or max_stretch >= 0.014
            )
        )
        actual_wall_gap = (
            self._native_strict_actual_wall_gap_mm()
            if self.is_native_strict
            else clearance
        )
        light_wall_contact = bool(
            self.is_native_strict
            and wall_contact
            and (not strict_contact_active)
            and barrier_nodes <= int(ELASTICROD_STRICT_GUI_LIGHT_CONTACT_MAX_BARRIER_NODES)
            and np.isfinite(clearance)
            and clearance >= 0.22
            and (not np.isfinite(actual_wall_gap) or actual_wall_gap >= 0.22)
            and max(float(steering_metric), float(steering)) < max(float(ELASTICROD_REALTIME_STEERING_ENTER_DEG) + 5.0, 40.0)
            and bend_severity < max(float(ELASTICROD_STRICT_RUNTIME_BEND_SEVERITY_TRANSITION), 0.35)
            and max_head_stretch <= max(1.25 * head_stretch_soft_limit, 0.025)
            and max_stretch <= 0.02
        )
        guided_wall_follow_contact = bool(
            self.is_native_strict
            and wall_contact
            and (not light_wall_contact)
            and barrier_nodes <= int(ELASTICROD_STRICT_GUI_GUIDED_CONTACT_MAX_BARRIER_NODES)
            and np.isfinite(clearance)
            and clearance >= max(float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM) + 0.02, 0.32)
            and (not np.isfinite(actual_wall_gap) or actual_wall_gap >= max(float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM), 0.30))
            and max_head_stretch <= max(1.5 * head_stretch_soft_limit, 0.015)
            and max_stretch <= 0.02
        )
        transition_exit_clearance = ELASTICROD_REALTIME_CLEARANCE_TRANSITION_MM + 0.05

        if current == 'contact':
            if safe_barrier_contact:
                return choose_band('contact', 'safeBarrierContact')
            if hard_wall_contact and (not light_wall_contact):
                if guided_wall_follow_contact:
                    return choose_band('transition', 'guidedWallFollow')
                return choose_band('contact', 'wallContact')
            if wall_contact:
                return choose_band('transition', 'wallContactHold')
            if light_wall_contact:
                return choose_band('transition', 'lightWallContact')
            if strict_contact_active:
                return choose_band('contact', 'barrierContact')
            if strict_bend_contact_hold:
                return choose_band('contact', 'bendContactHold')
            if strict_near_wall_bend_preview:
                return choose_band('contact', 'nearWallBendPreview')
            if recent_contact_damping:
                return choose_band('contact', 'recentContactDamping')
            if strict_kink_preview:
                return choose_band('contact', 'headStretchPreview')
            if (
                steering_contact_active
                or strict_transition_active
                or steering_transition_active
                or steering_precontact_active
                or recent_contact_release
                or clearance <= transition_exit_clearance
            ):
                return choose_band('transition', 'contactReleasePreview' if not recent_contact_release else 'recentContactRelease')
            return choose_band('free', 'contactReleased')

        if current == 'transition':
            if safe_barrier_contact:
                return choose_band('contact', 'safeBarrierContact')
            if hard_wall_contact and (not light_wall_contact):
                if guided_wall_follow_contact:
                    return choose_band('transition', 'guidedWallFollow')
                return choose_band('contact', 'wallContact')
            if wall_contact:
                return choose_band('transition', 'wallContactHold')
            if light_wall_contact:
                return choose_band('transition', 'lightWallContact')
            if strict_contact_active:
                return choose_band('contact', 'barrierContact')
            if strict_bend_contact_hold:
                return choose_band('contact', 'bendContactHold')
            if strict_near_wall_bend_preview:
                return choose_band('contact', 'nearWallBendPreview')
            if recent_contact_damping:
                return choose_band('contact', 'recentContactDamping')
            if strict_kink_preview:
                return choose_band('contact', 'headStretchPreview')
            if (
                steering_contact_active
                or strict_transition_active
                or steering_transition_active
                or steering_precontact_active
                or recent_contact_release
                or clearance <= transition_exit_clearance
            ):
                return choose_band('transition', 'precontactPreview' if not recent_contact_release else 'recentContactRelease')
            return choose_band('free', 'transitionReleased')

        if safe_barrier_contact:
            return choose_band('contact', 'safeBarrierContact')
        if hard_wall_contact and (not light_wall_contact):
            if guided_wall_follow_contact:
                return choose_band('transition', 'guidedWallFollow')
            return choose_band('contact', 'wallContact')
        if wall_contact:
            return choose_band('transition', 'wallContactHold')
        if light_wall_contact:
            return choose_band('transition', 'lightWallContact')
        if strict_contact_active:
            return choose_band('contact', 'barrierContact')
        if strict_bend_contact_hold:
            return choose_band('contact', 'bendContactHold')
        if strict_near_wall_bend_preview:
            return choose_band('contact', 'nearWallBendPreview')
        if recent_contact_damping:
            return choose_band('contact', 'recentContactDamping')
        if strict_kink_preview:
            return choose_band('contact', 'headStretchPreview')
        if (
            steering_contact_active
            or strict_transition_active
            or steering_transition_active
            or steering_precontact_active
            or recent_contact_release
            or clearance <= ELASTICROD_REALTIME_CLEARANCE_TRANSITION_MM
        ):
            return choose_band('transition', 'precontactPreview' if not recent_contact_release else 'recentContactRelease')
        return choose_band('free', 'clear')

    def _native_realtime_band_settings(self, band: str) -> tuple[float, int, float, float]:
        if self.is_native_strict:
            # Strict headless diagnostics must stay on the validated solver-time
            # band. Reusing the looser GUI wall-clock band here made free-space
            # cases drift and eventually blow up even with zero push and zero
            # magnetic field. Only actual runSofa / GUI wall-clock launches
            # should use the GUI-tuned numbers.
            if self.use_native_gui_wallclock_control:
                if band == 'contact':
                    return (
                        float(ELASTICROD_GUI_DT_CONTACT_S),
                        int(ELASTICROD_GUI_SOLVER_MAX_ITER_CONTACT),
                        float(ELASTICROD_GUI_SOLVER_TOL_CONTACT),
                        float(ELASTICROD_REALTIME_SPEED_SCALE_CONTACT),
                    )
                if band == 'transition':
                    return (
                        float(ELASTICROD_GUI_DT_TRANSITION_S),
                        int(ELASTICROD_GUI_SOLVER_MAX_ITER_TRANSITION),
                        float(ELASTICROD_GUI_SOLVER_TOL_TRANSITION),
                        float(ELASTICROD_REALTIME_SPEED_SCALE_TRANSITION),
                    )
                return (
                    float(ELASTICROD_GUI_DT_FREE_S),
                    int(ELASTICROD_GUI_SOLVER_MAX_ITER_FREE),
                    float(ELASTICROD_GUI_SOLVER_TOL_FREE),
                    float(ELASTICROD_REALTIME_SPEED_SCALE_FREE),
                )
            if band == 'contact':
                return (
                    float(ELASTICROD_REALTIME_DT_CONTACT_S),
                    int(ELASTICROD_REALTIME_SOLVER_MAX_ITER_CONTACT),
                    float(ELASTICROD_REALTIME_SOLVER_TOL_CONTACT),
                    float(ELASTICROD_REALTIME_SPEED_SCALE_CONTACT),
                )
            if band == 'transition':
                return (
                    float(ELASTICROD_REALTIME_DT_TRANSITION_S),
                    int(ELASTICROD_REALTIME_SOLVER_MAX_ITER_TRANSITION),
                    float(ELASTICROD_REALTIME_SOLVER_TOL_TRANSITION),
                    float(ELASTICROD_REALTIME_SPEED_SCALE_TRANSITION),
                )
            return (
                float(ELASTICROD_REALTIME_DT_FREE_S),
                int(ELASTICROD_REALTIME_SOLVER_MAX_ITER_FREE),
                float(ELASTICROD_REALTIME_SOLVER_TOL_FREE),
                float(ELASTICROD_REALTIME_SPEED_SCALE_FREE),
            )
        if self.use_native_gui_wallclock_control:
            if band == 'contact':
                return (
                    float(ELASTICROD_GUI_DT_CONTACT_S),
                    int(ELASTICROD_GUI_SOLVER_MAX_ITER_CONTACT),
                    float(ELASTICROD_GUI_SOLVER_TOL_CONTACT),
                    float(ELASTICROD_REALTIME_SPEED_SCALE_CONTACT),
                )
            if band == 'transition':
                return (
                    float(ELASTICROD_GUI_DT_TRANSITION_S),
                    int(ELASTICROD_GUI_SOLVER_MAX_ITER_TRANSITION),
                    float(ELASTICROD_GUI_SOLVER_TOL_TRANSITION),
                    float(ELASTICROD_REALTIME_SPEED_SCALE_TRANSITION),
                )
            return (
                float(ELASTICROD_GUI_DT_FREE_S),
                int(ELASTICROD_GUI_SOLVER_MAX_ITER_FREE),
                float(ELASTICROD_GUI_SOLVER_TOL_FREE),
                float(ELASTICROD_REALTIME_SPEED_SCALE_FREE),
            )
        if band == 'contact':
            return (
                float(ELASTICROD_REALTIME_DT_CONTACT_S),
                int(ELASTICROD_REALTIME_SOLVER_MAX_ITER_CONTACT),
                float(ELASTICROD_REALTIME_SOLVER_TOL_CONTACT),
                float(ELASTICROD_REALTIME_SPEED_SCALE_CONTACT),
            )
        if band == 'transition':
            return (
                float(ELASTICROD_REALTIME_DT_TRANSITION_S),
                int(ELASTICROD_REALTIME_SOLVER_MAX_ITER_TRANSITION),
                float(ELASTICROD_REALTIME_SOLVER_TOL_TRANSITION),
                float(ELASTICROD_REALTIME_SPEED_SCALE_TRANSITION),
            )
        return (
            float(ELASTICROD_REALTIME_DT_FREE_S),
            int(ELASTICROD_REALTIME_SOLVER_MAX_ITER_FREE),
            float(ELASTICROD_REALTIME_SOLVER_TOL_FREE),
            float(ELASTICROD_REALTIME_SPEED_SCALE_FREE),
        )

    def _apply_native_runtime_settings(self) -> None:
        if not self.is_native_realtime:
            return

        root = self.root_node
        if root is None:
            try:
                root = self.getContext().getRootContext()
            except Exception:
                root = None
        if root is not None:
            try:
                root.dt = float(self._native_runtime_dt_s)
            except Exception:
                pass
            try:
                dt_data = root.findData('dt')
                if dt_data is not None:
                    dt_data.value = float(self._native_runtime_dt_s)
            except Exception:
                pass
            if self.constraint_solver is None:
                try:
                    self.constraint_solver = root.getObject('constraintSolver')
                except Exception:
                    self.constraint_solver = None

        if self.constraint_solver is not None:
            try:
                max_iter_data = self.constraint_solver.findData('maxIterations')
                if max_iter_data is not None:
                    max_iter_data.value = int(self._native_runtime_solver_max_iter)
            except Exception:
                pass
            try:
                tol_data = self.constraint_solver.findData('tolerance')
                if tol_data is not None:
                    tol_data.value = float(self._native_runtime_solver_tolerance)
            except Exception:
                pass

        for obj in (self.rod_model, self.native_mass):
            if obj is None:
                continue
            try:
                dt_data = obj.findData('dt')
                if dt_data is not None:
                    dt_data.value = float(self._native_runtime_dt_s)
            except Exception:
                pass

    def _update_native_runtime_profile(self) -> None:
        if not self.is_native_realtime:
            return

        band = self._native_realtime_target_band()
        dt_s, max_iter, tol, speed_scale = self._native_realtime_band_settings(band)
        band_changed = band != self._native_runtime_band
        settings_changed = (
            abs(float(self._native_runtime_dt_s) - float(dt_s)) > 1.0e-12
            or int(self._native_runtime_solver_max_iter) != int(max_iter)
            or abs(float(self._native_runtime_solver_tolerance) - float(tol)) > 1.0e-15
        )

        self._native_runtime_band = band
        self._native_runtime_dt_s = float(dt_s)
        self._native_runtime_solver_max_iter = int(max_iter)
        self._native_runtime_solver_tolerance = float(tol)
        self._native_runtime_speed_scale = float(speed_scale)

        if band_changed or settings_changed or (not self._native_runtime_settings_applied):
            self._apply_native_runtime_settings()
            self._native_runtime_settings_applied = True
        if band_changed:
            print(
                f'[INFO] [elasticrod-runtime] band={band} step={self.step_count} '
                f'dt={self._native_runtime_dt_s:.6f}s solver={self._native_runtime_solver_max_iter}/{self._native_runtime_solver_tolerance:.1e} '
                f'speedScale={self._native_runtime_speed_scale:.2f} '
                f'reason={self._native_runtime_band_reason} '
                f'wallContact={self.wall_contact_active} clearance={self.wall_contact_clearance_mm:.4f} mm '
                f'steering={self.steering_angle_deg:.2f} deg'
            )

    def _current_push_force_scale(self) -> float:
        if self.is_native_strict:
            if bool(ELASTICROD_STRICT_ALWAYS_PUSH_FORWARD):
                self._push_scale_reason = f'band={self._native_runtime_band},alwaysPush'
                return 1.0
            scale = 1.0
            reasons = [f'band={self._native_runtime_band}']
            hard_wall_contact = self._native_strict_hard_wall_contact()
            bend_severity = float(np.clip(self._native_debug_scalar(self._debug_bend_severity, default=0.0), 0.0, 1.0))
            wall_gap_mm = self._native_strict_actual_wall_gap_mm()
            physical_gap_mm = self._native_strict_physical_contact_clearance_mm()
            barrier_nodes = self._native_strict_barrier_active_node_count()
            barrier_contact_gate = self._native_strict_barrier_contact_gate()
            head_stretch = self._native_strict_max_head_stretch()
            head_stretch_soft_limit, head_stretch_hard_limit = self._native_strict_head_stretch_limits()
            global_stretch_soft_limit, global_stretch_hard_limit = self._native_strict_global_stretch_limits()
            support_stretch = self._native_debug_array(self._native_debug_stretch)
            max_stretch = float(np.max(np.abs(support_stretch))) if support_stretch.size else 0.0
            min_clearance_mm = self._native_strict_min_lumen_clearance_mm()
            guided_wall_follow_contact = bool(
                self.wall_contact_active
                and barrier_nodes <= int(ELASTICROD_STRICT_GUI_GUIDED_CONTACT_MAX_BARRIER_NODES)
                and np.isfinite(self.wall_contact_clearance_mm)
                and float(self.wall_contact_clearance_mm) >= max(float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM) + 0.02, 0.32)
                and (not np.isfinite(wall_gap_mm) or wall_gap_mm >= max(float(ELASTICROD_STRICT_TIP_WALL_CONTACT_EXIT_MM), 0.30))
                and max_stretch <= 0.02
                and head_stretch <= max(1.5 * head_stretch_soft_limit, 0.015)
            )
            clearance_gate = 0.0
            if np.isfinite(wall_gap_mm):
                clearance_gate = float(np.clip((0.22 - wall_gap_mm) / 0.17, 0.0, 1.0))
            barrier_force_gate = 0.0
            if barrier_nodes > 0 and barrier_contact_gate > 0.0 and np.isfinite(min_clearance_mm):
                barrier_count_gate = float(
                    np.clip(
                        (barrier_nodes - max(int(ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT) - 2, 2)) / 3.0,
                        0.0,
                        1.0,
                    )
                )
                barrier_clearance_gate = float(np.clip((0.18 - min_clearance_mm) / 0.12, 0.0, 1.0))
                barrier_force_gate = barrier_contact_gate * min(barrier_count_gate, barrier_clearance_gate)
            wall_follow_contact = np.isfinite(wall_gap_mm) and wall_gap_mm > 0.0
            contact_gate = max(float(hard_wall_contact), clearance_gate, barrier_force_gate)
            if contact_gate > 0.0:
                contact_scale = (
                    (1.0 - bend_severity) * float(ELASTICROD_STRICT_PUSH_SCALE_STRAIGHT_CONTACT)
                    + bend_severity * float(ELASTICROD_STRICT_PUSH_SCALE_BEND_CONTACT)
                )
                contact_scale = float(np.clip(contact_scale, 0.0, 1.0))
                contact_metric_candidates = []
                if np.isfinite(min_clearance_mm):
                    contact_metric_candidates.append(float(min_clearance_mm))
                if np.isfinite(wall_gap_mm):
                    contact_metric_candidates.append(float(wall_gap_mm))
                if hard_wall_contact and np.isfinite(self.wall_contact_clearance_mm):
                    contact_metric_candidates.append(float(self.wall_contact_clearance_mm))
                contact_metric_mm = min(contact_metric_candidates) if contact_metric_candidates else float('inf')
                if barrier_force_gate > 0.0 and np.isfinite(contact_metric_mm):
                    barrier_hard_gate = float(np.clip((0.12 - contact_metric_mm) / 0.08, 0.0, 1.0))
                    barrier_contact_scale = (
                        (1.0 - barrier_hard_gate) * contact_scale
                        + barrier_hard_gate * 0.52
                    )
                    contact_scale = float(
                        (1.0 - barrier_force_gate) * contact_scale
                        + barrier_force_gate * barrier_contact_scale
                    )
                    reasons.append(f'barrierGate={barrier_force_gate:.3f}')
                if hard_wall_contact and np.isfinite(self.wall_contact_clearance_mm):
                    nominal_contact_scale = float(contact_scale)
                    if guided_wall_follow_contact:
                        guided_contact_scale = max(
                            nominal_contact_scale,
                            float(max(
                                ELASTICROD_STRICT_GUIDED_CONTACT_PUSH_SCALE,
                                ELASTICROD_STRICT_LIGHT_CONTACT_PUSH_SCALE,
                            )),
                        )
                        contact_scale = guided_contact_scale
                        reasons.append('guidedFollow')
                    elif wall_follow_contact:
                        light_follow_gate = float(np.clip((0.10 - contact_metric_mm) / 0.10, 0.0, 1.0))
                        light_contact_scale = max(0.72, nominal_contact_scale)
                        contact_scale = (
                            (1.0 - light_follow_gate) * nominal_contact_scale
                            + light_follow_gate * light_contact_scale
                        )
                        reasons.append('wallFollow')
                    else:
                        physical_clearance_mm = float(self._native_strict_physical_contact_clearance_mm())
                        if np.isfinite(physical_clearance_mm):
                            contact_metric_mm = min(contact_metric_mm, physical_clearance_mm)
                        hard_contact_gate = float(np.clip((0.20 - contact_metric_mm) / 0.12, 0.0, 1.0))
                        contact_scale = (
                            (1.0 - hard_contact_gate) * nominal_contact_scale
                            + hard_contact_gate * 0.60
                        )
                        deep_penetration_gate = 0.0
                        if np.isfinite(wall_gap_mm):
                            deep_penetration_gate = float(np.clip(((-float(wall_gap_mm)) - 0.20) / 0.70, 0.0, 1.0))
                        if deep_penetration_gate > 0.0:
                            deep_contact_scale = 0.10
                            contact_scale = (
                                (1.0 - deep_penetration_gate) * contact_scale
                                + deep_penetration_gate * deep_contact_scale
                            )
                            reasons.append(f'deepContact={deep_penetration_gate:.3f}')
                        stalled_tip_speed = max(float(self.filtered_tip_forward_speed_mm_s), 0.0)
                        stall_gate = float(np.clip((0.80 - stalled_tip_speed) / 0.80, 0.0, 1.0))
                        penetration_gate = float(np.clip((0.06 - contact_metric_mm) / 0.10, 0.0, 1.0))
                        if penetration_gate > 0.0 and stall_gate > 0.0:
                            hard_stall_scale = 0.35
                            contact_scale = (
                                (1.0 - penetration_gate * stall_gate) * contact_scale
                                + penetration_gate * stall_gate * hard_stall_scale
                            )
                            reasons.append(f'stallGate={penetration_gate * stall_gate:.3f}')
                if hard_wall_contact:
                    scale = min(scale, contact_scale)
                    reasons.append(f'contactGate={contact_gate:.3f}')
                else:
                    if barrier_contact_gate <= 0.0:
                        contact_scale = max(contact_scale, float(ELASTICROD_STRICT_LIGHT_CONTACT_PUSH_SCALE))
                    precontact_gate = max(clearance_gate, barrier_force_gate)
                    scale = min(
                        scale,
                        float((1.0 - precontact_gate) * scale + precontact_gate * contact_scale),
                    )
                    reasons.append(f'precontactGate={precontact_gate:.3f}')
            else:
                reasons.append('precontact')
            kink_gate = 0.0
            near_wall_for_kink = bool(
                self.wall_contact_active
                or (np.isfinite(physical_gap_mm) and physical_gap_mm <= 0.35)
                or (np.isfinite(min_clearance_mm) and min_clearance_mm <= 0.18)
            )
            if barrier_nodes > 0 and near_wall_for_kink:
                if head_stretch_soft_limit > 1.0e-9:
                    stretch_gate = float(
                        np.clip(
                            (head_stretch - head_stretch_soft_limit)
                            / max(head_stretch_hard_limit - head_stretch_soft_limit, 1.0e-6),
                            0.0,
                            1.0,
                        )
                    )
                else:
                    stretch_gate = 0.0
                global_stretch_gate = float(
                    np.clip(
                        (max_stretch - global_stretch_soft_limit)
                        / max(global_stretch_hard_limit - global_stretch_soft_limit, 1.0e-6),
                        0.0,
                        1.0,
                    )
                )
                clearance_kink_gate = 0.0
                if np.isfinite(min_clearance_mm):
                    clearance_kink_gate = float(np.clip((0.18 - min_clearance_mm) / 0.12, 0.0, 1.0))
                barrier_gate = float(
                    np.clip(
                        (barrier_nodes - max(int(ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT) - 2, 2)) / 3.0,
                        0.0,
                        1.0,
                    )
                )
                kink_gate = max(
                    min(stretch_gate, max(clearance_kink_gate, barrier_gate)),
                    min(global_stretch_gate, max(clearance_kink_gate, barrier_gate)),
                )
            if kink_gate > 0.0:
                scale = min(scale, float((1.0 - kink_gate) * scale + kink_gate * 0.55))
                reasons.append(f'antiKink={kink_gate:.3f}')
            self._push_scale_reason = ','.join(reasons)
            return scale
        if self.is_native_realtime:
            self._push_scale_reason = f'band={self._native_runtime_band}'
            return float(self._native_runtime_speed_scale)
        if (not self.is_native_backend) and (not self.use_kinematic_beam_insertion):
            self._push_scale_reason = 'beamPhysical'
            return 1.0
        scale = 1.0
        reasons = []
        if self.use_kinematic_beam_insertion and self.beam_compression_active:
            scale = min(scale, PUSH_FORCE_REDUCED_SCALE_ON_COMPRESSION)
            reasons.append('beamCompression')
        if (not self.is_native_backend) and (not self.use_kinematic_beam_insertion) and self.beam_stall_active:
            scale = min(scale, PUSH_FORCE_REDUCED_SCALE_ON_STALL)
            reasons.append('beamStall')
        if self.navigation_mode == 2 and self.wall_contact_active and (not self.use_kinematic_beam_insertion):
            scale = min(scale, PUSH_FORCE_REDUCED_SCALE_ON_WALL)
            reasons.append('wallContact')
        if self.is_native_backend and self.navigation_mode == 2 and self.steering_misaligned:
            scale = min(scale, PUSH_FORCE_REDUCED_SCALE_ON_STEERING)
            reasons.append('steering')
        self._push_scale_reason = ','.join(reasons) if reasons else 'nominal'
        return scale

    def _update_smoothed_push_scale(self, dt: float) -> None:
        target_scale = self._current_push_force_scale()
        if self.is_native_realtime and target_scale <= self.push_force_scale:
            self.push_force_scale = target_scale
            return
        tau = PUSH_FORCE_SCALE_DROP_TIME_S if target_scale < self.push_force_scale else PUSH_FORCE_SCALE_RISE_TIME_S
        if tau <= 1e-9:
            self.push_force_scale = target_scale
            return
        alpha = float(1.0 - np.exp(-max(dt, 0.0) / tau))
        self.push_force_scale = (1.0 - alpha) * self.push_force_scale + alpha * target_scale

    def _write_native_backend_commands(self) -> None:
        if self._native_commanded_insertion is not None:
            # Keep the native commanded insertion synchronized with the Python
            # push command. Writing zero here desynchronizes the strict support
            # window and the actual material progress.
            self._native_commanded_insertion.value = float(self.commanded_push_mm)
        if self._native_commanded_twist is not None:
            self._native_commanded_twist.value = 0.0
        if self._native_br_vector is not None:
            self._native_br_vector.value = self._native_nominal_br_vector.tolist()
        if self._native_external_field_scale is not None:
            if self.is_native_strict:
                self._native_external_field_scale.value = float(
                    np.clip(
                        self._native_strict_magnetic_release_scale() * self._native_strict_field_damping_scale(),
                        0.0,
                        1.0,
                    )
                )
            else:
                self._native_external_field_scale.value = 1.0
        if self._native_external_control_dt is not None:
            # Drive the native magnetic ramp/filter with the controller's
            # actuation dt, not the backend's internal micro-step. Otherwise
            # the field can take hundreds or thousands of solve substeps to
            # reach useful strength in strict GUI runs, which makes the tip
            # look frozen even though the steering target is already correct.
            self._native_external_control_dt.value = float(max(self._native_control_dt_s, 0.0))
        if self._native_external_surface_clearance is not None:
            clearance = float(self.wall_contact_clearance_mm)
            if self.is_native_strict and not np.isfinite(clearance):
                clearance = float(self._native_strict_physical_contact_clearance_mm())
            self._native_external_surface_clearance.value = float(clearance) if np.isfinite(clearance) else float('inf')
        if self._native_external_surface_contact_active is not None:
            self._native_external_surface_contact_active.value = bool(
                self.wall_contact_active and np.isfinite(self.wall_contact_clearance_mm)
            )
        if self.enable_native_virtual_sheath:
            self._write_native_virtual_sheath_targets()

    def _native_virtual_sheath_target_points(self, commanded_push_mm: float | None = None) -> np.ndarray:
        if (not self.enable_native_virtual_sheath) or self._native_virtual_sheath_offsets_mm.size == 0:
            return np.zeros((0, 3), dtype=float)
        push_mm = float(self.commanded_push_mm if commanded_push_mm is None else commanded_push_mm)
        base = self.initial_wire_centers[0, :3].reshape(1, 3)
        offsets = self._native_virtual_sheath_offsets_mm.reshape(-1, 1) + push_mm
        return base + offsets * self.insertion_direction.reshape(1, 3)

    def _write_native_virtual_sheath_targets(self) -> None:
        if (not self.enable_native_virtual_sheath) or self._native_virtual_sheath_target_pos is None:
            return
        target = self._native_virtual_sheath_target_points()
        if target.size == 0:
            return
        self._native_virtual_sheath_target_pos.value = target.tolist()
        if self._native_virtual_sheath_target_rest is not None:
            self._native_virtual_sheath_target_rest.value = target.tolist()
    def _native_virtual_sheath_reaction_n(self) -> float:
        native_drive_reaction = float(max(self._native_debug_scalar(self._native_debug_drive_reaction), 0.0))
        if self._native_debug_drive_reaction is not None:
            # The thrust limiter should react to the axial load carried by the
            # native insertion driver, not to preload inside the virtual sheath.
            # The sheath springs are an internal support mechanism; counting them
            # as "collision resistance" stalls insertion before the tip ever
            # reaches the vessel wall.
            self._native_virtual_sheath_last_reaction_n = native_drive_reaction
            return native_drive_reaction
        if not self.enable_native_virtual_sheath:
            self._native_virtual_sheath_last_reaction_n = native_drive_reaction
            return native_drive_reaction
        if self._native_virtual_sheath_target_pos is None and self._native_debug_drive_reaction is not None:
            self._native_virtual_sheath_last_reaction_n = native_drive_reaction
            return native_drive_reaction
        if len(self._native_virtual_sheath_indices) == 0:
            self._native_virtual_sheath_last_reaction_n = native_drive_reaction
            return native_drive_reaction
        rigid = np.asarray(_read(self._pos), dtype=float)
        if rigid.ndim != 2 or rigid.shape[0] == 0:
            self._native_virtual_sheath_last_reaction_n = native_drive_reaction
            return native_drive_reaction
        indices = np.asarray(self._native_virtual_sheath_indices, dtype=int)
        actual = rigid[indices, :3]
        target = self._native_virtual_sheath_target_points()
        if target.shape != actual.shape:
            count = min(target.shape[0], actual.shape[0], self._native_virtual_sheath_stiffnesses.size)
            actual = actual[:count]
            target = target[:count]
            stiffness = self._native_virtual_sheath_stiffnesses[:count]
        else:
            stiffness = self._native_virtual_sheath_stiffnesses
        if target.size == 0 or stiffness.size == 0:
            self._native_virtual_sheath_last_reaction_n = native_drive_reaction
            return native_drive_reaction
        axial_gap_m = 1.0e-3 * np.maximum((target - actual) @ self.insertion_direction.reshape(3), 0.0)
        sheath_reaction = float(np.sum(stiffness * axial_gap_m))
        reaction = sheath_reaction + native_drive_reaction
        self._native_virtual_sheath_last_reaction_n = reaction
        return reaction

    def _native_thrust_limit_blocks_advance(self) -> bool:
        self._native_strict_hold_active_this_step = False
        if (not self.enable_native_thrust_limit) or self.native_thrust_force_n <= 0.0:
            return False
        reaction = self._native_virtual_sheath_reaction_n()
        wall_gap_mm = self._native_strict_actual_wall_gap_mm() if self.is_native_strict else self._native_debug_scalar(
            self._native_debug_min_lumen_clearance_mm,
            default=float('inf'),
        )
        strict_physical_gap_mm = self._native_strict_physical_contact_clearance_mm() if self.is_native_strict else float('inf')
        strict_min_clearance_mm = self._native_strict_min_lumen_clearance_mm() if self.is_native_strict else float('inf')
        strict_barrier_nodes = self._native_strict_barrier_active_node_count() if self.is_native_strict else 0
        strict_soft_head_stretch = 0.0
        strict_hard_head_stretch = 0.0
        strict_soft_global_stretch = 0.0
        strict_hard_global_stretch = 0.0
        strict_kink_head_hold_threshold = 0.0
        strict_kink_global_hold_threshold = 0.0
        strict_current_head_stretch = 0.0
        strict_current_global_stretch = 0.0
        strict_precontact_kink_danger = False
        strict_near_wall_for_kink = False
        if self.is_native_strict:
            strict_soft_head_stretch, strict_hard_head_stretch = self._native_strict_head_stretch_limits()
            strict_soft_global_stretch, strict_hard_global_stretch = self._native_strict_global_stretch_limits()
            strict_kink_head_hold_threshold = max(
                strict_soft_head_stretch + 0.5 * max(strict_hard_head_stretch - strict_soft_head_stretch, 0.0),
                strict_soft_head_stretch + 0.004,
            )
            strict_kink_global_hold_threshold = max(
                strict_soft_global_stretch + 0.35 * max(strict_hard_global_stretch - strict_soft_global_stretch, 0.0),
                strict_soft_global_stretch + 0.002,
            )
            strict_current_head_stretch = self._native_strict_max_head_stretch()
            strict_support_stretch = self._native_debug_array(self._native_debug_stretch)
            strict_current_global_stretch = float(np.max(np.abs(strict_support_stretch))) if strict_support_stretch.size else 0.0
            strict_near_wall_for_kink = bool(
                self.wall_contact_active
                or (np.isfinite(strict_physical_gap_mm) and strict_physical_gap_mm <= 0.30)
                or (np.isfinite(strict_min_clearance_mm) and strict_min_clearance_mm <= 0.12)
            )
            precontact_barrier_threshold = max(int(ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT) - 1, 4)
            strict_precontact_kink_danger = bool(
                strict_barrier_nodes >= precontact_barrier_threshold
                and strict_near_wall_for_kink
                and (
                    strict_current_head_stretch >= strict_kink_head_hold_threshold
                    or strict_current_global_stretch >= strict_kink_global_hold_threshold
                    or (np.isfinite(strict_min_clearance_mm) and strict_min_clearance_mm <= 0.10)
                )
            )
        enforce_precontact_reaction_limit = False
        strict_precontact_danger = bool(
            self.is_native_strict
            and (
                (np.isfinite(strict_physical_gap_mm) and strict_physical_gap_mm <= 0.0)
                or (np.isfinite(wall_gap_mm) and wall_gap_mm <= -0.05)
            )
        )
        if (
            (not self._native_virtual_sheath_paused)
            and
            (not self.wall_contact_active)
            and (not enforce_precontact_reaction_limit)
            and (not strict_precontact_danger)
            and (not strict_precontact_kink_danger)
        ):
            # The thrust limiter is intended to cap collision resistance once the
            # guidewire is actually pressing on the vessel wall. During free
            # insertion the proximal driver and sheath accumulate elastic load
            # internally, and treating that preload as "collision resistance"
            # stalls the rod long before contact. Keep advancing until wall
            # contact is detected, then use the measured axial reaction to gate
            # further push.
            if self._native_virtual_sheath_pause_reason not in {'clearance', 'headStretch', 'kink'}:
                self._native_virtual_sheath_paused = False
                self._native_virtual_sheath_pause_reason = ''
            return False
        if self.is_native_strict:
            physical_gap_mm = strict_physical_gap_mm
            min_clearance_mm = strict_min_clearance_mm
            barrier_nodes = strict_barrier_nodes
            soft_head_stretch, hard_head_stretch = self._native_strict_head_stretch_limits()
            soft_global_stretch, hard_global_stretch = self._native_strict_global_stretch_limits()
            current_head_stretch = strict_current_head_stretch
            current_global_stretch = strict_current_global_stretch
            severe_head_stretch = current_head_stretch >= hard_head_stretch
            severe_penetration = (
                (np.isfinite(physical_gap_mm) and physical_gap_mm <= -max(self.native_strict_lumen_clamp_tolerance_mm, 0.05))
                or (np.isfinite(wall_gap_mm) and wall_gap_mm <= -0.10)
            )
            kink_barrier_threshold = max(int(ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT) - 1, 4)
            severe_kink = bool(
                barrier_nodes >= kink_barrier_threshold
                and strict_near_wall_for_kink
                and (
                    current_head_stretch >= strict_kink_head_hold_threshold
                    or current_global_stretch >= strict_kink_global_hold_threshold
                    or (np.isfinite(min_clearance_mm) and min_clearance_mm <= 0.10)
                )
            )
            recovery_clearance_mm = max(
                0.08,
                float(ELASTICROD_STRICT_BARRIER_SAFETY_MARGIN_MM),
            )
            if self._native_virtual_sheath_paused:
                if self._native_virtual_sheath_pause_reason == 'kink':
                    recovered = bool(
                        (not severe_penetration)
                        and (not severe_kink)
                        and current_head_stretch <= max(soft_head_stretch, 0.010)
                        and current_global_stretch <= max(soft_global_stretch, 0.018)
                        and barrier_nodes <= 2
                        and ((not np.isfinite(physical_gap_mm)) or physical_gap_mm >= max(recovery_clearance_mm, 0.25))
                        and ((not np.isfinite(min_clearance_mm)) or min_clearance_mm >= max(recovery_clearance_mm, 0.25))
                    )
                else:
                    recovered = bool(
                        (not severe_penetration)
                        and (not severe_kink)
                        and current_head_stretch <= soft_head_stretch
                        and current_global_stretch <= soft_global_stretch
                        and barrier_nodes <= max(kink_barrier_threshold - 2, 1)
                        and ((not np.isfinite(physical_gap_mm)) or physical_gap_mm >= recovery_clearance_mm)
                        and ((not np.isfinite(min_clearance_mm)) or min_clearance_mm >= recovery_clearance_mm)
                    )
                if recovered:
                    print(
                        f'[INFO] [elasticrod-thrust] resumed push at step={self.step_count} '
                        f'wallGap={physical_gap_mm:.4e} mm minClearance={min_clearance_mm:.4e} mm '
                        f'headStretch={current_head_stretch:.4e} globalStretch={current_global_stretch:.4e}'
                    )
                    self._native_virtual_sheath_paused = False
                    self._native_virtual_sheath_pause_reason = ''
                else:
                    self._native_strict_hold_active_this_step = True
                    return True
            if severe_penetration:
                self._native_virtual_sheath_paused = True
                self._native_virtual_sheath_pause_reason = 'clearance'
                print(
                    f'[INFO] [elasticrod-thrust] paused push at step={self.step_count} '
                    f'wallGap={physical_gap_mm:.4e} mm nativeGap={wall_gap_mm:.4e} mm '
                    f'headStretch={current_head_stretch:.4e}'
                )
                self._native_strict_hold_active_this_step = True
                return True
            if severe_kink:
                self._native_virtual_sheath_paused = True
                self._native_virtual_sheath_pause_reason = 'kink'
                print(
                    f'[INFO] [elasticrod-thrust] paused push at step={self.step_count} '
                    f'because strict anti-kink engaged: barrierNodes={barrier_nodes} '
                    f'minClearance={min_clearance_mm:.4e} mm wallGap={physical_gap_mm:.4e} mm '
                    f'headStretch={current_head_stretch:.4e} globalStretch={current_global_stretch:.4e}'
                )
                self._native_strict_hold_active_this_step = True
                return True
            if severe_head_stretch:
                self._native_virtual_sheath_paused = True
                self._native_virtual_sheath_pause_reason = 'headStretch'
                print(
                    f'[INFO] [elasticrod-thrust] paused push at step={self.step_count} '
                    f'wallGap={physical_gap_mm:.4e} mm headStretch={current_head_stretch:.4e}'
                )
                self._native_strict_hold_active_this_step = True
                return True
            if self._native_virtual_sheath_pause_reason in ('clearance', 'headStretch', 'kink'):
                self._native_virtual_sheath_paused = False
                self._native_virtual_sheath_pause_reason = ''
            return False
        pause_threshold = float(self.native_thrust_force_n)
        resume_threshold = 0.8 * pause_threshold
        clearance_hold_threshold = max(
            float(ELASTICROD_CONTACT_DISTANCE_MM) if self.is_native_strict else float(CONTACT_DISTANCE_MM),
            0.05,
        )
        strict_realign_required = (
            self.is_native_strict
            and self.wall_contact_active
            and self._native_virtual_sheath_pause_reason == 'reaction'
            and (
                self.steering_angle_deg >= max(float(STEERING_MISALIGN_EXIT_DEG), 18.0)
                or self.filtered_tip_forward_speed_mm_s < -0.10
            )
        )
        if self._native_virtual_sheath_paused:
            if strict_realign_required:
                return True
            clearance_blocked = np.isfinite(wall_gap_mm) and wall_gap_mm <= 0.0
            if (not clearance_blocked) and reaction <= resume_threshold and (
                (not np.isfinite(wall_gap_mm)) or wall_gap_mm >= clearance_hold_threshold
            ):
                self._native_virtual_sheath_paused = False
                self._native_virtual_sheath_pause_reason = ''
                print(
                    f'[INFO] [elasticrod-thrust] resumed push at step={self.step_count} '
                    f'reaction={reaction:.4e} N limit={pause_threshold:.4e} N'
                )
            else:
                return True
        if np.isfinite(wall_gap_mm) and wall_gap_mm <= 0.0:
            self._native_virtual_sheath_paused = True
            self._native_virtual_sheath_pause_reason = 'clearance'
            print(
                f'[INFO] [elasticrod-thrust] paused push at step={self.step_count} '
                f'wallGap={wall_gap_mm:.4e} mm reaction={reaction:.4e} N'
            )
            return True
        if reaction >= pause_threshold:
            self._native_virtual_sheath_paused = True
            self._native_virtual_sheath_pause_reason = 'reaction'
            print(
                f'[INFO] [elasticrod-thrust] paused push at step={self.step_count} '
                f'reaction={reaction:.4e} N limit={pause_threshold:.4e} N'
            )
            return True
        return False

    def _native_strict_driver_follow_scale(self) -> float:
        if not (self.is_native_strict and self.native_strict_boundary_driver_enabled):
            return 1.0

        axial_error_mm = max(
            self._native_debug_scalar(self._native_debug_max_axial_boundary_error, default=0.0),
            0.0,
        )
        drive_reaction_n = max(
            self._native_debug_scalar(self._native_debug_drive_reaction, default=0.0),
            0.0,
        )

        scale = 1.0
        reasons: list[str] = []

        soft_axial = max(float(ELASTICROD_STRICT_DRIVER_AXIAL_ERROR_SOFT_LIMIT_MM), 0.0)
        hard_axial = max(float(ELASTICROD_STRICT_DRIVER_AXIAL_ERROR_HARD_LIMIT_MM), soft_axial)
        soft_reaction = max(float(ELASTICROD_STRICT_DRIVER_REACTION_SOFT_LIMIT_N), 0.0)
        hard_reaction = max(float(ELASTICROD_STRICT_DRIVER_REACTION_HARD_LIMIT_N), soft_reaction)
        if hard_axial <= 0.0 and hard_reaction <= 0.0:
            self._native_strict_driver_limited = False
            self._native_strict_driver_limit_reason = ''
            return 1.0
        if hard_axial > soft_axial and axial_error_mm > soft_axial:
            axial_scale = float(np.clip((hard_axial - axial_error_mm) / (hard_axial - soft_axial), 0.0, 1.0))
            scale = min(scale, axial_scale)
            reasons.append(f'axialError={axial_error_mm:.4f} mm')
        elif hard_axial > 0.0 and axial_error_mm >= hard_axial:
            scale = 0.0
            reasons.append(f'axialError={axial_error_mm:.4f} mm')

        if hard_reaction > soft_reaction and drive_reaction_n > soft_reaction:
            reaction_scale = float(np.clip((hard_reaction - drive_reaction_n) / (hard_reaction - soft_reaction), 0.0, 1.0))
            scale = min(scale, reaction_scale)
            reasons.append(f'driveReaction={drive_reaction_n:.4e} N')
        elif hard_reaction > 0.0 and drive_reaction_n >= hard_reaction:
            scale = 0.0
            reasons.append(f'driveReaction={drive_reaction_n:.4e} N')

        if scale < 1.0 - 1.0e-6:
            reason = ', '.join(reasons) if reasons else 'proximal driver lag'
            if (not self._native_strict_driver_limited) or reason != self._native_strict_driver_limit_reason:
                print(
                    f'[INFO] [elasticrod-strict] insertion throttled at step={self.step_count} '
                    f'scale={scale:.3f} because {reason}'
                )
            self._native_strict_driver_limited = True
            self._native_strict_driver_limit_reason = reason
        elif self._native_strict_driver_limited:
            print(
                f'[INFO] [elasticrod-strict] insertion resumed at step={self.step_count} '
                f'axialError={axial_error_mm:.4f} mm driveReaction={drive_reaction_n:.4e} N'
            )
            self._native_strict_driver_limited = False
            self._native_strict_driver_limit_reason = ''

        return float(scale)

    def _enforce_native_proximal_boundary(self) -> None:
        if not self.is_native_backend:
            return
        # `elasticrod` 闁告艾娴烽顒佺▔瀹ュ懎鏅欓柛?Python 濞撴皜鍛暠閺夆晜鍨归顒佹媴瀹ュ洨鏋傜痪顓у墲椤╊偊宕樺▎宥佸亾?        # 濞戞柨顑呮晶鐘虫交濞嗘挸娅℃慨锝呯箰閹舵岸鎯勭€涙ê澶嶉柡鈧悷鏉款枀闁告垹濮抽柌婊堟嚍閸屾粌浠柣銊ュ缂嶅懐绱旈鍏煎濠殿喗瀵ч埀顑跨筏缁辨繃瀵煎顒佸闁告鍠撻弫鎾存綇閸︻厽娅曠€殿喖婀遍崫鈧柕?        # 闂傚懏鍔曠槐鈥承ч崒婢帡宕抽妸銈勭鞍闁告瑥锕﹂～顐﹀箻閻愬弶鎯欓幖瀛樻煣缁ㄤ即鎯勯崨濠傗叺闁哄顔愮槐婵嬪嫉閳ь剛绱掗崼锝冣偓鍐偝妫颁浇绀嬪Λ锝嗙墪閹舵氨绮ｆ担绋跨秮闁告粌鏈弳鐔煎磹閻撳骸绲洪柡渚婄祷閳?        # 闁告鍠撻弫鎾诲触鎼达綆浼傞柣婊勬緲濠€顏堝矗椤忓懎澶嶉柡鈧?`commandedInsertion / commandedTwist` 闁告稒鍨濋幎銈夋晬?        # 閻庡湱鍋ゅ顖涙綇閸︻厽娅曢柛娆忕Т婵繒鈧懓鑻崣蹇涙偩濞嗘垹鑸?C++ 闁告梹绋戦鐔肺熼垾宕団偓閿嬪緞閸曨厽鍊為柕?        return

    def _update_push_force(self, dt: float) -> None:
        if self.proximal_push_ff is None:
            return
        ff_indices: list[int] | None = None
        if self.is_native_backend and self.is_native_strict and self.native_strict_boundary_driver_enabled:
            active_indices = []
            ff_indices = [0] if self.node_count > 0 else []
        elif self.is_native_backend and self.is_native_strict:
            active_indices = self._strict_hand_push_indices()
        else:
            active_indices = (
                self._native_entry_push_indices()
                if self.is_native_backend
                else list(range(self.drive_count))
            )
        if ff_indices is None:
            ff_indices = active_indices
        if self.is_native_backend:
            self._set_forcefield_indices(self._proximal_push_indices_data, ff_indices)
        if self.use_kinematic_beam_insertion:
            total_force = 0.0
            per_node_force = self.insertion_direction * 0.0
            mode_desc = 'Kinematic insertion'
        else:
            if self.is_native_backend and self.use_native_displacement_feed:
                if self.is_native_strict and (self.wall_contact_active or self._native_strict_hold_active_this_step):
                    ramp = float(np.clip(self._native_startup_ramp_scale(), 0.0, 1.0))
                    contact_push_scale = 0.22
                    total_force = float(
                        np.clip(
                            contact_push_scale
                            * float(ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_N)
                            * ramp
                            * float(np.clip(self.push_force_scale, 0.0, 1.0)),
                            0.0,
                            0.40 * float(ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_MAX_N),
                        )
                    )
                    mode_desc = 'Strict contact fallback push (ConstantForceField)'
                else:
                    total_force = 0.0
                    mode_desc = (
                        'Safe displacement push'
                        if self.is_native_safe else
                        'Strict direct tail feed'
                    )
            elif self.is_native_backend and self.is_native_strict and self.native_strict_boundary_driver_enabled:
                total_force = 0.0
                mode_desc = 'Strict native boundary insertion'
            elif self.is_native_backend and self.is_native_strict:
                ramp = float(np.clip(self._native_startup_ramp_scale(), 0.0, 1.0))
                if self._native_virtual_sheath_paused or self._native_strict_hold_active_this_step:
                    total_force = 0.0
                else:
                    total_force = float(
                        np.clip(
                            float(ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_N) * ramp * float(np.clip(self.push_force_scale, 0.0, 1.0)),
                            0.0,
                            float(ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_MAX_N),
                        )
                    )
                mode_desc = 'Strict external support push (ConstantForceField)'
            else:
                total_force = float(np.clip(self.nominal_push_force_total * self.push_force_scale, 0.0, PUSH_FORCE_MAX_TOTAL))
                if self.is_native_backend and self._native_virtual_sheath_paused:
                    total_force = 0.0
                mode_desc = 'Physical push-force insertion'
            per_node_force = self.insertion_direction * (total_force / max(len(active_indices), 1))
        self.proximal_push_ff.forces.value = [
            [float(per_node_force[0]), float(per_node_force[1]), float(per_node_force[2]), 0.0, 0.0, 0.0]
            for _ in ff_indices
        ]
        if not self._force_diag_printed:
            node_label = (
                'nativeBoundaryWindow'
                if (self.is_native_backend and self.is_native_strict and self.native_strict_boundary_driver_enabled)
                else ('externalPushNodes' if (self.is_native_backend and self.is_native_strict) else 'driveNodes')
            )
            print(
                f'[INFO] {mode_desc} enabled: {node_label}={active_indices}, '
                f'targetSpeed={self.push_force_target_speed_mm_s:.3f} mm/s, navigationMode={self.navigation_mode}'
            )
            self._force_diag_printed = True

    def _update_native_axial_path_assist_force(self) -> None:
        if self.native_axial_assist_ff is None:
            return
        deficit_mm = max(float(self.commanded_push_mm - self.tip_progress_raw_mm), 0.0)
        assist_targets, distal_mode = self._native_axial_assist_targets(deficit_mm)
        active_indices = [int(idx) for idx, _ in assist_targets]
        self._set_forcefield_indices(self._native_axial_assist_indices_data, active_indices)
        zero = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in active_indices]
        if self.is_native_strict:
            self.native_axial_assist_ff.forces.value = zero
            return
        if (not self.enable_native_axial_path_assist) or len(active_indices) == 0:
            self.native_axial_assist_ff.forces.value = zero
            return

        if (
            self._native_virtual_sheath_paused
            or self.commanded_push_mm <= 0.0
        ):
            self.native_axial_assist_ff.forces.value = zero
            return

        threshold = max(float(ELASTICROD_AXIAL_PATH_ASSIST_DEFICIT_MM), 1.0e-6)
        scale = float(np.clip((deficit_mm - threshold) / threshold, 0.0, 1.0))
        contact_gate = 0.0
        if self.wall_contact_active:
            clearance_gate = 0.65
            if np.isfinite(self.wall_contact_clearance_mm):
                clearance_gate = float(np.clip((0.30 - self.wall_contact_clearance_mm) / 0.24, 0.0, 1.0))
            stall_gate = float(np.clip((1.20 - max(self.filtered_tip_forward_speed_mm_s, 0.0)) / 1.20, 0.0, 1.0))
            contact_gate = max(clearance_gate, stall_gate, 0.65)
            scale = max(
                scale,
                float(ELASTICROD_AXIAL_PATH_ASSIST_CONTACT_MIN_SCALE) * contact_gate,
            )
        if scale <= 1.0e-6:
            self.native_axial_assist_ff.forces.value = zero
            return

        assist_push_scale = float(
            np.clip(
                self.push_force_scale,
                0.0,
                max(1.0, float(ELASTICROD_AXIAL_PATH_ASSIST_MAX_PUSH_SCALE)),
            )
        )
        total_force = (
            float(ELASTICROD_AXIAL_PATH_ASSIST_FORCE_N)
            * scale
            * assist_push_scale
            * float(self._native_startup_ramp_scale())
        )
        if contact_gate > 0.0:
            total_force *= float(
                1.0
                + (float(ELASTICROD_AXIAL_PATH_ASSIST_CONTACT_FORCE_SCALE) - 1.0) * contact_gate
            )
        if self.is_native_safe and total_force > 0.0:
            head_stretch = max(
                float(self._native_debug_scalar(self._native_debug_max_head_stretch, default=0.0)),
                0.0,
            )
            stretch_profile = self._native_debug_array(self._native_debug_stretch)
            max_stretch = float(np.max(np.abs(stretch_profile))) if stretch_profile.size else 0.0
            barrier_nodes = max(self._native_strict_barrier_active_node_count(), 0)
            anti_kink_gate = 0.0
            if self.wall_contact_active or barrier_nodes > 0:
                if np.isfinite(self.wall_contact_clearance_mm):
                    anti_kink_gate = max(
                        anti_kink_gate,
                        float(np.clip((0.18 - float(self.wall_contact_clearance_mm)) / 0.12, 0.0, 1.0)),
                    )
                anti_kink_gate = max(
                    anti_kink_gate,
                    float(np.clip((head_stretch - 1.2e-2) / 1.2e-2, 0.0, 1.0)),
                    float(np.clip((max_stretch - 4.0e-2) / 8.0e-2, 0.0, 1.0)),
                    float(np.clip((barrier_nodes - 4.0) / 4.0, 0.0, 1.0)),
                )
            if anti_kink_gate > 0.0:
                anti_kink_floor = 0.45
                total_force *= float((1.0 - anti_kink_gate) + anti_kink_gate * anti_kink_floor)
        if total_force <= 0.0:
            self.native_axial_assist_ff.forces.value = zero
            return

        if distal_mode and len(assist_targets) > 1:
            path_s = np.asarray([float(s) for _, s in assist_targets], dtype=float)
            span = max(float(np.max(path_s) - np.min(path_s)), 1.0e-6)
            weights = 0.85 + 0.45 * (float(np.max(path_s)) - path_s) / span
        else:
            weights = np.ones(len(assist_targets), dtype=float)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1.0e-9:
            self.native_axial_assist_ff.forces.value = zero
            return

        mode = 'distalWindow' if distal_mode else 'entryWindow'
        if mode != self._native_axial_assist_mode:
            print(
                f'[INFO] [elasticrod-safe] axial assist switched to {mode}: '
                f'nodes={active_indices}, deficit={deficit_mm:.3f} mm, '
                f'wallContact={self.wall_contact_active}, totalForce={total_force:.3f} N'
            )
            self._native_axial_assist_mode = mode

        forces = []
        for (node_idx, path_s_mm), weight in zip(assist_targets, weights):
            tangent = self._centerline_tangent(path_s_mm if distal_mode else self._node_s(node_idx))
            force = (total_force * float(weight) / weight_sum) * tangent
            forces.append([float(force[0]), float(force[1]), float(force[2]), 0.0, 0.0, 0.0])
        self.native_axial_assist_ff.forces.value = forces

    def _tip_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        rigid = self._current_rigid_state()
        tip = rigid[-1]
        return tip[:3].copy(), tip[3:7].copy()

    def _tip_dir(self, tip_quat: np.ndarray) -> np.ndarray:
        rigid = self._current_rigid_state()
        if rigid.shape[0] >= 2:
            d = _normalize(rigid[-1, :3] - rigid[-2, :3])
            if np.linalg.norm(d) > 1e-12:
                return d
        d = _normalize(_quat_rotate(tip_quat, [0.0, 0.0, 1.0]))
        return d if np.linalg.norm(d) > 1e-12 else self.insertion_direction.copy()

    def _update_target_marker(self, point: np.ndarray) -> None:
        if self._target_marker_pos is None:
            return
        marker = _marker_points(point, TARGET_MARKER_SIZE_MM)
        with _writeable(self._target_marker_pos) as pos:
            pos[:] = marker

    def _update_force_arrow(self, field_vec: np.ndarray) -> None:
        if self._force_arrow_pos is None:
            return
        p = self.force_arrow_anchor
        direction = _normalize(field_vec)
        if np.linalg.norm(direction) < 1e-12:
            direction = self.insertion_direction.copy()
        shaft_end = p + MAGNETIC_FORCE_ARROW_LENGTH_MM * direction
        head_base = shaft_end - MAGNETIC_FORCE_ARROW_HEAD_LENGTH_MM * direction
        side = np.cross(direction, np.array([0.0, 0.0, 1.0], dtype=float))
        if np.linalg.norm(side) < 1e-12:
            side = np.cross(direction, np.array([0.0, 1.0, 0.0], dtype=float))
        side = _normalize(side)
        if np.linalg.norm(side) < 1e-12:
            side = np.array([1.0, 0.0, 0.0], dtype=float)
        up = _normalize(np.cross(direction, side))
        if np.linalg.norm(up) < 1e-12:
            up = np.array([0.0, 1.0, 0.0], dtype=float)
        left = head_base + MAGNETIC_FORCE_ARROW_HEAD_WIDTH_MM * side
        right = head_base - MAGNETIC_FORCE_ARROW_HEAD_WIDTH_MM * side
        top = head_base + MAGNETIC_FORCE_ARROW_HEAD_WIDTH_MM * up
        bottom = head_base - MAGNETIC_FORCE_ARROW_HEAD_WIDTH_MM * up
        points = np.asarray([p, shaft_end, left, right, top, bottom], dtype=float)
        with _writeable(self._force_arrow_pos) as pos:
            pos[:] = points

    def _debug_vector(self, data, fallback: np.ndarray) -> np.ndarray:
        if data is None:
            return np.asarray(fallback, dtype=float).reshape(3)
        try:
            arr = np.asarray(data.value, dtype=float).reshape(3)
            return arr
        except Exception:
            return np.asarray(fallback, dtype=float).reshape(3)

    def _update_steering_state(
        self,
        tip_pos: np.ndarray,
        tip_dir: np.ndarray,
        target_point: np.ndarray,
        desired_dir: np.ndarray | None = None,
    ) -> None:
        if desired_dir is None:
            desired_dir = np.asarray(target_point, dtype=float).reshape(3) - np.asarray(tip_pos, dtype=float).reshape(3)
        desired_dir = _normalize(np.asarray(desired_dir, dtype=float).reshape(3))
        if np.linalg.norm(desired_dir) < 1e-12:
            desired_dir = tip_dir
        cos_theta = float(np.clip(np.dot(_normalize(tip_dir), desired_dir), -1.0, 1.0))
        self.steering_angle_deg = float(np.degrees(np.arccos(cos_theta)))
        if self.steering_misaligned:
            self.steering_misaligned = self.steering_angle_deg >= STEERING_MISALIGN_EXIT_DEG
        else:
            self.steering_misaligned = self.steering_angle_deg >= STEERING_MISALIGN_ENTER_DEG

    def _update_beam_compression_state(self) -> None:
        """
        `beam` 闁告艾娴烽顒勬儍閸曨垪鍋撴担椋庮偩濞寸姴绉堕崝褔寮伴婵堢闁告柣鍔岄鐔煎川閹存帗濮㈤柨娑樿嫰濞叉粌顫㈤妶鍡樹粯闁告浜跺▍鎾绘儍閸曨剙鍓伴柛鎰悁缁楀寮伴妞诲亾濠婂啫顫斿棰濅簻閵囧洭鍨惧┑鎾剁
        闁兼澘鏈Σ鎼佸灳濠婂棛绠紒鏃戝灠閹斥剝绂掗妶澶嗗亾娴ｇ寮抽梺鎻掔箲鐎垫梻绱掗鐑嗘澔闁告梻濯寸槐婵囨媴閸℃姣炵紒鏃戝灣濠€锛勨偓鍦仜婢х姵娼诲☉銏犳婵炲备鍓濆﹢渚€宕ョ仦缁㈠妱濠⒀呭仱閺嗛亶鍨惧┑鍕ㄥ亾?
        閺夆晜鐟ょ槐浼村箮婵犲偆妯嬪ù锝嗙懇閺嗚鲸鎯旈敃鈧敮鍥╃磽閳轰焦韬柛蹇嬪劚瑜版盯宕畝鍐缂佹棏鍨伴梿鍡涘即閸欏鍞介梺鎻掔焿缁辨繈寮甸埀顒傜磼閸絻鈧啴鎮虫０浣界閻忕偐鍋撻梺顔哄妼閻寮撮幓鎺撳闁搞儺婢€缁楋綁濡?        閺夆晜鐟╅崳鐑芥偨閵娿倗顏卞☉鎿冧簽閻ｆ繈宕￠弴鈾€鍋撶仦鐐畳闁轰礁鐗忓▓鎴炵┍濠靛洤袘闂佹彃楠忕槐?        `compression = commanded_push_mm - tip_progress_raw_mm`
        濞戞挴鍋撻柡鍐跨畱閻ｇ姷鎼鹃崨鎵畺闂傚啫鐗嗛埀顒傘€嬬槐婵堜焊鏉堛劌惟闂侇偂妞掔粭锝夋焻閻斿嘲顔婇梻鍕Т閸╁矂寮告担椋庣У闁挎稑鐬煎ú鍧楀礆閺夎法姣炵紒鏃戝灦閸ｆ悂寮幏宀€顎€濞戞挸顭堥埀?        """
        if self.is_native_backend or self.use_kinematic_beam_insertion:
            self.beam_compression_mm = 0.0
            self.beam_compression_active = False
            return
        self.beam_compression_mm = max(float(self.drive_push_mm - self.tip_progress_raw_mm), 0.0)
        self.beam_compression_active = False
        return

    def _update_beam_stall_state(self) -> None:
        if self.is_native_backend or self.use_kinematic_beam_insertion:
            self.beam_stall_active = False
            return

        compression = float(max(self.beam_compression_mm, 0.0))
        speed = float(max(self.filtered_tip_forward_speed_mm_s, 0.0))
        prev = self.beam_stall_active
        if self.beam_stall_active:
            self.beam_stall_active = (
                compression >= BEAM_STALL_COMPRESSION_EXIT_MM
                and speed <= BEAM_STALL_SPEED_EXIT_MM_S
            )
        else:
            self.beam_stall_active = (
                compression >= BEAM_STALL_COMPRESSION_ENTER_MM
                and speed <= BEAM_STALL_SPEED_ENTER_MM_S
            )

        if prev != self.beam_stall_active:
            state = 'entered' if self.beam_stall_active else 'released'
            print(
                f'[INFO] Beam stall regulator {state}: compression={compression:.4f} mm, '
                f'tipSpeed={speed:.3f} mm/s'
            )

    def _clear_tip_torque_ff(self) -> None:
        if self.tip_torque_ff is None:
            return
        self.tip_torque_ff.forces.value = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for _ in range(len(self.distal_indices))
        ]

    def _fallback_target_state(self, tip_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proj_point, proj_s = self._project_to_centerline(np.asarray(tip_pos, dtype=float).reshape(3))
        if self.use_native_displacement_feed:
            dynamic_lookahead_mm = max(float(ELASTICROD_MAGNETIC_LOOKAHEAD_DISTANCE_MM), 0.5 * self.rest_spacing_mm)
            dynamic_lookahead_mm *= max(float(self.push_force_scale), 0.5)
            lookahead_s = float(np.clip(proj_s + dynamic_lookahead_mm, 0.0, self.path_len))
            self._fallback_nav_s_mm = float(proj_s)
        else:
            self._fallback_nav_s_mm = max(self._fallback_nav_s_mm, float(proj_s))
            lookahead_s = float(np.clip(self._fallback_nav_s_mm + MAGNETIC_LOOKAHEAD_DISTANCE_MM, 0.0, self.path_len))
        target_point = _interp(self.centerline[:, :3], self.centerline_cum, lookahead_s)
        center_pull = proj_point - np.asarray(tip_pos, dtype=float).reshape(3)
        forward_pull = target_point - proj_point
        if self.use_native_displacement_feed:
            target_dir = _normalize(0.70 * forward_pull + 0.30 * center_pull)
        else:
            target_dir = _normalize(target_point - np.asarray(tip_pos, dtype=float).reshape(3))
        if np.linalg.norm(target_dir) < 1e-12:
            target_dir = _normalize(target_point - proj_point)
        if np.linalg.norm(target_dir) < 1e-12:
            target_dir = self.insertion_direction.copy()
        return target_point, target_dir

    def _fallback_nearest_segment_tangent(self, tip_pos: np.ndarray) -> np.ndarray:
        tip = np.asarray(tip_pos, dtype=float).reshape(3)
        if self.centerline.shape[0] < 2:
            return self.insertion_direction.copy()

        min_distance = float('inf')
        contact_index = 0
        for i in range(self.centerline.shape[0] - 1):
            a = self.centerline[i, :3]
            b = self.centerline[i + 1, :3]
            x_current_dis = 0.5 * (float(np.linalg.norm(tip - a)) + float(np.linalg.norm(tip - b)))
            if x_current_dis < min_distance:
                min_distance = x_current_dis
                contact_index = i

        tangent = _normalize(self.centerline[contact_index + 1, :3] - self.centerline[contact_index, :3])
        return tangent if np.linalg.norm(tangent) > 1e-12 else self.insertion_direction.copy()

    def _apply_python_magnetic_fallback(self, tip_pos: np.ndarray, tip_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.use_python_magnetic_fallback or self._fallback_field_controller is None or self._fallback_head_model is None:
            self._clear_tip_torque_ff()
            self._fallback_target_point = self.entry_point.copy()
            self._fallback_ba_vector = self.insertion_direction.copy()
            self._fallback_force_vector = self.insertion_direction.copy()
            return self._fallback_target_point, self._fallback_ba_vector, self._fallback_force_vector

        rigid = np.asarray(_read(self._pos), dtype=float)
        target_point, target_dir = self._fallback_target_state(tip_pos)
        moment_dir = self._fallback_head_model.head_direction(rigid)
        if np.linalg.norm(moment_dir) < 1e-12:
            moment_dir = tip_dir

        # Use direction from tip to target point for magnetic field, not centerline tangent
        # This ensures the magnetic field always pulls the tip toward the target
        ba_dir = target_dir.copy()
        self._fallback_field_controller.filtered_direction = ba_dir.copy()
        self._fallback_field_controller.field.set_direction(ba_dir)
        b_vec = self._fallback_field_controller.field.vector.copy()

        # Calculate angle between current tip direction and target direction
        # Only apply magnetic torque when there's a significant bend ahead
        tip_to_target_angle_deg = np.degrees(np.arccos(np.clip(np.dot(moment_dir, ba_dir), -1.0, 1.0)))
        magnetic_activation_threshold_deg = 15.0  # Only activate magnetic field when bend > 15 degrees

        if tip_to_target_angle_deg > magnetic_activation_threshold_deg:
            torque = self._fallback_field_controller.magnetic_torque(moment_dir)
        else:
            # No significant bend ahead, disable magnetic torque to avoid unwanted deflection
            torque = np.zeros(3, dtype=float)

        force = np.zeros(3, dtype=float)
        if self.tip_torque_ff is not None:
            # Apply torque ONLY to the tip node, not distributed across all magnetic head nodes
            # This prevents unrealistic internal bending when there's no external constraint
            wrenches = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in self.distal_indices]
            if len(self.distal_indices) > 0:
                # Apply full torque only to the last node (tip)
                wrenches[-1] = [0.0, 0.0, 0.0, float(torque[0]), float(torque[1]), float(torque[2])]
            self.tip_torque_ff.forces.value = wrenches

        self._fallback_target_point = np.asarray(target_point, dtype=float).reshape(3)
        self._fallback_ba_vector = _normalize(b_vec)
        if np.linalg.norm(self._fallback_ba_vector) < 1e-12:
            self._fallback_ba_vector = self.insertion_direction.copy()
        self._fallback_force_vector = self._fallback_ba_vector.copy()
        return self._fallback_target_point, self._fallback_ba_vector, self._fallback_force_vector

    def _sync_debug_visuals(self, tip_pos: np.ndarray, tip_dir: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.magnetic_force_field is not None:
            target_point = self._debug_vector(self._debug_target_point, self.entry_point)
            ba_vector = self._debug_vector(self._debug_ba_vector, self.insertion_direction)
            raw_force_vector = self._debug_vector(
                self._debug_force_vector,
                ba_vector if np.linalg.norm(ba_vector) > 1e-12 else self.insertion_direction,
            )
            force_vector = raw_force_vector
            self._clear_tip_torque_ff()
        else:
            tip_dir = self._tip_dir(np.asarray(self._tip_pose()[1], dtype=float))
            target_point, ba_vector, force_vector = self._apply_python_magnetic_fallback(tip_pos, tip_dir)

        refresh_visuals = True
        if self.is_native_realtime:
            refresh_visuals = (
                self.step_count <= 1
                or self._native_runtime_last_visual_band != self._native_runtime_band
                or (self.step_count - self._native_runtime_last_visual_step) >= 5
            )
        if refresh_visuals:
            self._update_target_marker(target_point)
            # The UI arrow is intended to show the magnetic field direction.
            # In the native strict path `debugForceVector` is the resultant
            # distal nodal force, which can look outward even after `baVector`
            # has already rotated inward toward the centerline.
            arrow_vector = ba_vector if self.is_native_backend else force_vector
            self._update_force_arrow(arrow_vector)
            self._native_runtime_last_visual_step = self.step_count
            self._native_runtime_last_visual_band = self._native_runtime_band
        return target_point, ba_vector, force_vector

    def _update_camera_follow(self, tip_pos: np.ndarray) -> None:
        """
        閺夆晜鐟╅崳鐑藉矗椤忓懏鏆柛娆樺灥椤宕犻弽顐ｇゲ闁哄牏灏ㄧ槐婵囩▔瀹ュ棙鏆ù鐘侯唺缂嶅秹宕欓悩杈╃Э闁靛棔娴囬惌鎯ь嚗閸曨剙鐏楅柟璨夊啫鐓戠紒鐘愁殕绾爼濡?
        闁稿纰嶇涵璺侯嚗閸垺绾柟鎭掑劵缁?        - `lookAt` 婵絽绻愰幎姘舵煥娴ｇ鐓傞悗鐢告？缁楋絾寰勯幘顔煎姤濞达絽绉堕悿鍡涙晬鐏炶偐绠介悹鍥︾閵囨棃鏌堥妸銉綏缂備礁鐗呯紞鍛鎼淬値娼掗梺鎻掔凹閼垫垼绠涢崘璺ㄥ耿
        - `position` 濞ｅ洦绻冪€垫梹绋夐埀顒佺▔椤忓嫭绁奸悗瑙勭煯缁楁﹢鎮剧仦鎴掔剨缂佸绱槐婵堢驳婢跺骞嗗ù婊冪岸閳ь剚绮撻弳鍛緞绾惧顎€闁活偀鍋撳鑸垫尦閸庡瓨绋夐埀顒傛導瀹勪即鎸紒澶屽劦閳ь剚绻愮槐?          閺夆晜鐟﹂悧閬嶆偨婵犳碍妗ㄥ☉鎾崇С缁辨壆鎼炬繝鍐€欓悺鎺戯龚缁绘瑩鏁嶇仦鑲╃槏濞戞挸绉崇槐鎵玻娴ｅ搫濮х紓鍌楁櫅缁惰鲸寰勯鍥╃闁?        """
        if self._camera_position is None or self._camera_lookat is None:
            return
        tip = np.asarray(tip_pos, dtype=float).reshape(3)
        camera_pos = tip + self.camera_follow_offset
        self._camera_position.value = camera_pos.tolist()
        self._camera_lookat.value = tip.tolist()

    def _log_step_state(
        self,
        tip_dir: np.ndarray,
        target_point: np.ndarray,
        ba_vector: np.ndarray,
        force_vector: np.ndarray,
    ) -> None:
        if self.is_native_realtime:
            if self.step_count % 250 != 0:
                return
        elif DEBUG_PRINT_EVERY <= 0 or self.step_count % DEBUG_PRINT_EVERY != 0:
            return
        native_tail = ''
        if self.is_native_backend:
            if self.is_native_strict:
                active_push_nodes = self._strict_hand_push_indices()
                torque_sin = self._native_debug_scalar(self._debug_torque_sin)
                tangent_field_angle = self._native_debug_scalar(self._debug_distal_tangent_field_angle_deg)
                upcoming_turn_deg = self._native_strict_upcoming_turn_deg()
                bend_severity = self._native_strict_bend_severity()
                scheduled_field_scale = self._native_strict_scheduled_field_scale()
                scheduled_field_scale_base = self._native_debug_scalar(
                    self._debug_scheduled_field_scale_base,
                    default=scheduled_field_scale,
                )
                strict_need_alpha = self._native_debug_scalar(self._debug_strict_steering_need_alpha, default=0.0)
                entry_release_alpha = self._native_debug_scalar(self._debug_entry_release_alpha, default=1.0)
                recenter_alpha = self._native_strict_recentering_alpha()
                distal_node_force = self._debug_vector(self._debug_force_vector, [0.0, 0.0, 0.0])
                torque_vector = self._debug_vector(self._debug_torque_vector, [0.0, 0.0, 0.0])
                assist_vector = self._debug_vector(self._debug_assist_force_vector, [0.0, 0.0, 0.0])
                barrier_vector = self._debug_vector(self._native_debug_barrier_force_vector, [0.0, 0.0, 0.0])
                support_occupancy, drive_occupancy = self._native_strict_support_stats()
                native_tail = (
                    f' runtimeBand={self._native_runtime_band} runtimeDt={self._native_runtime_dt_s:.6f}s '
                    f'runtimeBandReason={self._native_runtime_band_reason} '
                    f'controlMode={self._native_control_time_mode} controlDt={self._native_control_dt_s:.6f}s '
                    f'externalPushNodes={active_push_nodes} '
                    f'virtualSheathArc={self._native_virtual_sheath_actual_arc_mm:.3f}/{self._native_virtual_sheath_configured_arc_mm:.3f} mm '
                    f'supportCorridorOccupancy={support_occupancy} driveWindowOccupancy={drive_occupancy} '
                    f'baseProgress={self.base_progress_mm:.4f} mm midProgress={self.mid_progress_mm:.4f} mm '
                    f'tipProgress={self.tip_axial_progress_mm:.4f} mm tipPathProgress={self.tip_progress_raw_mm:.4f} mm '
                    f'feedBoost={self._native_strict_guided_feed_boost():.3f} '
                    f'maxStretch={self._native_debug_max_abs(self._native_debug_stretch):.4e} '
                    f'headStretch={self._native_strict_max_head_stretch():.4e} '
                    f'minClearance={self._native_strict_min_lumen_clearance_mm():.4f} mm '
                    f'bendSeverity={bend_severity:.3f} upcomingTurn={upcoming_turn_deg:.3f} deg '
                    f'entryReleaseAlpha={entry_release_alpha:.3f} strictNeedAlpha={strict_need_alpha:.3f} '
                    f'scheduledFieldScale={scheduled_field_scale_base:.3f}->{scheduled_field_scale:.3f} recenterAlpha={recenter_alpha:.3f} '
                    f'magTorqueSin={torque_sin:.4f} '
                    f'magTangentFieldAngle={tangent_field_angle:.3f} deg '
                    f'tipCenteringForce={np.round(assist_vector, 6).tolist()} '
                    f'barrierForce={np.round(barrier_vector, 6).tolist()} '
                    f'magTorque={np.round(torque_vector, 6).tolist()} '
                    f'magDistalNodeForce={np.round(distal_node_force, 6).tolist()}'
                )
            else:
                torque_vector = self._debug_vector(self._debug_torque_vector, [0.0, 0.0, 0.0])
                moment_vector = self._debug_vector(self._debug_magnetic_moment_vector, [0.0, 0.0, 0.0])
                assist_vector = self._debug_vector(self._debug_assist_force_vector, [0.0, 0.0, 0.0])
                torque_sin = self._native_debug_scalar(self._debug_torque_sin)
                outward_assist = self._native_debug_scalar(self._debug_outward_assist_component)
                tangent_field_angle = self._native_debug_scalar(self._debug_distal_tangent_field_angle_deg)
                barrier_vector = self._debug_vector(self._native_debug_barrier_force_vector, [0.0, 0.0, 0.0])
                support_occupancy, drive_occupancy = self._native_strict_support_stats()
                native_tail = (
                    f' firstContactStep={self._native_first_contact_step} '
                    f'postContactJump={self._native_max_post_contact_jump_mm:.4f} mm '
                    f'safeRecovery={self.is_native_safe and ELASTICROD_ENABLE_SAFE_RECOVERY} '
                    f'runtimeBand={self._native_runtime_band} runtimeDt={self._native_runtime_dt_s:.6f}s '
                    f'controlMode={self._native_control_time_mode} controlDt={self._native_control_dt_s:.6f}s '
                    f'runtimeSolver={self._native_runtime_solver_max_iter}/{self._native_runtime_solver_tolerance:.1e} '
                    f'virtualSheath={self.enable_native_virtual_sheath} '
                    f'thrustReaction={self._native_virtual_sheath_last_reaction_n:.4e} N '
                    f'thrustPaused={self._native_virtual_sheath_paused} '
                    f'pauseReason={self._native_virtual_sheath_pause_reason or "none"} '
                    f'supportCorridorOccupancy={support_occupancy} '
                    f'driveWindowOccupancy={drive_occupancy} '
                    f'lumenClearance={self._native_strict_min_lumen_clearance_mm():.4f} mm '
                    f'barrierNodes={self._native_strict_barrier_active_node_count()} '
                    f'barrierForce={np.round(barrier_vector, 6).tolist()} '
                    f'headStretch={self._native_strict_max_head_stretch():.4e} '
                    f'magMoment={np.round(moment_vector, 6).tolist()} '
                    f'magTorque={np.round(torque_vector, 6).tolist()} '
                    f'magTorqueSin={torque_sin:.4f} '
                    f'magAssist={np.round(assist_vector, 6).tolist()} '
                    f'magAssistOutward={outward_assist:.4e} N '
                    f'magTangentFieldAngle={tangent_field_angle:.3f} deg'
                )
        print(
            f'[guidewire] backend={self.backend_name} step={self.step_count} '
            f'tipSpeed={self.filtered_tip_forward_speed_mm_s:.3f} mm/s '
            f'commandedPush={self.commanded_push_mm:.3f} mm estimatedPush={self.estimated_push_mm:.3f} mm '
            f'wallContact={self.wall_contact_active} clearance={self.wall_contact_clearance_mm:.4f} mm '
            f'steeringMisaligned={self.steering_misaligned} steeringAngle={self.steering_angle_deg:.2f} deg '
            f'compression={self.beam_compression_mm:.3f} mm compressionGuard={self.beam_compression_active} '
            f'stallRegulator={self.beam_stall_active} '
            f'pushScale={self.push_force_scale:.3f} '
            f'pushScaleReason={self._push_scale_reason} '
            f'correction={self.tip_contact_correction_mm:.4f} mm '
            f'tipDir={np.round(_normalize(tip_dir), 3).tolist()} '
            f'baVector={np.round(_normalize(ba_vector), 3).tolist()} '
            f'targetPoint={np.round(target_point, 3).tolist()} '
            f'forceDir={np.round(_normalize(force_vector), 3).tolist()}'
            f'{native_tail}'
        )

    def _native_debug_max_abs(self, data) -> float:
        if data is None:
            return 0.0
        try:
            arr = np.asarray(data.value, dtype=float)
        except Exception:
            return 0.0
        if arr.size == 0:
            return 0.0
        if arr.ndim >= 2 and arr.shape[-1] == 3:
            norms = np.linalg.norm(arr.reshape(-1, 3), axis=1)
            return float(np.max(norms)) if norms.size else 0.0
        return float(np.max(np.abs(arr)))

    def _native_debug_scalar(self, data, default: float = 0.0) -> float:
        if data is None:
            return float(default)
        try:
            return float(data.value)
        except Exception:
            return float(default)

    def _native_debug_int(self, data, default: int = -1) -> int:
        if data is None:
            return int(default)
        try:
            return int(data.value)
        except Exception:
            return int(default)

    def _native_debug_array(self, data) -> np.ndarray:
        if data is None:
            return np.zeros(0, dtype=float)
        try:
            return np.asarray(data.value, dtype=float).reshape(-1)
        except Exception:
            return np.zeros(0, dtype=float)

    def _native_strict_min_lumen_clearance_mm(self) -> float:
        if self._native_strict_min_lumen_clearance_cache_step == self.step_count:
            return float(self._native_strict_min_lumen_clearance_cache)
        native_clearance = self._native_debug_scalar(self._native_debug_min_lumen_clearance_mm)
        surface_clearance = float('inf')
        if (not np.isfinite(native_clearance)) or self._strict_surface_exact_monitor_required(native_clearance):
            surface_clearance = self._surface_min_clearance_mm()
        exact_surface_safe_override = bool(
            np.isfinite(surface_clearance)
            and surface_clearance > 0.20
            and np.isfinite(native_clearance)
            and native_clearance < -self.native_strict_lumen_clamp_tolerance_mm
        )
        if exact_surface_safe_override:
            clearance = float(surface_clearance)
        elif np.isfinite(native_clearance) and np.isfinite(surface_clearance):
            clearance = float(min(native_clearance, surface_clearance))
        elif np.isfinite(surface_clearance):
            clearance = float(surface_clearance)
        elif self.native_strict_barrier_enabled and np.isfinite(native_clearance):
            clearance = float(native_clearance)
        else:
            clearance = float(self.wall_contact_clearance_mm)
        self._native_strict_min_lumen_clearance_cache_step = self.step_count
        self._native_strict_min_lumen_clearance_cache = clearance
        return clearance

    def _native_strict_actual_wall_gap_mm(self) -> float:
        if self.native_strict_barrier_enabled:
            clearance = self._native_strict_min_lumen_clearance_mm()
            if not np.isfinite(clearance):
                return clearance
            return float(clearance + float(ELASTICROD_STRICT_BARRIER_SAFETY_MARGIN_MM))
        head_profile_clearance = self._head_wall_clearance()
        head_surface_clearance = self._head_surface_clearance()
        false_profile_override = self._native_strict_false_profile_contact_clearance_mm(
            head_profile_clearance_mm=float(head_profile_clearance),
            head_surface_clearance_mm=float(head_surface_clearance),
        )
        if false_profile_override is not None:
            return float(false_profile_override)
        surface_trustworthy = self._native_strict_head_surface_contact_is_trustworthy(
            float(head_surface_clearance),
            native_gap_mm=float(head_profile_clearance),
            head_profile_clearance_mm=float(head_profile_clearance),
        )
        if surface_trustworthy and np.isfinite(head_surface_clearance):
            if np.isfinite(head_profile_clearance):
                return float(min(head_profile_clearance, head_surface_clearance))
            return float(head_surface_clearance)
        return float(head_profile_clearance)

    def _native_strict_barrier_active_node_count(self) -> int:
        return max(self._native_debug_int(self._native_debug_barrier_active_node_count, default=0), 0)

    def _native_strict_barrier_active(self) -> bool:
        if self._native_strict_barrier_active_cache_step == self.step_count:
            return bool(self._native_strict_barrier_active_cache)
        active = False
        if self.native_strict_barrier_enabled:
            if self._native_strict_barrier_active_node_count() > 0:
                active = True
            else:
                activation_margin = float(
                    max(
                        ELASTICROD_STRICT_BARRIER_SAFETY_MARGIN_MM + 0.08,
                        min(ELASTICROD_STRICT_BARRIER_ACTIVATION_MARGIN_MM, 0.26),
                    )
                )
                native_clearance = self._native_debug_scalar(
                    self._native_debug_min_lumen_clearance_mm,
                    default=float('inf'),
                )
                if np.isfinite(native_clearance):
                    active = float(native_clearance) <= activation_margin
                elif self._surface_min_clearance_mm() <= activation_margin:
                    active = True
                else:
                    active = self._native_strict_min_lumen_clearance_mm() <= activation_margin
        self._native_strict_barrier_active_cache_step = self.step_count
        self._native_strict_barrier_active_cache = bool(active)
        return bool(active)

    def _native_strict_upcoming_turn_deg(self) -> float:
        return float(max(self._native_debug_scalar(self._debug_upcoming_turn_deg), 0.0))

    def _native_strict_bend_severity(self) -> float:
        return float(np.clip(self._native_debug_scalar(self._debug_bend_severity), 0.0, 1.0))

    def _native_strict_scheduled_field_scale(self) -> float:
        return float(max(self._native_debug_scalar(self._debug_scheduled_field_scale, default=1.0), 0.0))

    def _native_strict_recentering_alpha(self) -> float:
        return float(np.clip(self._native_debug_scalar(self._debug_recentering_alpha), 0.0, 1.0))

    def _native_strict_max_head_stretch(self) -> float:
        return float(max(self._native_debug_scalar(self._native_debug_max_head_stretch), 0.0))

    def _native_strict_head_stretch_limits(self) -> tuple[float, float]:
        soft_limit = float(ELASTICROD_STRICT_HEAD_STRETCH_SOFT_LIMIT)
        hard_limit = float(ELASTICROD_STRICT_HEAD_STRETCH_LIMIT)
        return soft_limit, hard_limit

    def _native_strict_global_stretch_limits(self) -> tuple[float, float]:
        soft_limit = float(ELASTICROD_STRICT_GLOBAL_STRETCH_SOFT_LIMIT)
        hard_limit = float(ELASTICROD_STRICT_GLOBAL_STRETCH_HARD_LIMIT)
        return soft_limit, hard_limit

    def _stop_root_animation(self, reason: str) -> None:
        if self._native_failfast_triggered:
            return
        self._native_failfast_triggered = True
        # strict 婵☆垪鈧磭纭€濞寸姴鎳撻鍥亹?failfast 闁告稑锕ㄩ鐔兼晬鐏炶偐鐟濋柤濂変簻婵晠宕戝鈧挒銏ゆ儑閻旂补鍋?        print(f'[ERROR] [elasticrod-failfast] strict warning (simulation continues): {reason}')

    def _sync_native_rod_to_display(self) -> None:
        """Sync Vec6d physics state 闁?Rigid3d display 闁?visual node positions.

        The C++ ElasticRodRigidStateAdapter declares isMechanical=false,
        so SOFA's mechanical propagation visitors skip it.  We replicate
        the conversion (center m闁愁偅濮, tangent quaternion) AND directly
        push the positions into the visual MechanicalObject children so
        that OglModel renders correctly even if UpdateMappingVisitor
        does not re-apply the RigidMapping after our write.
        """
        if self._rod_state_pos is None:
            return
        rod = np.asarray(_read(self._rod_state_pos), dtype=float)
        n = rod.shape[0]
        if n == 0:
            return
        centers_mm = rod[:, :3] * 1000.0
        quats = np.empty((n, 4), dtype=float)
        if n < 2:
            quats[:] = [0.0, 0.0, 0.0, 1.0]
        else:
            edges = centers_mm[1:] - centers_mm[:-1]
            lens = np.linalg.norm(edges, axis=1, keepdims=True)
            lens = np.maximum(lens, 1e-12)
            tangents = edges / lens
            for i in range(n):
                ei = min(i, n - 2)
                quats[i] = _quat_from_z_to(tangents[ei])
        with _writeable(self._pos) as rigid:
            rigid[:n, :3] = centers_mm
            rigid[:n, 3:7] = quats
        if self._rod_state_vel is not None:
            rod_vel = np.asarray(_read(self._rod_state_vel), dtype=float)
            if rod_vel.ndim == 2 and rod_vel.shape[0] >= n:
                mapped_vel = np.zeros((n, 6), dtype=float)
                cols = min(3, rod_vel.shape[1])
                mapped_vel[:, :cols] = 1000.0 * rod_vel[:n, :cols]
                with _writeable(self._vel) as rigid_vel:
                    rigid_vel[:n, :6] = mapped_vel
        self._invalidate_geometry_cache()

        sync_interval = 80 if self.use_native_gui_wallclock_control else 50
        if self.step_count <= 3 or self.step_count % sync_interval == 0:
            tip = centers_mm[-1] if n else np.zeros(3)
            base = centers_mm[0] if n else np.zeros(3)
            print(
                f'[SYNC] step={self.step_count} n={n} '
                f'base=[{base[0]:.2f},{base[1]:.2f},{base[2]:.2f}] '
                f'tip=[{tip[0]:.2f},{tip[1]:.2f},{tip[2]:.2f}]'
            )

        if (not self._native_visual_sync_targets) and self.step_count <= 3:
            self._refresh_native_visual_sync_targets()
        for child_name, vis_dofs_data, indices, ogl_pos in self._native_visual_sync_targets:
            try:
                count = min(len(indices), n)
                if count <= 0:
                    continue
                vis_points = centers_mm[indices[:count]]
                with _writeable(vis_dofs_data) as arr:
                    arr[:count, :3] = vis_points
                if ogl_pos is not None:
                    with _writeable(ogl_pos) as oarr:
                        oarr[:count, :3] = vis_points
            except Exception as exc:
                if self.step_count <= 3:
                    print(f'[SYNC-WARN] {child_name}: {exc}')

    def _emit_native_diagnostics(self) -> None:
        if not self.is_native_backend:
            return

        positions = self._current_rigid_state()
        velocities = np.asarray(_read(self._vel), dtype=float)
        centers = positions[:, :3] if positions.ndim == 2 and positions.shape[1] >= 3 else np.zeros((0, 3), dtype=float)

        displacements = (
            np.linalg.norm(centers - self._diagnostic_prev_centers, axis=1)
            if centers.shape == self._diagnostic_prev_centers.shape and centers.size
            else np.zeros(0, dtype=float)
        )
        linear_speeds = (
            np.linalg.norm(velocities[:, :3], axis=1)
            if velocities.ndim == 2 and velocities.shape[1] >= 3
            else np.zeros(0, dtype=float)
        )
        angular_speeds = (
            np.linalg.norm(velocities[:, 3:], axis=1)
            if velocities.ndim == 2 and velocities.shape[1] >= 6
            else np.zeros(0, dtype=float)
        )

        max_disp = float(np.max(displacements)) if displacements.size else 0.0
        max_lin_speed = float(np.max(linear_speeds)) if linear_speeds.size else 0.0
        max_ang_speed = float(np.max(angular_speeds)) if angular_speeds.size else 0.0
        finite_state = bool(np.isfinite(positions).all() and np.isfinite(velocities).all())
        startup_scale = self._native_startup_ramp_scale()
        stretch = self._native_debug_array(self._native_debug_stretch)
        twist = self._native_debug_array(self._native_debug_twist)
        max_stretch = float(np.max(np.abs(stretch))) if stretch.size else 0.0
        max_twist = float(np.max(np.abs(twist))) if twist.size else 0.0
        min_lumen_clearance_mm = self._native_strict_min_lumen_clearance_mm()
        barrier_active_nodes = self._native_strict_barrier_active_node_count()
        max_head_stretch = self._native_strict_max_head_stretch()
        if self.wall_contact_active:
            self._native_max_post_contact_jump_mm = max(self._native_max_post_contact_jump_mm, max_disp)

        linear_speed_warn_threshold = float(ELASTICROD_DIAGNOSTIC_LINEAR_SPEED_WARN_MM_S)
        if self.is_native_realtime:
            linear_speed_warn_threshold *= max(float(self._native_runtime_speed_scale), 1.0)
        if self.use_native_gui_wallclock_control:
            linear_speed_warn_threshold = max(
                linear_speed_warn_threshold,
                float(ELASTICROD_GUI_DIAGNOSTIC_LINEAR_SPEED_WARN_MM_S),
            )

        warn = (
            (not finite_state)
            or (max_disp >= ELASTICROD_DIAGNOSTIC_DISPLACEMENT_WARN_MM)
            or (max_lin_speed >= linear_speed_warn_threshold)
            or (max_ang_speed >= ELASTICROD_DIAGNOSTIC_ANGULAR_SPEED_WARN_RAD_S)
        )
        if self.is_native_strict:
            warn = warn or (max_head_stretch > ELASTICROD_STRICT_HEAD_STRETCH_SOFT_LIMIT) or (max_stretch > ELASTICROD_STRICT_GLOBAL_STRETCH_SOFT_LIMIT)
        severe = (
            (not finite_state)
            or (max_disp >= ELASTICROD_RECOVERY_TRIGGER_DISPLACEMENT_MM)
            or (max_stretch >= ELASTICROD_RECOVERY_TRIGGER_MAX_STRETCH)
            or (
                (not self.is_native_strict)
                and (
                    (max_lin_speed >= ELASTICROD_RECOVERY_TRIGGER_LINEAR_SPEED_MM_S)
                    or (max_ang_speed >= ELASTICROD_SAFE_RECOVERY_ANGULAR_SPEED_RAD_S)
                )
            )
            or (
                self.is_native_strict
                and (
                    (max_head_stretch > ELASTICROD_STRICT_HEAD_STRETCH_LIMIT)
                    or (max_stretch > ELASTICROD_STRICT_GLOBAL_STRETCH_HARD_LIMIT)
                )
            )
        )
        in_window = self.step_count <= max(int(ELASTICROD_DIAGNOSTIC_STEP_WINDOW), 0)
        if self.native_diagnostic_realtime:
            if self.use_native_gui_wallclock_control:
                gui_interval = max(int(ELASTICROD_GUI_DIAGNOSTIC_MIN_LOG_INTERVAL_STEPS), 1)
                should_log = severe or self.step_count <= 3 or (self.step_count % gui_interval == 0)
            else:
                should_log = warn or (self.step_count % 250 == 0)
        else:
            should_log = warn or (
                in_window
                and max(int(ELASTICROD_DIAGNOSTIC_PRINT_EVERY), 1) > 0
                and self.step_count % max(int(ELASTICROD_DIAGNOSTIC_PRINT_EVERY), 1) == 0
            )
        stall_warning = (
            self.wall_contact_active
            and (not self._native_stall_logged)
            and self.filtered_tip_forward_speed_mm_s <= 0.25
            and self.commanded_push_mm >= self.tip_progress_mm + 0.5
        )
        need_detail_metrics = bool(should_log or stall_warning or (not self.is_native_strict))
        max_kappa = 0.0
        edge_lengths = np.zeros(0, dtype=float)
        abnormal_edge = -1
        abnormal_edge_length_mm = 0.0
        abnormal_edge_ref_length_mm = 0.0
        max_axial_boundary_error_mm = 0.0
        max_lateral_boundary_error_mm = 0.0
        max_internal_force = 0.0
        max_stretch_force = 0.0
        max_boundary_force = 0.0
        max_boundary_torque = 0.0
        max_bend_residual = 0.0
        barrier_force = np.zeros(3, dtype=float)
        if need_detail_metrics:
            max_kappa = self._native_debug_max_abs(self._native_debug_kappa)
            edge_lengths = self._native_debug_array(self._native_debug_edge_length)
            abnormal_edge = self._native_debug_int(self._native_debug_abnormal_edge_index, default=-1)
            abnormal_edge_length_mm = self._native_debug_scalar(self._native_debug_abnormal_edge_length_mm)
            abnormal_edge_ref_length_mm = self._native_debug_scalar(self._native_debug_abnormal_edge_ref_length_mm)
            max_axial_boundary_error_mm = self._native_debug_scalar(self._native_debug_max_axial_boundary_error)
            max_lateral_boundary_error_mm = self._native_debug_scalar(self._native_debug_max_lateral_boundary_error)
            max_internal_force = self._native_debug_scalar(self._native_debug_max_internal_force)
            max_stretch_force = self._native_debug_scalar(self._native_debug_max_stretch_force)
            max_boundary_force = self._native_debug_scalar(self._native_debug_max_boundary_force)
            max_boundary_torque = self._native_debug_scalar(self._native_debug_max_boundary_torque)
            max_bend_residual = self._native_debug_scalar(self._native_debug_max_bend_residual)
            barrier_force = self._debug_vector(self._native_debug_barrier_force_vector, [0.0, 0.0, 0.0])

        safe_contact_clearance_mm = min(
            float(v)
            for v in (
                self.surface_wall_contact_clearance_mm,
                self.wall_contact_clearance_mm,
            )
            if np.isfinite(float(v))
        ) if any(np.isfinite(float(v)) for v in (self.surface_wall_contact_clearance_mm, self.wall_contact_clearance_mm)) else float('inf')
        distal_edge_compressed = bool(
            edge_lengths.size > 0 and float(edge_lengths[-1]) < 0.94 * float(self.rest_spacing_mm)
        )
        stalled_in_contact = bool(
            self.wall_contact_active
            and self.filtered_tip_forward_speed_mm_s <= 0.80
            and self.commanded_push_mm >= self.tip_progress_mm + 0.35
        )
        severe_distal_head_kink = bool(
            max_stretch >= 0.085
            or max_head_stretch >= 2.5e-3
            or (distal_edge_compressed and max_twist >= 0.035)
        )
        near_surface_scrape = bool(
            safe_contact_clearance_mm <= 0.02
            and (
                max_stretch >= 0.045
                or max_head_stretch >= 9.0e-4
                or distal_edge_compressed
            )
        )
        recent_safe_recovery = bool(
            self.is_native_safe
            and self._native_safe_last_recovery_step >= 0
            and self.step_count - self._native_safe_last_recovery_step <= 4
        )
        distal_head_recovery_override = bool(
            max_stretch >= 5.0e-2
            or max_head_stretch >= 1.0e-2
            or barrier_active_nodes >= 3
            or safe_contact_clearance_mm <= 0.10
        )
        distal_head_recovery = bool(
            self.is_native_safe
            and ELASTICROD_ENABLE_SAFE_RECOVERY
            and (
                self._native_safe_recovery_cooldown <= 0
                or distal_head_recovery_override
            )
            and finite_state
            and self.wall_contact_active
            and safe_contact_clearance_mm >= -0.08
            and (
                severe_distal_head_kink
                or near_surface_scrape
                or (recent_safe_recovery and max_head_stretch >= 6.0e-3)
            )
            and (
                stalled_in_contact
                or safe_contact_clearance_mm <= 0.12
                or max_twist >= 0.08
                or max_head_stretch >= 1.2e-2
                or barrier_active_nodes >= 4
                or recent_safe_recovery
            )
        )
        if distal_head_recovery:
            self._recover_native_safe_distal_head(
                max_stretch=max_stretch,
                max_head_stretch=max_head_stretch,
                contact_clearance_mm=safe_contact_clearance_mm,
                barrier_active_nodes=barrier_active_nodes,
            )
            return

        if should_log:
            level = 'WARN' if warn else 'INFO'
            print(
                f'[{level}] [elasticrod-diag] step={self.step_count} t={self.sim_time_s:.4f}s '
                f'ramp={startup_scale:.3f} finite={finite_state} '
                f'band={self._native_runtime_band} bandDt={self._native_runtime_dt_s:.6f}s '
                f'bandSolver={self._native_runtime_solver_max_iter}/{self._native_runtime_solver_tolerance:.1e} '
                f'bandSpeedScale={self._native_runtime_speed_scale:.2f} '
                f'maxDisp={max_disp:.4f} mm maxLinSpeed={max_lin_speed:.3f} mm/s maxAngSpeed={max_ang_speed:.3f} rad/s '
                f'maxStretch={max_stretch:.4e} maxKappa={max_kappa:.4e} maxTwist={max_twist:.4e} '
                f'maxAxialBoundaryErr={max_axial_boundary_error_mm:.4f} mm maxLateralBoundaryErr={max_lateral_boundary_error_mm:.4f} mm '
                f'maxInternalForce={max_internal_force:.4e} N maxStretchForce={max_stretch_force:.4e} N maxBoundaryForce={max_boundary_force:.4e} N '
                f'maxBoundaryTorque={max_boundary_torque:.4e} N.m maxBendResidual={max_bend_residual:.4e} '
                f'minLumenClearance={min_lumen_clearance_mm:.4f} mm barrierActiveNodes={barrier_active_nodes} '
                f'barrierForce={np.round(barrier_force, 6).tolist()} maxHeadStretch={max_head_stretch:.4e} '
                f'abnormalEdge={abnormal_edge} abnormalEdgeLen={abnormal_edge_length_mm:.4f} mm abnormalEdgeRef={abnormal_edge_ref_length_mm:.4f} mm '
                f'wallContact={self.wall_contact_active} firstContactStep={self._native_first_contact_step} '
                f'postContactJump={self._native_max_post_contact_jump_mm:.4f} mm'
            )
        if stall_warning:
            scheduled_field_scale = self._native_strict_scheduled_field_scale()
            scheduled_field_scale_base = self._native_debug_scalar(
                self._debug_scheduled_field_scale_base,
                default=scheduled_field_scale,
            )
            strict_need_alpha = self._native_debug_scalar(self._debug_strict_steering_need_alpha, default=0.0)
            entry_release_alpha = self._native_debug_scalar(self._debug_entry_release_alpha, default=1.0)
            print(
                f'[WARN] [elasticrod-stall] step={self.step_count} t={self.sim_time_s:.4f}s '
                f'tipSpeed={self.filtered_tip_forward_speed_mm_s:.4f} mm/s '
                f'commandedPush={self.commanded_push_mm:.3f} mm tipProgress={self.tip_progress_mm:.3f} mm '
                f'wallClearance={self.wall_contact_clearance_mm:.4f} mm '
                f'maxBoundaryForce={max_boundary_force:.4e} N maxInternalForce={max_internal_force:.4e} N maxStretchForce={max_stretch_force:.4e} N '
                f'runtimeBand={self._native_runtime_band} runtimeBandReason={self._native_runtime_band_reason} '
                f'pushScale={self.push_force_scale:.3f} pushScaleReason={self._push_scale_reason} '
                f'entryReleaseAlpha={entry_release_alpha:.3f} strictNeedAlpha={strict_need_alpha:.3f} '
                f'scheduledFieldScale={scheduled_field_scale_base:.3f}->{scheduled_field_scale:.3f}'
            )
            self._native_stall_logged = True
        if severe and self.is_native_safe and self.enable_vessel_lumen_constraint and ELASTICROD_ENABLE_SAFE_RECOVERY:
            self._recover_native_safe_state(
                finite_state=finite_state,
                max_disp=max_disp,
                max_lin_speed=max_lin_speed,
                max_ang_speed=max_ang_speed,
                max_stretch=max_stretch,
            )
            return
        if severe and self.is_native_strict and self._native_last_severe_step != self.step_count:
            print(
                f'[WARN] [elasticrod-strict] severe state detected without Python recovery: '
                f'step={self.step_count} maxDisp={max_disp:.4f} mm maxLinSpeed={max_lin_speed:.3f} mm/s '
                f'maxAngSpeed={max_ang_speed:.3f} rad/s maxStretch={max_stretch:.4e}'
            )
            self._native_last_severe_step = self.step_count
        contact_kinematic_blowup = (
            self.wall_contact_active
            and (
                self._native_max_post_contact_jump_mm >= 2.0
                or max_stretch >= 0.10
                or max_twist >= 1.0
            )
            and (
                max_lin_speed >= ELASTICROD_SAFE_RECOVERY_LINEAR_SPEED_MM_S
                or max_ang_speed >= ELASTICROD_SAFE_RECOVERY_ANGULAR_SPEED_RAD_S
            )
        )
        surface_emergency_margin_mm = max(0.5, 2.0 * float(self._contact_radius_mm()))
        if self.is_native_strict:
            failfast = (
                (not finite_state)
                or (max_head_stretch > ELASTICROD_STRICT_HEAD_STRETCH_LIMIT)
                or (max_stretch > ELASTICROD_STRICT_GLOBAL_STRETCH_HARD_LIMIT)
                or (self.native_strict_barrier_enabled and min_lumen_clearance_mm < -surface_emergency_margin_mm)
            )
        else:
            failfast = (
                (not finite_state)
                or (abnormal_edge >= 0 and max_stretch >= ELASTICROD_FAILFAST_EDGE_STRETCH_RATIO)
                or (max_stretch >= ELASTICROD_FAILFAST_MAX_STRETCH)
                or contact_kinematic_blowup
                or (self.native_strict_barrier_enabled and min_lumen_clearance_mm < -surface_emergency_margin_mm)
            )
        if failfast and (edge_lengths.size == 0) and self.is_native_strict:
            edge_lengths = self._native_debug_array(self._native_debug_edge_length)
            abnormal_edge = self._native_debug_int(self._native_debug_abnormal_edge_index, default=-1)
            abnormal_edge_length_mm = self._native_debug_scalar(self._native_debug_abnormal_edge_length_mm)
            abnormal_edge_ref_length_mm = self._native_debug_scalar(self._native_debug_abnormal_edge_ref_length_mm)
            max_axial_boundary_error_mm = self._native_debug_scalar(self._native_debug_max_axial_boundary_error)
            max_lateral_boundary_error_mm = self._native_debug_scalar(self._native_debug_max_lateral_boundary_error)
        if failfast and self.is_native_strict:
            reason = (
                f'step={self.step_count} finite={finite_state} maxStretch={max_stretch:.4e} '
                f'maxLinSpeed={max_lin_speed:.3f} mm/s maxAngSpeed={max_ang_speed:.3f} rad/s '
                f'abnormalEdge={abnormal_edge} edgeLen={abnormal_edge_length_mm:.4f} mm edgeRef={abnormal_edge_ref_length_mm:.4f} mm '
                f'maxAxialBoundaryErr={max_axial_boundary_error_mm:.4f} mm maxLateralBoundaryErr={max_lateral_boundary_error_mm:.4f} mm '
                f'minLumenClearance={min_lumen_clearance_mm:.4f} mm maxHeadStretch={max_head_stretch:.4e}'
            )
            if edge_lengths.size:
                head = np.round(edge_lengths[: min(4, edge_lengths.size)], 4).tolist()
                tail = np.round(edge_lengths[max(0, edge_lengths.size - 4):], 4).tolist()
                reason += f' edgeLenHead={head} edgeLenTail={tail}'
            self._stop_root_animation(reason)
        if centers.size:
            self._diagnostic_prev_centers = centers.copy()
            if self.is_native_safe and ELASTICROD_ENABLE_SAFE_RECOVERY:
                light_contact_snapshot = bool(self.wall_contact_active or barrier_active_nodes > 0)
                if light_contact_snapshot:
                    snapshot_clearance_ok = (not np.isfinite(safe_contact_clearance_mm)) or safe_contact_clearance_mm >= 0.10
                    snapshot_shape_ok = max_stretch <= 2.0e-2 and max_head_stretch <= 1.5e-2
                else:
                    snapshot_clearance_ok = (not np.isfinite(safe_contact_clearance_mm)) or safe_contact_clearance_mm >= 0.05
                    snapshot_shape_ok = max_stretch <= 0.045 and max_head_stretch <= 6.0e-3
                if finite_state and snapshot_clearance_ok and snapshot_shape_ok:
                    self._native_safe_last_stable_pos = positions.copy()
                    self._native_safe_last_stable_vel = velocities.copy()
                    if self._rod_state_pos is not None:
                        self._native_safe_last_stable_rod_pos = np.array(_read(self._rod_state_pos), dtype=float, copy=True)
                    if self._rod_state_vel is not None:
                        self._native_safe_last_stable_rod_vel = np.array(_read(self._rod_state_vel), dtype=float, copy=True)
                    if self._rod_state_free_pos is not None:
                        self._native_safe_last_stable_rod_free_pos = np.array(_read(self._rod_state_free_pos), dtype=float, copy=True)
                    if self._rod_state_free_vel is not None:
                        self._native_safe_last_stable_rod_free_vel = np.array(_read(self._rod_state_free_vel), dtype=float, copy=True)

    def _recover_native_safe_state(
        self,
        *,
        finite_state: bool,
        max_disp: float,
        max_lin_speed: float,
        max_ang_speed: float,
        max_stretch: float,
    ) -> None:
        if (
            self._native_safe_last_stable_pos is None
            or self._native_safe_last_stable_vel is None
            or self._native_safe_last_stable_rod_pos is None
        ):
            return

        restored_pos = np.array(self._native_safe_last_stable_pos, dtype=float, copy=True)
        restored_vel = np.zeros_like(self._native_safe_last_stable_vel, dtype=float)
        restored_rod_pos = np.array(self._native_safe_last_stable_rod_pos, dtype=float, copy=True)
        restored_rod_vel = (
            np.zeros_like(self._native_safe_last_stable_rod_vel, dtype=float)
            if self._native_safe_last_stable_rod_vel is not None
            else np.zeros_like(restored_rod_pos, dtype=float)
        )
        restored_rod_free_pos = (
            np.array(self._native_safe_last_stable_rod_free_pos, dtype=float, copy=True)
            if self._native_safe_last_stable_rod_free_pos is not None
            else restored_rod_pos.copy()
        )
        restored_rod_free_vel = (
            np.zeros_like(self._native_safe_last_stable_rod_free_vel, dtype=float)
            if self._native_safe_last_stable_rod_free_vel is not None
            else restored_rod_vel.copy()
        )

        with _writeable(self._rod_state_pos) as rod_pos:
            rod_pos[:] = restored_rod_pos
        if self._rod_state_vel is not None:
            with _writeable(self._rod_state_vel) as rod_vel:
                rod_vel[:] = restored_rod_vel
        if self._rod_state_free_pos is not None:
            with _writeable(self._rod_state_free_pos) as rod_free_pos:
                rod_free_pos[:] = restored_rod_free_pos
        if self._rod_state_free_vel is not None:
            with _writeable(self._rod_state_free_vel) as rod_free_vel:
                rod_free_vel[:] = restored_rod_free_vel

        try:
            self.rod_model.reinit()
        except Exception as exc:
            print(f'[WARN] [elasticrod-safe] rod reinit after recovery failed: {exc}')

        self._sync_native_rod_to_display()
        with _writeable(self._vel) as vel:
            vel[:] = restored_vel
        self._invalidate_geometry_cache()
        self._invalidate_surface_probe_cache()

        hard_recovery = (
            (not finite_state)
            or (max_stretch >= 0.15)
            or (max_disp >= 2.0)
            or (max_lin_speed >= 5.0 * ELASTICROD_RECOVERY_TRIGGER_LINEAR_SPEED_MM_S)
        )
        retract_mm = (
            float(ELASTICROD_SAFE_RECOVERY_RETRACT_MM)
            if ((not finite_state) or (max_stretch >= ELASTICROD_FAILFAST_MAX_STRETCH))
            else 0.0
        )
        cooldown_steps = int(ELASTICROD_SAFE_RECOVERY_COOLDOWN_STEPS) if hard_recovery else 0
        self.commanded_push_mm = float(max(self.commanded_push_mm - retract_mm, 0.0))
        self._native_safe_recovery_cooldown = max(cooldown_steps, 0)
        self._native_safe_last_recovery_step = self.step_count
        self._native_safe_last_recovery_kind = 'full'
        self._native_safe_distal_recovery_streak = 0
        if self._native_commanded_insertion is not None:
            self._native_commanded_insertion.value = float(self.commanded_push_mm)
        if self._native_commanded_twist is not None:
            self._native_commanded_twist.value = 0.0

        tip_pos = restored_pos[-1, :3].copy()
        self.prev_tip_pos = tip_pos
        _, self.prev_tip_proj_s_mm = self._project_to_centerline(tip_pos)
        self._diagnostic_prev_centers = restored_pos[:, :3].copy()
        self._native_safe_last_stable_pos = restored_pos
        self._native_safe_last_stable_vel = restored_vel
        self._native_safe_last_stable_rod_pos = restored_rod_pos
        self._native_safe_last_stable_rod_vel = restored_rod_vel
        self._native_safe_last_stable_rod_free_pos = restored_rod_free_pos
        self._native_safe_last_stable_rod_free_vel = restored_rod_free_vel
        print(
            '[WARN] [elasticrod-safe] numerical recovery triggered: '
            f'finite={finite_state} maxDisp={max_disp:.4f} mm maxLinSpeed={max_lin_speed:.3f} mm/s '
            f'maxAngSpeed={max_ang_speed:.3f} rad/s maxStretch={max_stretch:.4e} '
            f'mode={"hard" if hard_recovery else "soft"} '
            f'cooldownSteps={self._native_safe_recovery_cooldown} commandedPush={self.commanded_push_mm:.3f} mm'
        )

    def _recover_native_safe_distal_head(
        self,
        *,
        max_stretch: float,
        max_head_stretch: float,
        contact_clearance_mm: float,
        barrier_active_nodes: int,
    ) -> None:
        if (
            self._native_safe_last_stable_rod_pos is None
            or self._rod_state_pos is None
        ):
            return

        rod_pos = np.array(_read(self._rod_state_pos), dtype=float, copy=True)
        stable_rod_pos = np.array(self._native_safe_last_stable_rod_pos, dtype=float, copy=True)
        if rod_pos.ndim != 2 or stable_rod_pos.ndim != 2 or rod_pos.shape[0] == 0:
            return

        n = min(int(rod_pos.shape[0]), int(stable_rod_pos.shape[0]))
        head_node_count = max(self.magnetic_head_edge_count + 3, 5)
        head_start = max(n - head_node_count, 0)
        recent_recovery_steps = (
            int(self.step_count - self._native_safe_last_recovery_step)
            if self._native_safe_last_recovery_step >= 0 and self.step_count >= self._native_safe_last_recovery_step
            else 10**9
        )
        repeated_recovery = recent_recovery_steps <= 6
        if self._native_safe_last_recovery_kind == 'distal' and recent_recovery_steps <= 2:
            self._native_safe_distal_recovery_streak += 1
        else:
            self._native_safe_distal_recovery_streak = 1
        persistent_recovery = self._native_safe_distal_recovery_streak >= 5
        severe_recovery = bool(
            max_head_stretch >= 1.2e-2
            or max_stretch >= 5.5e-2
            or barrier_active_nodes >= 4
            or contact_clearance_mm < 0.02
        )
        extreme_recovery = bool(
            max_head_stretch >= 2.0e-2
            or max_stretch >= 7.5e-2
            or barrier_active_nodes >= 5
            or contact_clearance_mm < -0.02
        )
        escalate_to_full_recovery = bool(
            repeated_recovery
            and self._native_safe_distal_recovery_streak >= 6
            and (
                (max_stretch >= 0.16 and contact_clearance_mm < 0.05)
                or max_head_stretch >= 3.8e-2
                or barrier_active_nodes >= 10
                or contact_clearance_mm < 0.0
            )
        )
        if (
            escalate_to_full_recovery
            and self._native_safe_last_stable_pos is not None
            and self._native_safe_last_stable_rod_pos is not None
        ):
            print(
                '[INFO] [elasticrod-safe] escalating distal recovery to full-state restore: '
                f'step={self.step_count} streak={self._native_safe_distal_recovery_streak} '
                f'contactClearance={contact_clearance_mm:.4f} mm '
                f'maxStretch={max_stretch:.4e} maxHeadStretch={max_head_stretch:.4e}'
            )
            self._recover_native_safe_state(
                finite_state=True,
                max_disp=0.0,
                max_lin_speed=0.0,
                max_ang_speed=0.0,
                max_stretch=max_stretch,
            )
            return
        proximal_extension = 1
        if repeated_recovery or barrier_active_nodes >= 3:
            proximal_extension += 2
        if severe_recovery:
            proximal_extension += 2
        if extreme_recovery:
            proximal_extension += 1
        if persistent_recovery:
            proximal_extension += 2
        blend_start = max(head_start - 1 - proximal_extension, 0)
        stretch_profile = self._native_debug_array(self._native_debug_stretch)
        if stretch_profile.size:
            hot_edge = int(np.argmax(np.abs(stretch_profile)))
            hot_edge_stretch = float(abs(stretch_profile[hot_edge]))
            if hot_edge_stretch >= 1.5e-2:
                hot_zone_start = max(min(hot_edge, n - 2) - 3, 0)
                if repeated_recovery:
                    hot_zone_start = max(hot_zone_start - 2, 0)
                if severe_recovery:
                    hot_zone_start = max(hot_zone_start - 1, 0)
                if persistent_recovery:
                    hot_zone_start = max(hot_zone_start - 2, 0)
                blend_start = min(blend_start, hot_zone_start)

        rod_vel = (
            np.array(_read(self._rod_state_vel), dtype=float, copy=True)
            if self._rod_state_vel is not None
            else None
        )
        rod_free_pos = (
            np.array(_read(self._rod_state_free_pos), dtype=float, copy=True)
            if self._rod_state_free_pos is not None
            else None
        )
        rod_free_vel = (
            np.array(_read(self._rod_state_free_vel), dtype=float, copy=True)
            if self._rod_state_free_vel is not None
            else None
        )

        for idx in range(blend_start, n):
            if idx < head_start:
                proximal_alpha = float((idx - blend_start) / max(head_start - blend_start, 1))
                proximal_blend_lo = 0.24
                proximal_blend_hi = 0.42
                if repeated_recovery:
                    proximal_blend_lo += 0.08
                    proximal_blend_hi += 0.10
                if severe_recovery:
                    proximal_blend_lo += 0.06
                    proximal_blend_hi += 0.08
                if extreme_recovery:
                    proximal_blend_lo += 0.04
                    proximal_blend_hi += 0.05
                if persistent_recovery:
                    proximal_blend_lo += 0.06
                    proximal_blend_hi += 0.08
                blend = float(np.clip(
                    proximal_blend_lo + (proximal_blend_hi - proximal_blend_lo) * proximal_alpha,
                    0.0,
                    0.78,
                ))
            else:
                alpha = float((idx - head_start) / max((n - 1) - head_start, 1))
                distal_blend_lo = 0.45
                if repeated_recovery:
                    distal_blend_lo += 0.08
                if severe_recovery:
                    distal_blend_lo += 0.08
                if extreme_recovery:
                    distal_blend_lo += 0.05
                if persistent_recovery:
                    distal_blend_lo += 0.08
                blend = float(np.clip(distal_blend_lo + (1.0 - distal_blend_lo) * alpha, 0.0, 1.0))
            rod_pos[idx, :3] = (1.0 - blend) * rod_pos[idx, :3] + blend * stable_rod_pos[idx, :3]
            if rod_free_pos is not None and idx < rod_free_pos.shape[0]:
                stable_free = (
                    np.array(self._native_safe_last_stable_rod_free_pos, dtype=float, copy=False)
                    if self._native_safe_last_stable_rod_free_pos is not None
                    else stable_rod_pos
                )
                source = stable_free[idx, :3] if idx < stable_free.shape[0] else stable_rod_pos[idx, :3]
                rod_free_pos[idx, :3] = (1.0 - blend) * rod_free_pos[idx, :3] + blend * source
            if rod_vel is not None and idx < rod_vel.shape[0]:
                rod_vel[idx, :] = 0.0
            if rod_free_vel is not None and idx < rod_free_vel.shape[0]:
                rod_free_vel[idx, :] = 0.0

        with _writeable(self._rod_state_pos) as out_pos:
            out_pos[:] = rod_pos
        if self._rod_state_free_pos is not None and rod_free_pos is not None:
            with _writeable(self._rod_state_free_pos) as out_free_pos:
                out_free_pos[:] = rod_free_pos
        if self._rod_state_vel is not None and rod_vel is not None:
            with _writeable(self._rod_state_vel) as out_vel:
                out_vel[:] = rod_vel
        if self._rod_state_free_vel is not None and rod_free_vel is not None:
            with _writeable(self._rod_state_free_vel) as out_free_vel:
                out_free_vel[:] = rod_free_vel

        try:
            self.rod_model.reinit()
        except Exception as exc:
            print(f'[WARN] [elasticrod-safe] distal-head reinit failed: {exc}')

        rollback_mm = 0.0
        self.commanded_push_mm = float(max(self.commanded_push_mm - rollback_mm, 0.0))
        if self._native_commanded_insertion is not None:
            self._native_commanded_insertion.value = float(self.commanded_push_mm)
        if self._native_commanded_twist is not None:
            self._native_commanded_twist.value = 0.0
        cooldown_steps = 1
        if repeated_recovery or severe_recovery:
            cooldown_steps = max(cooldown_steps, 2)
        if extreme_recovery:
            cooldown_steps = max(cooldown_steps, 3)
        if persistent_recovery:
            cooldown_steps = max(cooldown_steps, 4)
        self._native_safe_recovery_cooldown = max(self._native_safe_recovery_cooldown, cooldown_steps)
        self._native_safe_last_recovery_step = self.step_count
        self._native_safe_last_recovery_kind = 'distal'
        self._sync_native_rod_to_display()
        if self._vel is not None:
            rigid_vel = np.array(_read(self._vel), dtype=float, copy=True)
            if rigid_vel.ndim == 2 and rigid_vel.shape[0] >= n:
                rigid_vel[blend_start:n, :] = 0.0
                with _writeable(self._vel) as out_rigid_vel:
                    out_rigid_vel[:] = rigid_vel
        self._invalidate_geometry_cache()
        self._invalidate_surface_probe_cache()
        print(
            '[WARN] [elasticrod-safe] distal-head recovery triggered: '
            f'step={self.step_count} contactClearance={contact_clearance_mm:.4f} mm '
            f'maxStretch={max_stretch:.4e} maxHeadStretch={max_head_stretch:.4e} '
            f'headNodes={list(range(head_start, n))} barrierNodes={int(barrier_active_nodes)} '
            f'rollback={rollback_mm:.3f} mm cooldown={cooldown_steps} repeated={repeated_recovery} '
            f'streak={self._native_safe_distal_recovery_streak} '
            f'commandedPush={self.commanded_push_mm:.3f} mm'
        )

    def onAnimateBeginStep(self, dt):
        raise NotImplementedError('Use BeamGuidewireController or ElasticRodGuidewireController instead of the shared base class.')

    def onAnimateBeginEvent(self, event):
        self.onAnimateBeginStep(float(event.get('dt', 0.0)) if isinstance(event, dict) else float(self.getContext().dt.value))

    def onBeginAnimationStep(self, dt):
        self.onAnimateBeginStep(float(dt))

    def _update_camera_after_solve(self) -> None:
        tip_pos, _ = self._tip_pose()
        self._update_camera_follow(tip_pos)

    def onAnimateEndEvent(self, event):
        self._update_camera_after_solve()

    def onEndAnimationStep(self, dt):
        self._update_camera_after_solve()


class BeamGuidewireController(GuidewireControllerBase):
    def onAnimateBeginStep(self, dt):
        dt = float(dt)
        self.step_count += 1
        self._update_estimated_push_mm()
        tip_pos, tip_quat = self._tip_pose()
        self._update_tip_speed(tip_pos, dt)
        self._update_wall_contact_state()
        tip_dir = self._tip_dir(tip_quat)
        target_point, ba_vector, force_vector = self._sync_debug_visuals(tip_pos, tip_dir)
        self._update_camera_follow(tip_pos)
        self._update_steering_state(
            tip_pos,
            tip_dir,
            target_point,
            desired_dir=ba_vector if self.is_native_backend else None,
        )
        self._update_beam_compression_state()
        self._update_beam_stall_state()
        self._update_smoothed_push_scale(dt)
        if self.use_kinematic_beam_insertion:
            self._advance_commanded_push(dt)
            self._constrain_wire()
            self._update_estimated_push_mm()
        else:
            # 闁绘せ鏅濋幃?beam 閻犱警鍨扮欢鐐存媴鐠恒劍鏆忛柟顓熷笒閻ｇ偓娼婚幋鐙€浼傞柟鎭掑妼婵繘鏁嶇仦鑲╃憹闁告劕绉存禒娑㈠灳濠娾偓鐠愮喐娼婚悾灞剧獥闁哄秴娲埀顒傚枎鐎规娊鎳撳畝鍐ㄦ闁告柣鍔忛惃鐔煎礉濞戞巻鍋撳┑鍫熺暠闁硅矇鍐ㄧ厬閻炴稏鍎扮粩鐢稿Υ?            self.push_force_calibrated = True
            self.commanded_push_mm = self.drive_push_mm
        self._update_push_force(dt)
        self._update_estimated_push_mm()
        self.prev_tip_pos = tip_pos.copy()
        self._log_step_state(tip_dir, target_point, ba_vector, force_vector)

    def _update_camera_after_solve(self) -> None:
        if (not self.use_kinematic_beam_insertion) and self.use_beam_safety_projection:
            # beam 闁绘せ鏅濋幃濠勬崉椤栨氨绐為柟璺猴龚閸樹即宕橀崨顔碱潓鐟滄媽浜簺闁告帞澧楅惇鎵喆閿濆懏鍊甸柛蹇旂矊缁ㄦ娊鏁?            # 闁稿繐鐗愰鈧柣顏嗗枎閻ゅ嫬顫忔担绋款潝閻?+ 闁规亽鍎磋缂佹拝闄勫顐も偓鐟版湰閸ㄦ碍绋夐埀顒€顫㈤妷锔界函闁哄倸搴滅槐婵嬪礃瀹ュ懎娑уǎ鍥跺枟椤掓粌顔忛懠顒傜梾闁哄嫬瀛╁Ο澶愬礄妤﹀灝骞囬柣銊ュ婵☆參鎮欓獮搴撳亾?            self._constrain_wire()
            self._update_estimated_push_mm()
        super()._update_camera_after_solve()


class ElasticRodGuidewireController(GuidewireControllerBase):
    def onAnimateBeginStep(self, dt):
        solver_dt = float(dt)
        control_dt = self._native_control_dt(solver_dt)
        # GUI wall-clock mode should accelerate insertion commands for both
        # native paths. Strict still keeps its magnetic ramp tied to solver time
        # via `externalControlDt=0`, but insertion itself should not be stuck on
        # the tiny contact-band solver dt in runSofa.
        actuation_dt = control_dt
        self.step_count += 1
        self.sim_time_s += max(actuation_dt, 0.0)
        if self.use_native_entry_push_band:
            self._native_entry_push_indices()
        self._update_estimated_push_mm()
        tip_pos, tip_quat = self._tip_pose()
        self._update_tip_speed(tip_pos, solver_dt)
        self._update_wall_contact_state()
        tip_dir = self._tip_dir(tip_quat)
        target_point, ba_vector, force_vector = self._sync_debug_visuals(tip_pos, tip_dir)
        self._update_steering_state(
            tip_pos,
            tip_dir,
            target_point,
            desired_dir=ba_vector if self.is_native_backend else None,
        )
        self._update_native_runtime_profile()
        self._update_smoothed_push_scale(actuation_dt)
        self._advance_commanded_push(actuation_dt)
        self._update_displacement_push(actuation_dt)
        self._update_estimated_push_mm()
        self._write_native_backend_commands()
        if not self._force_diag_printed:
            effective_push_speed = float(self.push_force_target_speed_mm_s)
            startup_ramp_s = float(ELASTICROD_ACTIVE_STARTUP_RAMP_TIME_S)
            if self.use_native_gui_wallclock_control:
                effective_push_speed *= float(max(ELASTICROD_GUI_WALLCLOCK_PUSH_SPEED_SCALE, 1.0))
                startup_ramp_s = float(max(ELASTICROD_GUI_WALLCLOCK_STARTUP_RAMP_S, 0.0))
            if self.is_native_strict:
                if self.native_strict_boundary_driver_enabled:
                    support_occupancy, drive_occupancy = self._native_strict_support_stats()
                    print(
                        f'[INFO] Strict elasticrod insertion enabled: nativeBoundaryDriver=True, '
                        f'commandedInsertionOnly=True, externalPushNodes=[], '
                        f'externalSupportLength={self.external_support_length_mm:.3f} mm, '
                        f'externalSupportRadius={self.external_support_radius_mm:.3f} mm, '
                        f'supportCorridorOccupancy={support_occupancy}, driveWindowOccupancy={drive_occupancy}, '
                        f'commandedSpeed={effective_push_speed:.3f} mm/s, navigationMode={self.navigation_mode}, '
                        f'stabilizationMode={ELASTICROD_STABILIZATION_MODE}, '
                        f'runtimeProfile={ELASTICROD_RUNTIME_PROFILE}, startupRamp={startup_ramp_s:.3f} s, '
                        f'guiWallclockControl={self.use_native_gui_wallclock_control}'
                    )
                else:
                    external_push_nodes = [int(i) for i in self._strict_hand_push_indices()]
                    if self.use_native_displacement_feed:
                        print(
                            f'[INFO] Strict elasticrod insertion enabled: directTailFeed=True, '
                            f'externalPushNodes={external_push_nodes}, '
                            f'externalSupportLength={self.external_support_length_mm:.3f} mm, '
                            f'externalSupportRadius={self.external_support_radius_mm:.3f} mm, '
                            f'pushForce=0.000/0.000 N, '
                            f'commandedSpeed={effective_push_speed:.3f} mm/s, navigationMode={self.navigation_mode}, '
                            f'stabilizationMode={ELASTICROD_STABILIZATION_MODE}, '
                            f'runtimeProfile={ELASTICROD_RUNTIME_PROFILE}, startupRamp={startup_ramp_s:.3f} s, '
                            f'guiWallclockControl={self.use_native_gui_wallclock_control}'
                        )
                    else:
                        print(
                            f'[INFO] Strict elasticrod insertion enabled: externalPushNodes={external_push_nodes}, '
                            f'externalSupportLength={self.external_support_length_mm:.3f} mm, '
                            f'externalSupportRadius={self.external_support_radius_mm:.3f} mm, '
                            f'pushForce={ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_N:.3f}/{ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_MAX_N:.3f} N, '
                            f'commandedSpeed={effective_push_speed:.3f} mm/s, navigationMode={self.navigation_mode}, '
                            f'stabilizationMode={ELASTICROD_STABILIZATION_MODE}, '
                            f'runtimeProfile={ELASTICROD_RUNTIME_PROFILE}, startupRamp={startup_ramp_s:.3f} s, '
                            f'guiWallclockControl={self.use_native_gui_wallclock_control}'
                        )
            else:
                tail_push_indices = (
                    self._native_entry_push_indices()
                    if self.use_native_entry_push_band
                    else [int(i) for i in self._strict_hand_push_indices()]
                )
                native_drive_window_nodes = [int(i) for i in self.native_drive_window_indices]
                print(
                    f'[INFO] Native elasticrod insertion enabled: driveNodes={self.drive_reference_indices}, '
                    f'nativeDriveWindowNodes={native_drive_window_nodes}, '
                    f'tailPushNodes={tail_push_indices}, '
                    f'initialEntrySupportNodes={self.native_support_indices}, '
                    f'commandedSpeed={effective_push_speed:.3f} mm/s, navigationMode={self.navigation_mode}, '
                    f'stabilizationMode={ELASTICROD_STABILIZATION_MODE}, '
                    f'runtimeProfile={ELASTICROD_RUNTIME_PROFILE}, '
                    f'lumenSafetyProjection={self.enable_vessel_lumen_constraint}, '
                    f'strictGuard={self.enable_native_strict_postsolve_guard} '
                    f'(lumenClamp={self.enable_native_strict_lumen_clamp}, speedCap={self.native_strict_max_linear_speed_mm_s:.1f} mm/s), '
                    f'safeRecovery={self.is_native_safe and ELASTICROD_ENABLE_SAFE_RECOVERY}, '
                    f'kinematicSheath={self.native_kinematic_sheath_driver}, '
                    f'virtualSheath={self.enable_native_virtual_sheath}({len(self._native_virtual_sheath_indices)} nodes, {ELASTICROD_SHEATH_LENGTH_MM:.1f} mm), '
                    f'thrustLimit={self.enable_native_thrust_limit}({self.native_thrust_force_n:.3f} N), '
                    f'introducer={ELASTICROD_ENABLE_INTRODUCER}({ELASTICROD_INTRODUCER_LENGTH_MM:.2f} mm), '
                    f'startupRamp={startup_ramp_s:.3f} s, '
                    f'guiWallclockControl={self.use_native_gui_wallclock_control}'
                )
                if self.enable_native_strict_postsolve_guard:
                    print('[INFO] elasticrod strict post-solve guard enabled: monitor-only native barrier verification + soft recovery active.')
                elif not self.enable_vessel_lumen_constraint:
                    print('[WARN] elasticrod strict mode enabled: post-solve lumen safety projection is disabled.')
                wiring_warning = self._native_strict_wiring_warning()
                if wiring_warning:
                    print(f'[WARN] [elasticrod-strict] wiring mismatch: {wiring_warning}')
            self._force_diag_printed = True
        self._update_push_force(actuation_dt if self.is_native_strict else control_dt)
        self._update_native_axial_path_assist_force()
        self.prev_tip_pos = tip_pos.copy()
        self._log_step_state(tip_dir, target_point, ba_vector, force_vector)

    def _update_camera_after_solve(self) -> None:
        strict_corrected = 0
        strict_clipped = 0
        if self.enable_native_strict_postsolve_guard:
            strict_corrected, strict_clipped = self._apply_native_strict_postsolve_guard()
        self._sync_native_rod_to_display()
        if strict_corrected > 0 or strict_clipped > 0:
            self._update_wall_contact_state()
        if self.enable_vessel_lumen_constraint:
            self._constrain_wire()
            self._update_estimated_push_mm()
            self._update_wall_contact_state()
            self._emit_native_diagnostics()
            super()._update_camera_after_solve()
            return
            # `elasticrod` 闁告艾娴烽顒勫箮婵犲懎骞囬柛鎰噹閻ｃ劑宕楅妸锕€顫岀憸鎷岄哺閺備線宕氶幍鏂ュ亾濠婂嫮婀撮悷娆欑到閹宕楀鍐亢闁炽儲绻堝Ο浣糕枔绾板绐?            # 闁稿繐鐗愰鈧柛妯煎枔閺佹捇寮堕崱鏇犵Ъ闁?SOFA 闁规亽鍎磋闂佺偓鍎抽悾顒勫箣閹邦亞顏辨繛鍡磿濠€锛勨偓鍦仜婵繒鈧冻闄勫ú鍧楀棘鐢喚绀?            # 闁告劕绉磋ぐ褏鈧數鎳撻崙锛勭磼韫囨梹顫栭柡鍕劤閸ゎ參鎳欓弮鍌涚暠闁煎搫鍊婚崑锝夊磻濮橆厽浠橀悘蹇撶箣閹便劌顫㈤敐蹇曠闂侇剙鐏濋崢銈囨嫻閺夋埈姊块柡鍐煐閺嗙喖宕愭總鍓叉＇闁轰緤绲婚埀?            self._constrain_wire()
            self._update_estimated_push_mm()
        elif strict_corrected > 0 or strict_clipped > 0:
            self._update_estimated_push_mm()
        self._emit_native_diagnostics()
        super()._update_camera_after_solve()


# 闁稿繒鍘ч鎰板籍瑜嶉閬嶅礂閵夈儲鍊抽柨娑欏哺缁垳鎷嬮妶鍫㈩€€闂傚懎绻愮紞瀣礈瀹ュ甯崇紓鍐惧櫍閸ｇ兘鎯冮崟顐ｅ€电紒鏃戝灦閳ь剙顦扮€氥劑濡?GuidewireNavigationController = ElasticRodGuidewireController if GUIDEWIRE_BACKEND == 'elasticrod' else BeamGuidewireController

