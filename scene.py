# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from .config import (
    ALLOW_PLUGIN_MISSING_FALLBACK,
    BEAM_ACTIVE_DT_S,
    BEAM_ACTIVE_NODE_COUNT,
    BEAM_ACTIVE_PUSH_SPEED_MM_S,
    BEAM_PYTHON_MAGNETIC_FIELD_STRENGTH,
    BEAM_PYTHON_MAGNETIC_FORCE_GAIN,
    BEAM_PYTHON_MAGNETIC_MOMENT,
    BEAM_MAGNETIC_STRENGTH_SCALE,
    BEAM_RUNTIME_PROFILE,
    BEAM_VESSEL_QUERY_FACE_CANDIDATE_COUNT,
    BEAM_ENABLE_LEGACY_DRIVE_CONSTRAINT,
    BEAM_USE_KINEMATIC_INSERTION,
    BEAM_RAYLEIGH_MASS,
    BEAM_RAYLEIGH_STIFFNESS,
    BEAM_ENABLE_SELF_COLLISION,
    CONTACT_ALARM_DISTANCE_MM,
    CONTACT_MANAGER_RESPONSE_PARAMS,
    CONTACT_DISTANCE_MM,
    CONSTRAINT_SOLVER_MAX_ITER,
    CONSTRAINT_SOLVER_TOLERANCE,
    DEFAULT_CAMERA_ROUTE_FOCUS_FRACTION,
    DEFAULT_CAMERA_ROUTE_FOCUS_WEIGHT,
    DISTAL_FORCE_NODE_COUNT,
    DISTAL_VISUAL_NODE_COUNT,
    ELASTICROD_DISTAL_VISUAL_NODE_COUNT,
    ELASTICROD_MAGNETIC_HEAD_EDGES,
    ELASTICROD_ACTIVE_DT_S,
    ELASTICROD_ACTIVE_NODE_COUNT,
    ELASTICROD_ACTIVE_SOLVER_MAX_ITER,
    ELASTICROD_ACTIVE_SOLVER_TOLERANCE,
    ELASTICROD_RUNTIME_PROFILE,
    ELASTICROD_RUNTIME_GUI_NODE_BUDGET,
    ELASTICROD_DT_S,
    ELASTICROD_DISTRIBUTED_TRANSLATIONAL_DAMPING_N_S_PER_M,
    ELASTICROD_DISTRIBUTED_TWIST_DAMPING_NM_S_PER_RAD,
    ELASTICROD_PROXIMAL_ANGULAR_DAMPING_NM_S_PER_RAD,
    ELASTICROD_PROXIMAL_ANGULAR_STIFFNESS_NM_PER_RAD,
    ELASTICROD_PROXIMAL_AXIAL_STIFFNESS_N_PER_M,
    ELASTICROD_PROXIMAL_LATERAL_STIFFNESS_N_PER_M,
    ELASTICROD_PROXIMAL_LINEAR_DAMPING_N_S_PER_M,
    ELASTICROD_ENABLE_AXIAL_PATH_ASSIST,
    ELASTICROD_AXIAL_PATH_ASSIST_DEFICIT_MM,
    ELASTICROD_AXIAL_PATH_ASSIST_FORCE_N,
    ELASTICROD_PUSH_NODE_COUNT,
    ELASTICROD_RAYLEIGH_MASS,
    ELASTICROD_RAYLEIGH_STIFFNESS,
    ELASTICROD_ENABLE_LUMEN_SAFETY_PROJECTION,
    ELASTICROD_ENABLE_SELF_COLLISION,
    ELASTICROD_ENABLE_INTRODUCER,
    ELASTICROD_ENTRY_AXIS_SAMPLE_LENGTH_MM,
    ELASTICROD_INITIAL_TIP_INSERTION_MM,
    ELASTICROD_ENABLE_THRUST_LIMIT,
    ELASTICROD_ENABLE_VIRTUAL_SHEATH,
    ELASTICROD_USE_KINEMATIC_SHEATH_DRIVER,
    ELASTICROD_VESSEL_QUERY_FACE_CANDIDATE_COUNT,
    ELASTICROD_ENABLE_FIELD_GRADIENT,
    ELASTICROD_CONTACT_ALARM_DISTANCE_MM,
    ELASTICROD_COLLISION_DECIMATION_STRIDE,
    ELASTICROD_CONTACT_DISTANCE_MM,
    ELASTICROD_CONTACT_MANAGER_RESPONSE_PARAMS,
    ELASTICROD_CONTACT_OUTER_RADIUS_MM,
    ELASTICROD_CONSTRAINT_SOLVER_MAX_ITER,
    ELASTICROD_CONSTRAINT_SOLVER_TOLERANCE,
    ELASTICROD_AXIAL_DRIVE_NODE_COUNT,
    ELASTICROD_ENTRY_PUSH_BAND_LENGTH_MM,
    ELASTICROD_ENTRY_PUSH_BAND_MIN_NODE_COUNT,
    ELASTICROD_ENTRY_PUSH_BAND_OUTSIDE_OFFSET_MM,
    ELASTICROD_GUIDEWIRE_CONTACT_STIFFNESS,
    ELASTICROD_INTRODUCER_CLEARANCE_MM,
    ELASTICROD_INTRODUCER_LENGTH_MM,
    ELASTICROD_INTRODUCER_RADIAL_SEGMENTS,
    ELASTICROD_INTRODUCER_VISUAL_RGBA,
    ELASTICROD_MAGNETIC_LATERAL_FORCE_SCALE,
    ELASTICROD_MAGNETIC_LOOKAHEAD_DISTANCE_MM,
    ELASTICROD_MAGNETIC_FIELD_SMOOTHING_ALPHA,
    ELASTICROD_MAGNETIC_MAX_TURN_ANGLE_DEG,
    ELASTICROD_MAGNETIC_FIELD_RAMP_TIME_S,
    ELASTICROD_NATIVE_MAGNETIC_MIN_TORQUE_SIN,
    ELASTICROD_MAGNETIC_BR_VECTOR,
    ELASTICROD_MAGNETIC_STRENGTH_SCALE,
    ELASTICROD_STRICT_MAGNETIC_ASSIST,
    ELASTICROD_STRICT_MAGNETIC_PREVIEW_SCALING,
    ELASTICROD_STRICT_PHYSICAL_TORQUE_ONLY,
    ELASTICROD_STRICT_MAGNETIC_RECENTER,
    ELASTICROD_STRICT_TIP_TARGET_FORCE_N,
    ELASTICROD_STRICT_BARRIER_ACTIVATION_MARGIN_MM,
    ELASTICROD_STRICT_BARRIER_DAMPING_N_S_PER_M,
    ELASTICROD_STRICT_BARRIER_ENTRY_EXTENSION_MM,
    ELASTICROD_STRICT_BARRIER_MAX_FORCE_PER_NODE_N,
    ELASTICROD_STRICT_BARRIER_SAFETY_MARGIN_MM,
    ELASTICROD_STRICT_BARRIER_STIFFNESS_N_PER_M,
    ELASTICROD_STRICT_AXIAL_STIFFNESS_SCALE,
    ELASTICROD_STRICT_AXIAL_USE_BODY_FLOOR,
    ELASTICROD_SAFE_AXIAL_STIFFNESS_SCALE,
    ELASTICROD_SAFE_AXIAL_USE_BODY_FLOOR,
    ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER,
    ELASTICROD_STRICT_ENTRY_PUSH_BAND_ENABLED,
    ELASTICROD_STRICT_DRIVE_WINDOW_LENGTH_MM,
    ELASTICROD_STRICT_DRIVE_WINDOW_MIN_NODE_COUNT,
    ELASTICROD_STRICT_DRIVE_WINDOW_OUTSIDE_OFFSET_MM,
    ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_MAX_N,
    ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_N,
    ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT,
    ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM,
    ELASTICROD_STRICT_EXTERNAL_SUPPORT_LENGTH_MM,
    ELASTICROD_STRICT_INITIAL_STRAIGHT_PUSH_MM,
    ELASTICROD_STRICT_MAGNETIC_RECOVERY_LOOKAHEAD_MM,
    ELASTICROD_STRICT_MAGNETIC_RELEASE_SPAN_MM,
    ELASTICROD_STRICT_BEND_LOOKAHEAD_MM,
    ELASTICROD_STRICT_BEND_NEAR_WINDOW_MM,
    ELASTICROD_STRICT_BEND_TURN_MEDIUM_DEG,
    ELASTICROD_STRICT_BEND_TURN_HIGH_DEG,
    ELASTICROD_STRICT_FIELD_SCALE_STRAIGHT,
    ELASTICROD_STRICT_FIELD_SCALE_BEND,
    ELASTICROD_STRICT_RECENTER_CLEARANCE_MM,
    ELASTICROD_STRICT_RECENTER_OFFSET_MM,
    ELASTICROD_STRICT_RECENTER_BLEND,
    ELASTICROD_STRICT_HEAD_STRETCH_RELIEF_START,
    ELASTICROD_STRICT_HEAD_STRETCH_RELIEF_FULL,
    ELASTICROD_STRICT_HEAD_STRETCH_FIELD_SCALE_FLOOR,
    ELASTICROD_STRICT_NATIVE_LUMEN_BARRIER,
    ELASTICROD_STRICT_INITIAL_AXIS_HOLD_MM,
    ELASTICROD_STRICT_HAND_PUSH_NODE_COUNT,
    ELASTICROD_STRICT_SIMPLE_TAIL_DRIVE,
    ELASTICROD_STRICT_SUPPORT_RELEASE_MM,
    ELASTICROD_STRICT_SUPPORT_WINDOW_LENGTH_MM,
    ELASTICROD_SHEATH_LENGTH_MM,
    ELASTICROD_SHEATH_EXIT_STIFFNESS_RATIO,
    ELASTICROD_SHEATH_STIFFNESS_N_PER_M,
    ELASTICROD_VIRTUAL_SHEATH_RELEASE_NODE_COUNT,
    ELASTICROD_STABILIZATION_MODE,
    ELASTICROD_THRUST_FORCE_N,
    ELASTICROD_USE_IMPLICIT_BEND_TWIST,
    ELASTIC_ROD_PLUGIN_NAME,
    ENABLE_CAMERA_FOLLOW,
    ENABLE_LUMEN_SAFETY_PROJECTION,
    ENABLE_VIRTUAL_SHEATH,
    GUIDEWIRE_CONTACT_STIFFNESS,
    GUIDEWIRE_BACKEND,
    MAGNETIC_BA_VECTOR_REF,
    MAGNETIC_BR_VECTOR,
    MAGNETIC_FIELD_RAMP_TIME_S,
    MAGNETIC_FIELD_SMOOTHING_ALPHA,
    MAGNETIC_FORCE_ARROW_ANCHOR_REL,
    MAGNETIC_FORCE_ARROW_HEAD_LENGTH_MM,
    MAGNETIC_FORCE_ARROW_HEAD_WIDTH_MM,
    MAGNETIC_FORCE_ARROW_LENGTH_MM,
    MAGNETIC_HEAD_EDGES,
    MAGNETIC_LOOKAHEAD_DISTANCE_MM,
    MAGNETIC_MAX_TURN_ANGLE_DEG,
    MAGNETIC_MU_ZERO,
    NATIVE_WIRE_BODY_SHEAR_MODULUS_PA,
    NATIVE_WIRE_BODY_YOUNG_MODULUS_PA,
    NATIVE_WIRE_HEAD_SHEAR_MODULUS_PA,
    NATIVE_WIRE_HEAD_YOUNG_MODULUS_PA,
    NATIVE_WIRE_MECHANICAL_CORE_RADIUS_MM,
    NATIVE_WIRE_MASS_DENSITY,
    NATIVE_WIRE_NODE_COUNT,
    NATIVE_WIRE_RADIUS_MM,
    NATIVE_WIRE_SOFT_TIP_EDGE_COUNT,
    NATIVE_WIRE_TOTAL_LENGTH_MM,
    NAVIGATION_MODE,
    OPTION_TXT,
    ELASTICROD_MAGNETIC_CORE_RADIUS_MM,
    ELASTICROD_MATERIAL_PROFILE,
    ELASTICROD_PUSH_MODEL,
    PUSH_FORCE_NODE_COUNT,
    PUSH_FORCE_TARGET_SPEED_MM_S,
    ROOT_GRAVITY,
    ROUTE_DESCRIPTIONS,
    SCENE_AUTOPLAY,
    SCENE_BACKGROUND_RGBA,
    SELECTED_ROUTE_NAME,
    TARGET_MARKER_SIZE_MM,
    VESSEL_COLLISION_DEBUG_RGBA,
    VESSEL_VISUAL_RGBA,
    VESSEL_OBJ,
    VESSEL_VISUAL_OBJ,
    WIRE_BODY_YOUNG_MODULUS_PA,
    WIRE_MASS_DENSITY,
    WIRE_POISSON,
    WIRE_RADIUS_MM,
    WIRE_TOTAL_LENGTH_MM,
    option_parameter_lines,
    segmented_young,
    structured_guidewire_section_properties_mm,
)
from .controller import BeamGuidewireController, ElasticRodGuidewireController
from .geometry import _NearestSurface, _build_open_cylinder_shell, _initial_wire_state, _load_centerline, _load_obj_vertices_faces, _lumen_profile
from .math_utils import _cumlen, _marker_points, _normalize
from .runtime import ensure_sofa, load_elastic_rod_plugin
from .sofa_builders import _add_camera, _add_contact_manager, _add_full_collision

Sofa = ensure_sofa()


def _optional_plugin_load() -> tuple[str, bool]:
    try:
        loaded = load_elastic_rod_plugin()
        return loaded, True
    except Exception as exc:
        if not ALLOW_PLUGIN_MISSING_FALLBACK:
            raise
        loaded = f'PLUGIN_MISSING_FALLBACK: {exc}'
        print(f'[WARN] {loaded}')
        return loaded, False


def _required_plugin_load() -> str:
    try:
        return load_elastic_rod_plugin()
    except Exception as exc:
        raise RuntimeError(
            f'GUIDEWIRE_BACKEND="elasticrod" 需要先成功编译并加载 {ELASTIC_ROD_PLUGIN_NAME} 插件，但当前加载失败：{exc}'
        ) from exc
def _add_external_magnetic_force(
    guidewire,
    centerline: np.ndarray,
    insertion_dir: np.ndarray,
    rod_radius_mm: float,
    magnetic_core_radius_mm: float,
    magnetic_edge_count: int,
    br_vector: np.ndarray,
    ba_vector_ref: np.ndarray,
    look_ahead_distance_mm: float,
    field_smoothing_alpha: float,
    max_field_turn_angle_deg: float,
    field_ramp_time_s: float,
    recovery_look_ahead_distance_mm: float,
    min_torque_sin: float,
    lateral_force_scale: float,
    entry_straight_distance_mm: float,
    entry_steering_release_distance_mm: float,
    bend_look_ahead_distance_mm: float,
    bend_near_window_distance_mm: float,
    bend_turn_medium_deg: float,
    bend_turn_high_deg: float,
    field_scale_straight: float,
    field_scale_bend: float,
    recenter_clearance_mm: float,
    recenter_offset_mm: float,
    recenter_blend: float,
    head_stretch_relief_start: float,
    head_stretch_relief_full: float,
    head_stretch_field_scale_floor: float,
    strict_in_lumen_mode: bool,
    strict_physical_torque_only: bool,
    required: bool,
):
    ba_vector_ref = np.asarray(ba_vector_ref, dtype=float).reshape(3)
    if float(np.linalg.norm(ba_vector_ref)) <= 1.0e-9:
        ba_vector_ref = np.asarray(insertion_dir, dtype=float).reshape(3)
    try:
        return guidewire.addObject(
            'ExternalMagneticForceField',
            name='externalMagneticForce',
            tubeNodes=centerline.tolist(),
            brVector=np.asarray(br_vector, dtype=float).tolist(),
            baVectorRef=ba_vector_ref.tolist(),
            muZero=MAGNETIC_MU_ZERO,
            rodRadius=rod_radius_mm,
            magneticCoreRadiusMm=magnetic_core_radius_mm,
            magneticEdgeCount=int(magnetic_edge_count),
            lookAheadDistance=float(look_ahead_distance_mm),
            recoveryLookAheadDistance=float(recovery_look_ahead_distance_mm),
            fieldSmoothingAlpha=float(field_smoothing_alpha),
            maxFieldTurnAngleDeg=float(max_field_turn_angle_deg),
            fieldRampTime=float(field_ramp_time_s),
            minTorqueSin=float(min_torque_sin),
            lateralForceScale=float(lateral_force_scale),
            entryStraightDistance=float(entry_straight_distance_mm),
            entrySteeringReleaseDistance=float(entry_steering_release_distance_mm),
            bendLookAheadDistance=float(bend_look_ahead_distance_mm),
            bendNearWindowDistance=float(bend_near_window_distance_mm),
            bendTurnMediumDeg=float(bend_turn_medium_deg),
            bendTurnHighDeg=float(bend_turn_high_deg),
            fieldScaleStraight=float(field_scale_straight),
            fieldScaleBend=float(field_scale_bend),
            recenterClearanceMm=float(recenter_clearance_mm),
            recenterOffsetMm=float(recenter_offset_mm),
            recenterBlend=float(recenter_blend),
            headStretchReliefStart=float(head_stretch_relief_start),
            headStretchReliefFull=float(head_stretch_relief_full),
            headStretchFieldScaleFloor=float(head_stretch_field_scale_floor),
            strictInLumenMode=bool(strict_in_lumen_mode),
            strictPhysicalTorqueOnly=bool(strict_physical_torque_only),
            externalFieldScale=1.0,
            externalControlDt=0.0,
            debugTargetPoint=centerline[0, :3].tolist(),
            debugLookAheadPoint=centerline[min(1, centerline.shape[0] - 1), :3].tolist(),
            debugBaVector=insertion_dir.tolist(),
            debugForceVector=insertion_dir.tolist(),
            debugMagneticMomentVector=insertion_dir.tolist(),
            debugTorqueSin=0.0,
            debugAssistForceVector=[0.0, 0.0, 0.0],
            debugOutwardAssistComponentN=0.0,
            debugDistalTangentFieldAngleDeg=0.0,
            debugUpcomingTurnDeg=0.0,
            debugBendSeverity=0.0,
            debugScheduledFieldScale=1.0,
            debugScheduledFieldScaleBase=1.0,
            debugStrictSteeringNeedAlpha=0.0,
            debugEntryReleaseAlpha=1.0,
            debugRecenteringAlpha=0.0,
        )
    except Exception as exc:
        if required:
            raise RuntimeError(f'ExternalMagneticForceField 创建失败: {exc}') from exc
        print(f'[WARN] beam magnetic plugin fallback to Python controller: {exc}')
        return None


def _add_guidewire_visuals(guidewire, init_rigid: np.ndarray, distal_visual_count: int) -> None:
    node_count = int(init_rigid.shape[0])
    distal_visual_count = min(int(distal_visual_count), max(node_count, 1))
    body_end = max(1, node_count - distal_visual_count + 1)
    body_indices = list(range(body_end))
    distal_indices = list(range(node_count - distal_visual_count, node_count))

    body_vis = guidewire.addChild('BodyVisual')
    body_vis.addObject('EdgeSetTopologyContainer', name='topo', edges=[[i, i + 1] for i in range(max(0, len(body_indices) - 1))])
    body_vis.addObject('MechanicalObject', name='dofs', template='Vec3d', position=init_rigid[body_indices, :3].tolist(), showObject=False)
    body_vis.addObject('RigidMapping', input='@../dofs', output='@dofs', rigidIndexPerPoint=body_indices, globalToLocalCoords=True)
    body_model = body_vis.addChild('Model')
    body_model.addObject('OglModel', name='vis', src='@../topo', color='0.80 0.84 0.88 1.0', lineWidth=4)
    body_model.addObject('IdentityMapping', input='@../dofs', output='@vis')

    head = guidewire.addChild('MagneticHead')
    head.addObject('MechanicalObject', name='dofs', template='Vec3d', position=init_rigid[distal_indices, :3].tolist(), showObject=False)
    head.addObject('RigidMapping', input='@../dofs', output='@dofs', rigidIndexPerPoint=distal_indices, globalToLocalCoords=True)
    head.addObject('EdgeSetTopologyContainer', name='topo', edges=[[i, i + 1] for i in range(max(0, len(distal_indices) - 1))])
    head_vis = head.addChild('Visual')
    head_vis.addObject('OglModel', name='vis', src='@../topo', color='1.0 0.10 0.10 1.0', lineWidth=4)
    head_vis.addObject('IdentityMapping', input='@../dofs', output='@vis')


def _add_static_triangle_surface(root, name: str, vertices: np.ndarray, faces: np.ndarray, rgba=None, *, enable_collision: bool = True):
    node = root.addChild(name)
    node.addObject('TriangleSetTopologyContainer', name='topo', triangles=np.asarray(faces, dtype=int).tolist())
    node.addObject('MechanicalObject', name='dofs', template='Vec3d', position=np.asarray(vertices, dtype=float).tolist(), showObject=False)
    if enable_collision:
        node.addObject('TriangleCollisionModel', moving=0, simulated=0, bothSide=1)
    if rgba is not None:
        vis = node.addChild('Visual')
        vis.addObject(
            'OglModel',
            name='vis',
            position=np.asarray(vertices, dtype=float).tolist(),
            triangles=np.asarray(faces, dtype=int).tolist(),
            color=' '.join(str(v) for v in rgba),
        )
    return node


def _add_elasticrod_introducer(root, entry_point: np.ndarray, insertion_dir: np.ndarray):
    entry_point = np.asarray(entry_point, dtype=float).reshape(3)
    insertion_dir = np.asarray(insertion_dir, dtype=float).reshape(3)
    shell_radius_mm = NATIVE_WIRE_RADIUS_MM + ELASTICROD_INTRODUCER_CLEARANCE_MM
    if not ELASTICROD_ENABLE_INTRODUCER:
        return None, np.zeros((0, 3), dtype=float)
    shell_center = entry_point - 0.5 * ELASTICROD_INTRODUCER_LENGTH_MM * insertion_dir
    shell_vertices, shell_faces = _build_open_cylinder_shell(
        shell_center,
        insertion_dir,
        ELASTICROD_INTRODUCER_LENGTH_MM,
        shell_radius_mm,
        radial_segments=ELASTICROD_INTRODUCER_RADIAL_SEGMENTS,
    )
    # Strict elasticrod already models the sheath/support in native DER
    # boundary driving.  Keeping the grey introducer as an extra collision
    # shell adds a second rigid obstacle at the entrance, which can make the
    # rod appear "stuck" with only local head breathing.  The introducer here
    # is therefore visual-only.
    node = _add_static_triangle_surface(
        root,
        'Introducer',
        shell_vertices,
        shell_faces,
        rgba=ELASTICROD_INTRODUCER_VISUAL_RGBA,
        enable_collision=False,
    )
    return node, shell_vertices


def _strict_external_support_radius_mm() -> float:
    return NATIVE_WIRE_RADIUS_MM + ELASTICROD_INTRODUCER_CLEARANCE_MM


def _add_elasticrod_external_support(root, entry_point: np.ndarray, insertion_dir: np.ndarray):
    support_length_mm = max(float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM), 0.0)
    support_radius_mm = _strict_external_support_radius_mm()
    if support_length_mm <= 0.0:
        return None, np.zeros((0, 3), dtype=float), support_radius_mm
    support_center = np.asarray(entry_point, dtype=float).reshape(3) - 0.5 * support_length_mm * np.asarray(insertion_dir, dtype=float).reshape(3)
    support_vertices, support_faces = _build_open_cylinder_shell(
        support_center,
        insertion_dir,
        support_length_mm,
        support_radius_mm,
        radial_segments=ELASTICROD_INTRODUCER_RADIAL_SEGMENTS,
    )
    node = _add_static_triangle_surface(
        root,
        'ExternalSupport',
        support_vertices,
        support_faces,
        rgba=None,
        # The strict direct-tail-feed path already constrains material inside
        # the introducer corridor kinematically. Keeping a second collision
        # shell here only adds redundant contact work and can pin the entry.
        enable_collision=not (
            ELASTICROD_STABILIZATION_MODE == 'strict'
            and ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER
            and ELASTICROD_STRICT_SIMPLE_TAIL_DRIVE
        ),
    )
    return node, support_vertices, support_radius_mm


def _build_beam_backend(guidewire, init_rigid: np.ndarray, edges, centerline: np.ndarray, insertion_dir: np.ndarray, plugin_available: bool) -> dict:
    node_count = int(init_rigid.shape[0])
    guidewire.addObject('EulerImplicitSolver', name='odeSolver', rayleighStiffness=BEAM_RAYLEIGH_STIFFNESS, rayleighMass=BEAM_RAYLEIGH_MASS)
    guidewire.addObject('BTDLinearSolver', name='linearSolver', template='BTDMatrix6d')
    guidewire.addObject(
        'MechanicalObject',
        name='dofs',
        template='Rigid3d',
        position=init_rigid.tolist(),
        rest_position=init_rigid.tolist(),
        showObject=False,
    )
    guidewire.addObject('EdgeSetTopologyContainer', name='topo', edges=edges)

    beam_profiles = _build_structured_edge_profiles(init_rigid[:, :3])
    young = beam_profiles['beam_young_profile'] or segmented_young(len(edges))
    try:
        guidewire.addObject(
            'BeamInterpolation',
            name='beamInterpolation',
            crossSectionShape='circular',
            radius=WIRE_RADIUS_MM,
            defaultYoungModulus=young,
            defaultPoissonRatio=[WIRE_POISSON] * len(edges),
        )
    except Exception:
        guidewire.addObject(
            'BeamInterpolation',
            name='beamInterpolation',
            crossSectionShape='circular',
            radius=WIRE_RADIUS_MM,
            defaultYoungModulus=WIRE_BODY_YOUNG_MODULUS_PA,
            defaultPoissonRatio=WIRE_POISSON,
        )
    guidewire.addObject(
        'AdaptiveBeamForceFieldAndMass',
        name='beamForceFieldAndMass',
        interpolation='@beamInterpolation',
        massDensity=WIRE_MASS_DENSITY,
    )
    guidewire.addObject('LinearSolverConstraintCorrection')

    # `beam` 现在优先走“物理近端推力”路径：
    # - 默认不再靠 Python 直接拖动近端节点；
    # - 而是在尾端截面施加轴向恒定推力；
    # - 旧版 projective 驱动约束只保留为可选 fallback，不再默认启用。
    push_indices = list(range(min(PUSH_FORCE_NODE_COUNT, node_count)))
    proximal_push_ff = guidewire.addObject(
        'ConstantForceField',
        name='proximalPushForce',
        indices=push_indices,
        forces=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in push_indices],
        showArrowSize=0.0,
    )
    if BEAM_ENABLE_LEGACY_DRIVE_CONSTRAINT:
        try:
            guidewire.addObject(
                'PartialFixedProjectiveConstraint',
                name='beamLegacyDriveConstraint',
                indices=push_indices,
                fixedDirections='1 1 1 1 1 1',
            )
        except Exception as exc:
            print(f'[WARN] beam legacy drive constraint unavailable: {exc}')
    distal_force_count = min(int(DISTAL_FORCE_NODE_COUNT), max(node_count, 1))
    distal_indices = list(range(node_count - distal_force_count, node_count))
    tip_torque_ff = guidewire.addObject(
        'ConstantForceField',
        name='tipMagneticTorque',
        indices=distal_indices,
        forces=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in distal_indices],
        showArrowSize=0.0,
    )
    return {
        'controller_cls': BeamGuidewireController,
        'backend_name': 'beam',
        'rod_model': None,
        'native_mass': None,
        'proximal_push_ff': proximal_push_ff,
        'tip_torque_ff': tip_torque_ff,
        # beam 也明确回到旧版 Python 磁导航，不再混入 native 磁力场。
        'magnetic_force': None,
        'drive_node_count': len(push_indices),
        'use_python_magnetic_fallback': True,
    }


def _add_elasticrod_collision(
    physics_node,
    points: np.ndarray,
    edges,
    self_collision: bool,
    wire_radius_mm: float,
    contact_stiffness: float,
    sample_indices: list[int] | None = None,
):
    parent_points = np.asarray(points, dtype=float)
    parent_count = int(parent_points.shape[0])
    collision_indices = [
        int(i) for i in (sample_indices or list(range(parent_count)))
        if 0 <= int(i) < parent_count
    ]
    if len(collision_indices) < 2:
        collision_indices = list(range(parent_count))
    collision_points = parent_points[collision_indices, :3] if parent_count > 0 else parent_points
    collision_edges = [[i, i + 1] for i in range(max(0, len(collision_indices) - 1))]

    coll = physics_node.addChild('CollisionNode')
    coll.addObject('EdgeSetTopologyContainer', name='topo', edges=collision_edges)
    coll.addObject(
        'MechanicalObject',
        name='dofs',
        template='Vec3d',
        position=collision_points.tolist(),
        rest_position=collision_points.tolist(),
        showObject=False,
    )
    coll.addObject(
        'ElasticRodCollisionMapping',
        name='collisionMapping',
        input='@../rodState',
        output='@dofs',
        selectedIndices=collision_indices,
    )
    coll.addObject(
        'LineCollisionModel',
        moving=1,
        simulated=1,
        selfCollision=1 if self_collision else 0,
        proximity=float(wire_radius_mm),
        contactStiffness=float(contact_stiffness),
    )
    coll.addObject(
        'PointCollisionModel',
        moving=1,
        simulated=1,
        selfCollision=1 if self_collision else 0,
        proximity=float(wire_radius_mm),
        contactStiffness=float(contact_stiffness),
    )
    print(
        f'[INFO] Native reduced collision enabled: nodes={len(collision_indices)}/{parent_count}, '
        f'edges={len(collision_edges)}, mapping=ElasticRodCollisionMapping(SI Vec6d->scene Vec3d), '
        f'selfCollision={1 if self_collision else 0}, radius={float(wire_radius_mm):.3f} mm, '
        f'contactStiffness={float(contact_stiffness):.1f}'
    )
    return coll


def _add_elasticrod_virtual_sheath(guidewire_node, physics_node, collision_node, init_nodes: np.ndarray, insertion_dir: np.ndarray):
    if (not ELASTICROD_ENABLE_VIRTUAL_SHEATH) or collision_node is None:
        return None, [], np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    init_nodes = np.asarray(init_nodes, dtype=float)
    if init_nodes.ndim != 2 or init_nodes.shape[0] == 0:
        return None, [], np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    axis = _normalize(np.asarray(insertion_dir, dtype=float).reshape(3))
    if np.linalg.norm(axis) < 1.0e-12:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)

    offsets_all_mm = np.zeros(init_nodes.shape[0], dtype=float)
    if init_nodes.shape[0] > 1:
        offsets_all_mm[1:] = np.cumsum(np.linalg.norm(np.diff(init_nodes, axis=0), axis=1))

    sheath_length_mm = max(float(ELASTICROD_SHEATH_LENGTH_MM), 0.0)
    release_nodes = max(int(ELASTICROD_VIRTUAL_SHEATH_RELEASE_NODE_COUNT), 0)
    base_count = int(np.searchsorted(offsets_all_mm, sheath_length_mm + 1.0e-9, side='right'))
    support_count = int(min(max(base_count, 1), init_nodes.shape[0]))
    offsets_mm = offsets_all_mm[:support_count].copy()

    # Keep the release taper inside the configured sheath length. The previous
    # implementation extended the support arc by `release_nodes`, which turned a
    # nominal 30 mm sheath into a ~44 mm hidden guide and created an artificial
    # hinge when the soft distal segment finally left that region.
    exit_ratio = float(np.clip(ELASTICROD_SHEATH_EXIT_STIFFNESS_RATIO, 0.0, 1.0))
    stiffnesses = np.full(support_count, float(ELASTICROD_SHEATH_STIFFNESS_N_PER_M), dtype=float)
    taper_count = int(min(max(release_nodes, 1), support_count))
    taper_start = max(support_count - taper_count, 0)
    if taper_count > 0:
        alpha = np.linspace(0.0, 1.0, taper_count, dtype=float)
        taper = exit_ratio + (1.0 - exit_ratio) * 0.5 * (1.0 + np.cos(np.pi * alpha))
        stiffnesses[taper_start:] *= taper
    active = stiffnesses > 1.0e-9
    if not np.any(active):
        return None, [], np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    node_indices = np.nonzero(active)[0].astype(int).tolist()
    active_offsets_mm = offsets_mm[active]
    active_stiffnesses = stiffnesses[active]
    base = init_nodes[0, :3].reshape(1, 3)
    target_points = base + active_offsets_mm.reshape(-1, 1) * axis.reshape(1, 3)

    sheath = guidewire_node.addChild('VirtualSheath')
    target_mech = sheath.addObject(
        'MechanicalObject',
        name='targetState',
        template='Vec3d',
        position=target_points.tolist(),
        rest_position=target_points.tolist(),
        showObject=False,
    )
    for local_idx, (node_idx, stiffness) in enumerate(zip(node_indices, active_stiffnesses)):
        collision_node.addObject(
            'RestShapeSpringsForceField',
            name=f'virtualSheathSpring_{int(node_idx)}',
            template='Vec3d',
            points=[int(node_idx)],
            external_rest_shape='@../../VirtualSheath/targetState',
            external_points=[int(local_idx)],
            # CollisionNode is expressed in scene millimetres, while the sheath
            # stiffness is authored in N/m. RestShapeSpringsForceField consumes
            # the displacement numerically in child DOF units, so convert to
            # N/mm here to avoid a 1000x over-stiff proximal clamp.
            stiffness=float(stiffness) * 1.0e-3,
        )

    print(
        f'[INFO] Native virtual sheath enabled: nodes={node_indices}, '
        f'configuredLength={ELASTICROD_SHEATH_LENGTH_MM:.3f} mm, actualArc={active_offsets_mm[-1]:.3f} mm, '
        f'stiffness={ELASTICROD_SHEATH_STIFFNESS_N_PER_M:.3f} N/m'
    )
    return target_mech, node_indices, active_offsets_mm, active_stiffnesses


def _select_elasticrod_entry_push_indices(init_signed_s: np.ndarray, node_count: int) -> list[int]:
    signed_s = np.asarray(init_signed_s, dtype=float).reshape(-1)
    if signed_s.size == 0 or node_count <= 0:
        return list(range(min(max(1, int(ELASTICROD_PUSH_NODE_COUNT)), max(int(node_count), 0))))

    if not (
        ELASTICROD_STABILIZATION_MODE == 'strict'
        and ELASTICROD_STRICT_ENTRY_PUSH_BAND_ENABLED
    ):
        return list(range(min(int(ELASTICROD_PUSH_NODE_COUNT), int(node_count))))

    outside_offset = max(float(ELASTICROD_ENTRY_PUSH_BAND_OUTSIDE_OFFSET_MM), 0.0)
    band_length = max(float(ELASTICROD_ENTRY_PUSH_BAND_LENGTH_MM), 0.5)
    min_count = max(int(ELASTICROD_ENTRY_PUSH_BAND_MIN_NODE_COUNT), 1)
    s_max = -outside_offset
    s_min = -(outside_offset + band_length)

    band_mask = (signed_s >= s_min - 1.0e-9) & (signed_s <= s_max + 1.0e-9)
    indices = np.nonzero(band_mask)[0].astype(int).tolist()
    if len(indices) >= min_count:
        return indices

    outside_indices = np.nonzero(signed_s <= s_max + 1.0e-9)[0].astype(int)
    if outside_indices.size == 0:
        outside_indices = np.arange(min(int(node_count), signed_s.size), dtype=int)

    target_s = -(outside_offset + 0.5 * band_length)
    ranked = sorted(
        outside_indices.tolist(),
        key=lambda idx: (abs(float(signed_s[idx]) - target_s), abs(float(signed_s[idx]) - s_max), idx),
    )
    selected = sorted(ranked[:min(min_count, len(ranked))])
    if selected:
        return selected
    return list(range(min(min_count, int(node_count), signed_s.size)))


def _edge_midpoints_from_tip_mm(nodes_mm: np.ndarray) -> np.ndarray:
    pts = np.asarray(nodes_mm, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        return np.zeros(0, dtype=float)

    edge_lengths_mm = np.linalg.norm(pts[1:, :3] - pts[:-1, :3], axis=1)
    node_distance_from_tip_mm = np.zeros(pts.shape[0], dtype=float)
    for idx in range(pts.shape[0] - 2, -1, -1):
        node_distance_from_tip_mm[idx] = node_distance_from_tip_mm[idx + 1] + float(edge_lengths_mm[idx])
    return 0.5 * (node_distance_from_tip_mm[:-1] + node_distance_from_tip_mm[1:])


def _build_structured_edge_profiles(nodes_mm: np.ndarray) -> dict:
    edge_midpoints_from_tip_mm = _edge_midpoints_from_tip_mm(nodes_mm)
    if edge_midpoints_from_tip_mm.size == 0:
        return {
            'edge_ea_profile': [],
            'edge_ei_profile': [],
            'edge_gj_profile': [],
            'beam_young_profile': [],
        }

    sections = [
        structured_guidewire_section_properties_mm(float(distance_mm))
        for distance_mm in edge_midpoints_from_tip_mm
    ]
    return {
        'edge_ea_profile': [float(section['axial_ea_si']) for section in sections],
        'edge_ei_profile': [float(section['bending_ei_si']) for section in sections],
        'edge_gj_profile': [float(section['torsion_gj_si']) for section in sections],
        'beam_young_profile': [float(section['beam_effective_young_pa']) for section in sections],
    }


def _build_elasticrod_backend(
    guidewire,
    init_rigid: np.ndarray,
    init_signed_s: np.ndarray,
    edges,
    centerline: np.ndarray,
    insertion_dir: np.ndarray,
    vessel_vertices: np.ndarray,
    vessel_faces: np.ndarray,
) -> dict:
    init_nodes = np.asarray(init_rigid[:, :3], dtype=float)
    vessel_vertices = np.asarray(vessel_vertices, dtype=float)
    vessel_faces = np.asarray(vessel_faces, dtype=int).reshape(-1, 3)
    structured_profiles = _build_structured_edge_profiles(init_nodes)
    edge_ea_profile = structured_profiles['edge_ea_profile']
    edge_ei_profile = structured_profiles['edge_ei_profile']
    edge_gj_profile = structured_profiles['edge_gj_profile']
    strict_mode = ELASTICROD_STABILIZATION_MODE == 'strict'
    lumen_radii_mm = (
        _lumen_profile(centerline[:, :3], vessel_vertices[:, :3], vessel_faces[:, :3], face_candidate_count=1024)
        if vessel_vertices.ndim == 2 and vessel_vertices.shape[0] >= 8 and vessel_faces.shape[0] > 0
        else np.full(centerline.shape[0], max(2.0 * float(NATIVE_WIRE_RADIUS_MM), 1.0), dtype=float)
    )
    native_lumen_barrier = bool(
        ELASTICROD_STRICT_NATIVE_LUMEN_BARRIER
        and lumen_radii_mm.shape[0] == centerline.shape[0]
    )
    strict_candidate_node_count = (
        min(int(init_rigid.shape[0]), int(np.asarray(init_signed_s, dtype=float).size))
        if np.asarray(init_signed_s, dtype=float).size > 0
        else int(init_rigid.shape[0])
    )
    strict_native_windows = strict_mode
    disable_native_boundary_driver = bool(
        strict_native_windows and ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER
    )
    if strict_native_windows:
        if disable_native_boundary_driver:
            support_indices = []
            drive_window_indices = []
            support_candidate_indices = []
            drive_candidate_indices = []
            tail_push_indices = list(range(min(
                max(int(ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT), 1),
                int(init_rigid.shape[0]),
            )))
            push_indices = tail_push_indices.copy()
        else:
            # The native strict boundary driver advances a short external
            # corridor by commandedInsertion. Feed it the full material order so
            # the support/drive windows can migrate proximally instead of
            # remaining pinned to the initial tail nodes.
            support_candidate_indices = list(range(strict_candidate_node_count))
            drive_candidate_indices = support_candidate_indices.copy()
            support_indices = support_candidate_indices.copy()
            drive_window_indices = drive_candidate_indices.copy()
            tail_push_indices = []
            push_indices = []
        axial_assist_indices = []
    else:
        support_indices = list(range(min(int(ELASTICROD_PUSH_NODE_COUNT), int(init_rigid.shape[0]))))
        drive_window_indices = _select_elasticrod_entry_push_indices(
            init_signed_s,
            node_count=int(init_rigid.shape[0]),
        )
        support_candidate_indices = support_indices.copy()
        drive_candidate_indices = drive_window_indices.copy()
        push_indices = drive_window_indices.copy()
        tail_push_indices = push_indices.copy()
        axial_assist_indices = push_indices.copy()
    guidewire.addObject(
        'MechanicalObject',
        name='dofs',
        template='Rigid3d',
        position=init_rigid.tolist(),
        rest_position=init_rigid.tolist(),
        showObject=False,
    )
    guidewire.addObject('EdgeSetTopologyContainer', name='topo', edges=edges)
    physics = guidewire.addChild('Physics')
    physics.addObject('EulerImplicitSolver', name='odeSolver', rayleighStiffness=ELASTICROD_RAYLEIGH_STIFFNESS, rayleighMass=ELASTICROD_RAYLEIGH_MASS)
    physics.addObject('SparseLDLSolver', name='linearSolver', template='CompressedRowSparseMatrix')
    init_carrier = np.zeros((init_rigid.shape[0], 6), dtype=float)
    init_carrier[:, :3] = 1.0e-3 * init_rigid[:, :3]
    rod_state_mech = physics.addObject(
        'MechanicalObject',
        name='rodState',
        template='Vec6d',
        position=init_carrier.tolist(),
        rest_position=init_carrier.tolist(),
        showObject=False,
    )
    hand_k = int(max(1, ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT))
    if strict_native_windows:
        if disable_native_boundary_driver:
            push_ff_indices = list(range(min(
                max(int(ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT), hand_k),
                int(init_rigid.shape[0]),
            )))
        else:
            push_ff_indices = [0] if int(init_rigid.shape[0]) > 0 else []
    else:
        push_ff_indices = list(push_indices)
    proximal_push_ff = physics.addObject(
        'ConstantForceField',
        name='proximalPushForce',
        template='Vec6d',
        indices=push_ff_indices,
        forces=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in push_ff_indices],
        showArrowSize=0.0,
    )
    native_axial_assist_ff = None
    if (not strict_native_windows) and ELASTICROD_ENABLE_AXIAL_PATH_ASSIST and len(axial_assist_indices) > 0:
        native_axial_assist_ff = physics.addObject(
            'ConstantForceField',
            name='axialPathAssistForce',
            template='Vec6d',
            indices=axial_assist_indices,
            forces=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in axial_assist_indices],
            showArrowSize=0.0,
        )
    if ELASTICROD_DISTRIBUTED_TRANSLATIONAL_DAMPING_N_S_PER_M > 0.0 or ELASTICROD_DISTRIBUTED_TWIST_DAMPING_NM_S_PER_RAD > 0.0:
        physics.addObject(
            'DiagonalVelocityDampingForceField',
            name='distributedDamping',
            template='Vec6d',
            dampingCoefficient=[[
                float(ELASTICROD_DISTRIBUTED_TRANSLATIONAL_DAMPING_N_S_PER_M),
                float(ELASTICROD_DISTRIBUTED_TRANSLATIONAL_DAMPING_N_S_PER_M),
                float(ELASTICROD_DISTRIBUTED_TRANSLATIONAL_DAMPING_N_S_PER_M),
                float(ELASTICROD_DISTRIBUTED_TWIST_DAMPING_NM_S_PER_RAD),
                0.0,
                0.0,
            ]],
        )

    rod_model = physics.addObject(
        'ElasticRodGuidewireModel',
        name='elasticRodGuidewireModel',
        initialNodes=init_nodes.tolist(),
        undeformedNodes=init_nodes.tolist(),
        tubeNodes=centerline.tolist(),
        tubeRadiiMm=np.asarray(lumen_radii_mm, dtype=float).tolist(),
        nodeInitialPathSmm=np.asarray(init_signed_s, dtype=float).tolist(),
        rho=NATIVE_WIRE_MASS_DENSITY,
        rodRadius=NATIVE_WIRE_RADIUS_MM,
        mechanicalCoreRadiusMm=NATIVE_WIRE_MECHANICAL_CORE_RADIUS_MM,
        dt=ELASTICROD_ACTIVE_DT_S,
        youngHead=NATIVE_WIRE_HEAD_YOUNG_MODULUS_PA,
        youngBody=NATIVE_WIRE_BODY_YOUNG_MODULUS_PA,
        shearHead=NATIVE_WIRE_HEAD_SHEAR_MODULUS_PA,
        shearBody=NATIVE_WIRE_BODY_SHEAR_MODULUS_PA,
        edgeEAProfile=edge_ea_profile,
        edgeEIProfile=edge_ei_profile,
        edgeGJProfile=edge_gj_profile,
        rodLength=NATIVE_WIRE_TOTAL_LENGTH_MM,
        magneticEdgeCount=ELASTICROD_MAGNETIC_HEAD_EDGES,
        softTipEdgeCount=NATIVE_WIRE_SOFT_TIP_EDGE_COUNT,
        pushNodeCount=(0 if disable_native_boundary_driver else ELASTICROD_PUSH_NODE_COUNT),
        axialDriveNodeCount=(0 if disable_native_boundary_driver else ELASTICROD_AXIAL_DRIVE_NODE_COUNT),
        useDynamicSupportWindows=(strict_native_windows and (not disable_native_boundary_driver)),
        supportNodeIndices=([] if disable_native_boundary_driver else support_candidate_indices),
        driveNodeIndices=([] if disable_native_boundary_driver else drive_candidate_indices),
        supportWindowLengthMm=(0.0 if disable_native_boundary_driver else ELASTICROD_STRICT_SUPPORT_WINDOW_LENGTH_MM),
        supportReleaseDistanceMm=(0.0 if disable_native_boundary_driver else ELASTICROD_STRICT_SUPPORT_RELEASE_MM),
        driveWindowLengthMm=(0.0 if disable_native_boundary_driver else ELASTICROD_STRICT_DRIVE_WINDOW_LENGTH_MM),
        driveWindowOutsideOffsetMm=(0.0 if disable_native_boundary_driver else ELASTICROD_STRICT_DRIVE_WINDOW_OUTSIDE_OFFSET_MM),
        driveWindowMinNodeCount=(0 if disable_native_boundary_driver else ELASTICROD_STRICT_DRIVE_WINDOW_MIN_NODE_COUNT),
        commandedInsertion=0.0,
        commandedTwist=0.0,
        insertionDirection=insertion_dir.tolist(),
        proximalAxialStiffness=ELASTICROD_PROXIMAL_AXIAL_STIFFNESS_N_PER_M,
        proximalLateralStiffness=ELASTICROD_PROXIMAL_LATERAL_STIFFNESS_N_PER_M,
        proximalAngularStiffness=ELASTICROD_PROXIMAL_ANGULAR_STIFFNESS_NM_PER_RAD,
        proximalLinearDamping=ELASTICROD_PROXIMAL_LINEAR_DAMPING_N_S_PER_M,
        proximalAngularDamping=ELASTICROD_PROXIMAL_ANGULAR_DAMPING_NM_S_PER_RAD,
        axialStretchStiffnessScale=(
            ELASTICROD_STRICT_AXIAL_STIFFNESS_SCALE
            if strict_native_windows
            else ELASTICROD_SAFE_AXIAL_STIFFNESS_SCALE
        ),
        axialStretchUseBodyFloor=bool(
            ELASTICROD_STRICT_AXIAL_USE_BODY_FLOOR
            if strict_native_windows
            else ELASTICROD_SAFE_AXIAL_USE_BODY_FLOOR
        ),
        useImplicitBendTwist=ELASTICROD_USE_IMPLICIT_BEND_TWIST,
        useKinematicSupportBlock=(False if disable_native_boundary_driver else ELASTICROD_USE_KINEMATIC_SHEATH_DRIVER),
        strictLumenBarrierEnabled=native_lumen_barrier,
        strictLumenActivationMarginMm=ELASTICROD_STRICT_BARRIER_ACTIVATION_MARGIN_MM,
        strictLumenSafetyMarginMm=ELASTICROD_STRICT_BARRIER_SAFETY_MARGIN_MM,
        strictLumenBarrierStiffness=ELASTICROD_STRICT_BARRIER_STIFFNESS_N_PER_M,
        strictLumenBarrierDamping=ELASTICROD_STRICT_BARRIER_DAMPING_N_S_PER_M,
        strictLumenBarrierMaxForcePerNodeN=ELASTICROD_STRICT_BARRIER_MAX_FORCE_PER_NODE_N,
        strictLumenEntryExtensionMm=ELASTICROD_STRICT_BARRIER_ENTRY_EXTENSION_MM,
        strictLumenEntrySupportRadiusMm=(
            (_strict_external_support_radius_mm() + ELASTICROD_STRICT_BARRIER_ACTIVATION_MARGIN_MM)
            if strict_mode
            else 0.0
        ),
    )
    native_mass = physics.addObject(
        'ElasticRodLumpedMass',
        name='mass',
        initialNodes=init_nodes.tolist(),
        undeformedNodes=init_nodes.tolist(),
        rho=NATIVE_WIRE_MASS_DENSITY,
        rodRadius=NATIVE_WIRE_RADIUS_MM,
        mechanicalCoreRadiusMm=NATIVE_WIRE_MECHANICAL_CORE_RADIUS_MM,
        dt=ELASTICROD_ACTIVE_DT_S,
        youngHead=NATIVE_WIRE_HEAD_YOUNG_MODULUS_PA,
        youngBody=NATIVE_WIRE_BODY_YOUNG_MODULUS_PA,
        shearHead=NATIVE_WIRE_HEAD_SHEAR_MODULUS_PA,
        shearBody=NATIVE_WIRE_BODY_SHEAR_MODULUS_PA,
        rodLength=NATIVE_WIRE_TOTAL_LENGTH_MM,
        magneticEdgeCount=ELASTICROD_MAGNETIC_HEAD_EDGES,
        softTipEdgeCount=NATIVE_WIRE_SOFT_TIP_EDGE_COUNT,
    )
    physics.addObject(
        'GenericConstraintCorrection',
        linearSolver='@linearSolver',
        ODESolver='@odeSolver',
    )
    guidewire.addObject(
        'ElasticRodRigidStateAdapter',
        name='rigidStateAdapter',
        input='@Physics/rodState',
        output='@dofs',
    )
    magnetic_force = _add_external_magnetic_force(
        physics,
        centerline,
        insertion_dir,
        NATIVE_WIRE_RADIUS_MM,
        ELASTICROD_MAGNETIC_CORE_RADIUS_MM,
        ELASTICROD_MAGNETIC_HEAD_EDGES,
        ELASTICROD_MAGNETIC_BR_VECTOR,
        MAGNETIC_BA_VECTOR_REF,
        ELASTICROD_MAGNETIC_LOOKAHEAD_DISTANCE_MM,
        ELASTICROD_MAGNETIC_FIELD_SMOOTHING_ALPHA,
        ELASTICROD_MAGNETIC_MAX_TURN_ANGLE_DEG,
        ELASTICROD_MAGNETIC_FIELD_RAMP_TIME_S,
        (
            ELASTICROD_STRICT_MAGNETIC_RECOVERY_LOOKAHEAD_MM
            if strict_mode
            else ELASTICROD_MAGNETIC_LOOKAHEAD_DISTANCE_MM
        ),
        ELASTICROD_NATIVE_MAGNETIC_MIN_TORQUE_SIN,
        (
            ELASTICROD_STRICT_TIP_TARGET_FORCE_N
            if (strict_mode and ELASTICROD_STRICT_MAGNETIC_ASSIST)
            else (ELASTICROD_MAGNETIC_LATERAL_FORCE_SCALE if not strict_mode else 0.0)
        ),
        ELASTICROD_STRICT_INITIAL_STRAIGHT_PUSH_MM if strict_mode else 0.0,
        ELASTICROD_STRICT_MAGNETIC_RELEASE_SPAN_MM if strict_mode else 0.0,
        (
            ELASTICROD_STRICT_BEND_LOOKAHEAD_MM
            if strict_mode
            else ELASTICROD_MAGNETIC_LOOKAHEAD_DISTANCE_MM
        ),
        (
            ELASTICROD_STRICT_BEND_NEAR_WINDOW_MM
            if strict_mode
            else max(0.5, 0.5 * ELASTICROD_MAGNETIC_LOOKAHEAD_DISTANCE_MM)
        ),
        ELASTICROD_STRICT_BEND_TURN_MEDIUM_DEG,
        ELASTICROD_STRICT_BEND_TURN_HIGH_DEG,
        ELASTICROD_STRICT_FIELD_SCALE_STRAIGHT if strict_mode else 1.0,
        ELASTICROD_STRICT_FIELD_SCALE_BEND if strict_mode else 1.0,
        ELASTICROD_STRICT_RECENTER_CLEARANCE_MM if (strict_mode and ELASTICROD_STRICT_MAGNETIC_RECENTER) else 0.0,
        ELASTICROD_STRICT_RECENTER_OFFSET_MM if (strict_mode and ELASTICROD_STRICT_MAGNETIC_RECENTER) else 0.0,
        ELASTICROD_STRICT_RECENTER_BLEND if (strict_mode and ELASTICROD_STRICT_MAGNETIC_RECENTER) else 0.0,
        ELASTICROD_STRICT_HEAD_STRETCH_RELIEF_START,
        ELASTICROD_STRICT_HEAD_STRETCH_RELIEF_FULL,
        ELASTICROD_STRICT_HEAD_STRETCH_FIELD_SCALE_FLOOR,
        strict_mode,
        (ELASTICROD_STRICT_PHYSICAL_TORQUE_ONLY if strict_mode else False),
        required=True,
    )
    return {
        'controller_cls': ElasticRodGuidewireController,
        'backend_name': 'elasticrod',
        'rod_model': rod_model,
        'native_mass': native_mass,
        'physics_rod_state_mech': rod_state_mech,
        'proximal_push_ff': proximal_push_ff,
        'native_axial_assist_ff': native_axial_assist_ff,
        'native_axial_assist_indices': axial_assist_indices,
        'entry_push_indices': tail_push_indices,
        'tail_push_indices': tail_push_indices,
        'native_drive_window_indices': drive_window_indices,
        'tip_torque_ff': None,
        'magnetic_force': magnetic_force,
        'drive_node_count': len(tail_push_indices),
        'native_support_indices': support_indices,
        'native_support_candidate_indices': support_candidate_indices if strict_native_windows else support_indices,
        'strict_support_window_indices': support_indices,
        'use_python_magnetic_fallback': False,
        'strict_native_windows': strict_native_windows,
        'strict_native_boundary_driver_disabled': disable_native_boundary_driver,
        'strict_external_support_length_mm': float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM) if strict_mode else 0.0,
        'strict_external_support_radius_mm': float(NATIVE_WIRE_RADIUS_MM + ELASTICROD_INTRODUCER_CLEARANCE_MM) if strict_mode else 0.0,
        'strict_external_push_max_node_count': int(ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT) if strict_mode else 0,
    }


def createScene(root: Sofa.Core.Node):
    required_plugins = [
        'Sofa.Component.AnimationLoop',
        'Sofa.Component.Collision.Detection.Algorithm',
        'Sofa.Component.Collision.Detection.Intersection',
        'Sofa.Component.Collision.Geometry',
        'Sofa.Component.Collision.Response.Contact',
        'Sofa.Component.Constraint.Lagrangian.Correction',
        'Sofa.Component.Constraint.Lagrangian.Solver',
        'Sofa.Component.Constraint.Projective',
        'Sofa.Component.Controller',
        'Sofa.Component.IO.Mesh',
        'Sofa.Component.LinearSolver.Direct',
        'Sofa.Component.Mapping.Linear',
        'Sofa.Component.Mapping.NonLinear',
        'Sofa.Component.Mass',
        'Sofa.Component.MechanicalLoad',
        'Sofa.Component.ODESolver.Backward',
        'Sofa.Component.Setting',
        'Sofa.Component.SolidMechanics.Spring',
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Constant',
        'Sofa.Component.Topology.Container.Dynamic',
        'Sofa.Component.Visual',
        'Sofa.GL.Component.Rendering3D',
        'Sofa.GL.Component.Shader',
        'BeamAdapter',
    ]
    for i, plugin_name in enumerate(required_plugins):
        root.addObject('RequiredPlugin', name=f'RequiredPlugin_{i}', pluginName=plugin_name)

    if GUIDEWIRE_BACKEND == 'elasticrod':
        loaded_plugin = _required_plugin_load()
        plugin_available = True
    elif GUIDEWIRE_BACKEND == 'beam':
        loaded_plugin, plugin_available = _optional_plugin_load()
    else:
        raise ValueError(f'Unsupported GUIDEWIRE_BACKEND: {GUIDEWIRE_BACKEND}')

    scene_dt = BEAM_ACTIVE_DT_S if GUIDEWIRE_BACKEND == 'beam' else ELASTICROD_ACTIVE_DT_S
    root.dt = scene_dt
    root.gravity = ROOT_GRAVITY.tolist()
    root.addObject('VisualStyle', displayFlags='showVisual showVisualModels hideBehaviorModels hideCollisionModels')
    try:
        root.addObject('BackgroundSetting', color=SCENE_BACKGROUND_RGBA)
    except Exception:
        pass
    root.addObject('DefaultVisualManagerLoop')
    root.addObject('FreeMotionAnimationLoop')
    if GUIDEWIRE_BACKEND == 'beam':
        root.addObject(
            'GenericConstraintSolver',
            name='constraintSolver',
            maxIterations=CONSTRAINT_SOLVER_MAX_ITER,
            tolerance=CONSTRAINT_SOLVER_TOLERANCE,
        )
        contact_alarm_distance = CONTACT_ALARM_DISTANCE_MM
        contact_distance = CONTACT_DISTANCE_MM
        contact_response_params = CONTACT_MANAGER_RESPONSE_PARAMS
    else:
        root.addObject(
            'GenericConstraintSolver',
            name='constraintSolver',
            maxIterations=ELASTICROD_ACTIVE_SOLVER_MAX_ITER,
            tolerance=ELASTICROD_ACTIVE_SOLVER_TOLERANCE,
        )
        contact_alarm_distance = ELASTICROD_CONTACT_ALARM_DISTANCE_MM
        contact_distance = ELASTICROD_CONTACT_DISTANCE_MM
        contact_response_params = ELASTICROD_CONTACT_MANAGER_RESPONSE_PARAMS
    root.addObject('CollisionPipeline', name='pipeline')
    root.addObject('BruteForceBroadPhase')
    root.addObject('BVHNarrowPhase')
    root.addObject('LocalMinDistance', name='proximity', alarmDistance=contact_alarm_distance, contactDistance=contact_distance)
    contact_manager_kwargs = {}
    try:
        _add_contact_manager(root, response_params=contact_response_params, **contact_manager_kwargs)
    except RuntimeError:
        if contact_manager_kwargs:
            _add_contact_manager(root, response_params=contact_response_params)
        else:
            raise

    if not VESSEL_OBJ.exists():
        raise FileNotFoundError(f'Vessel mesh not found: {VESSEL_OBJ}')

    centerline, centerline_path = _load_centerline()
    centerline_cum = _cumlen(centerline)
    vessel_vertices, vessel_faces = _load_obj_vertices_faces(VESSEL_OBJ)
    controller_vessel_vertices, controller_vessel_faces = vessel_vertices, vessel_faces
    controller_vessel_query_count = (
        int(BEAM_VESSEL_QUERY_FACE_CANDIDATE_COUNT)
        if GUIDEWIRE_BACKEND == 'beam'
        else int(ELASTICROD_VESSEL_QUERY_FACE_CANDIDATE_COUNT)
    )
    vessel_query = _NearestSurface(vessel_vertices, vessel_faces, face_candidate_count=controller_vessel_query_count)

    vessel = root.addChild('Vessel')
    vessel.addObject('MeshOBJLoader', name='loader', filename=str(VESSEL_OBJ))
    vessel.addObject('MeshTopology', name='topo', src='@loader')
    vessel.addObject('MechanicalObject', name='dofs', template='Vec3d', position=vessel_vertices.tolist(), showObject=False)
    vessel.addObject('TriangleCollisionModel', moving=0, simulated=0, bothSide=1, color=' '.join(str(v) for v in VESSEL_COLLISION_DEBUG_RGBA))
    vessel_vis = vessel.addChild('Visual')
    vessel_vis.addObject('MeshOBJLoader', name='loader', filename=str(VESSEL_VISUAL_OBJ))
    vessel_vis.addObject('OglModel', name='vis', src='@loader', color=' '.join(str(v) for v in VESSEL_VISUAL_RGBA))

    centerline_node = root.addChild('Centerline')
    centerline_node.addObject('MechanicalObject', name='dofs', template='Vec3d', position=centerline.tolist(), showObject=False)
    centerline_node.addObject('EdgeSetTopologyContainer', name='topo', edges=[[i, i + 1] for i in range(centerline.shape[0] - 1)])
    centerline_vis = centerline_node.addChild('Visual')
    centerline_vis.addObject('OglModel', name='vis', src='@../topo', color='0.08 0.42 0.88 1.0', lineWidth=2)
    centerline_vis.addObject('IdentityMapping', input='@../dofs', output='@vis')

    target_marker = root.addChild('TargetMarker')
    target_marker.addObject(
        'MechanicalObject',
        name='dofs',
        template='Vec3d',
        position=_marker_points(centerline[0, :3], TARGET_MARKER_SIZE_MM).tolist(),
        showObject=False,
    )
    target_marker.addObject('EdgeSetTopologyContainer', name='topo', edges=[[0, 1], [2, 3], [4, 5]])
    target_marker_vis = target_marker.addChild('Visual')
    target_marker_vis.addObject('OglModel', name='vis', src='@../topo', color='1.0 0.92 0.12 1.0', lineWidth=5)
    target_marker_vis.addObject('IdentityMapping', input='@../dofs', output='@vis')

    # Calculate insertion direction as the tangent at vessel entry point
    # Use only the first segment for accurate entry direction
    insertion_dir = np.zeros(3, dtype=float)
    if centerline.shape[0] >= 2:
        first_seg = np.asarray(centerline[1, :3] - centerline[0, :3], dtype=float).reshape(3)
        insertion_dir = _normalize(first_seg)

    if np.linalg.norm(insertion_dir) < 1e-12:
        insertion_dir = np.array([0.0, 0.0, 1.0], dtype=float)

    wire_node_count = BEAM_ACTIVE_NODE_COUNT if GUIDEWIRE_BACKEND == 'beam' else NATIVE_WIRE_NODE_COUNT
    init_rigid, init_signed_s, inserted_tip_arc_mm, min_clearance_mm, inserted_node_count = _initial_wire_state(
        centerline,
        wire_node_count,
        insertion_dir,
        vessel_query,
        initial_tip_insertion_mm=(ELASTICROD_INITIAL_TIP_INSERTION_MM if GUIDEWIRE_BACKEND == 'elasticrod' else None),
        total_length_mm=(NATIVE_WIRE_TOTAL_LENGTH_MM if GUIDEWIRE_BACKEND == 'elasticrod' else WIRE_TOTAL_LENGTH_MM),
        wire_radius_mm=(NATIVE_WIRE_RADIUS_MM if GUIDEWIRE_BACKEND == 'elasticrod' else WIRE_RADIUS_MM),
        smooth_entry_transition=(GUIDEWIRE_BACKEND == 'elasticrod'),
        entry_blend_length_mm=(
            (
                max(ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM, ELASTICROD_INITIAL_TIP_INSERTION_MM)
                if ELASTICROD_STABILIZATION_MODE == 'strict'
                else max(ELASTICROD_INTRODUCER_LENGTH_MM, ELASTICROD_INITIAL_TIP_INSERTION_MM)
            )
            if GUIDEWIRE_BACKEND == 'elasticrod'
            else 0.0
        ),
        initial_axis_hold_mm=(
            ELASTICROD_STRICT_INITIAL_AXIS_HOLD_MM
            if GUIDEWIRE_BACKEND == 'elasticrod' and ELASTICROD_STABILIZATION_MODE == 'strict'
            else 0.0
        ),
    )
    print(
        f'[INFO] Initial guidewire insertion: initialTipInsertion='
        f'{(ELASTICROD_INITIAL_TIP_INSERTION_MM if GUIDEWIRE_BACKEND == "elasticrod" else inserted_tip_arc_mm):.2f} mm, '
        f'tipArc={inserted_tip_arc_mm:.2f} mm, '
        f'insideNodes={inserted_node_count}, minWallClearance={min_clearance_mm:.3f} mm'
    )
    introducer_node = None
    external_support_node = None
    support_vertices = np.zeros((0, 3), dtype=float)
    external_support_radius_mm = 0.0
    if GUIDEWIRE_BACKEND == 'elasticrod':
        if ELASTICROD_STABILIZATION_MODE == 'strict':
            external_support_node, support_vertices, external_support_radius_mm = _add_elasticrod_external_support(
                root,
                centerline[0, :3],
                insertion_dir,
            )
        else:
            introducer_node, support_vertices = _add_elasticrod_introducer(root, centerline[0, :3], insertion_dir)
            external_support_radius_mm = _strict_external_support_radius_mm()

    edges = [[i, i + 1] for i in range(wire_node_count - 1)]
    guidewire = root.addChild('Guidewire')
    if GUIDEWIRE_BACKEND == 'beam':
        backend = _build_beam_backend(guidewire, init_rigid, edges, centerline, insertion_dir, plugin_available)
    else:
        backend = _build_elasticrod_backend(
            guidewire,
            init_rigid,
            init_signed_s,
            edges,
            centerline,
            insertion_dir,
            vessel_vertices,
            vessel_faces,
        )

    _add_guidewire_visuals(
        guidewire,
        init_rigid,
        ELASTICROD_DISTAL_VISUAL_NODE_COUNT if backend['backend_name'] == 'elasticrod' else DISTAL_VISUAL_NODE_COUNT,
    )
    if backend['backend_name'] == 'beam':
        _add_full_collision(
            guidewire,
            init_rigid[:, :3],
            edges,
            self_collision=BEAM_ENABLE_SELF_COLLISION,
            wire_radius_mm=WIRE_RADIUS_MM,
            contact_stiffness=GUIDEWIRE_CONTACT_STIFFNESS,
        )
    else:
        collision_sample_indices = list(range(0, int(init_rigid.shape[0]), max(int(ELASTICROD_COLLISION_DECIMATION_STRIDE), 1)))
        if int(init_rigid.shape[0]) > 0 and (not collision_sample_indices or collision_sample_indices[-1] != int(init_rigid.shape[0]) - 1):
            collision_sample_indices.append(int(init_rigid.shape[0]) - 1)
        collision_node = _add_elasticrod_collision(
            guidewire.getChild('Physics'),
            init_rigid[:, :3],
            edges,
            self_collision=ELASTICROD_ENABLE_SELF_COLLISION,
            wire_radius_mm=NATIVE_WIRE_RADIUS_MM,
            contact_stiffness=ELASTICROD_GUIDEWIRE_CONTACT_STIFFNESS,
            sample_indices=collision_sample_indices,
        )
        if not ELASTICROD_ENABLE_VIRTUAL_SHEATH:
            target_mech = None
            sheath_indices = []
            sheath_offsets_mm = np.zeros(0, dtype=float)
            sheath_stiffnesses = np.zeros(0, dtype=float)
        else:
            target_mech, sheath_indices, sheath_offsets_mm, sheath_stiffnesses = _add_elasticrod_virtual_sheath(
                guidewire,
                guidewire.getChild('Physics'),
                collision_node,
                init_rigid[:, :3],
                insertion_dir,
            )
        if not (ELASTICROD_STABILIZATION_MODE == 'strict' and ELASTICROD_STRICT_DISABLE_NATIVE_BOUNDARY_DRIVER):
            print(
                f'[INFO] Native sheath support uses ElasticRodGuidewireModel boundary driving: '
                f'initialEntrySupportNodes={backend.get("strict_support_window_indices", backend.get("native_support_indices", []))}, '
                f'supportCandidateNodeCount={len(backend.get("native_support_candidate_indices", backend.get("native_support_indices", [])))}, '
                f'nativeDriveWindowNodes={backend.get("native_drive_window_indices", [])}, '
                f'tailPushNodes={backend.get("tail_push_indices", backend.get("entry_push_indices", []))}, '
                f'kinematic={bool(ELASTICROD_USE_KINEMATIC_SHEATH_DRIVER)}, mappedRestShapeSprings={target_mech is not None}'
            )
        native_drive_nodes = [int(i) for i in backend.get('native_drive_window_indices', [])]
        tail_push_nodes = [int(i) for i in backend.get('tail_push_indices', backend.get('entry_push_indices', []))]
        native_drive_set = set(native_drive_nodes)
        tail_push_set = set(tail_push_nodes)
        if native_drive_nodes and (
            native_drive_set == tail_push_set
            or (tail_push_set and native_drive_set.issubset(tail_push_set))
            or max(native_drive_nodes) <= max(tail_push_nodes or [ELASTICROD_STRICT_HAND_PUSH_NODE_COUNT - 1])
        ):
            print(
                '[WARN] [elasticrod-strict] wiring mismatch: '
                f'nativeDriveWindowNodes={native_drive_nodes} tailPushNodes={tail_push_nodes}'
            )
        backend['virtual_sheath_target'] = target_mech
        backend['virtual_sheath_indices'] = sheath_indices
        backend['virtual_sheath_offsets_mm'] = sheath_offsets_mm
        backend['virtual_sheath_stiffnesses'] = sheath_stiffnesses

    scene_points_base = np.vstack((vessel_vertices, centerline[:, :3], init_rigid[:, :3], support_vertices))
    scene_base_min = np.min(scene_points_base, axis=0)
    scene_base_max = np.max(scene_points_base, axis=0)
    scene_base_span = np.maximum(scene_base_max - scene_base_min, 1.0)
    force_arrow_anchor = scene_base_min + MAGNETIC_FORCE_ARROW_ANCHOR_REL * scene_base_span
    preview_side = np.cross(insertion_dir, np.array([0.0, 0.0, 1.0], dtype=float))
    if np.linalg.norm(preview_side) < 1e-12:
        preview_side = np.cross(insertion_dir, np.array([0.0, 1.0, 0.0], dtype=float))
    preview_side = _normalize(preview_side)
    if np.linalg.norm(preview_side) < 1e-12:
        preview_side = np.array([1.0, 0.0, 0.0], dtype=float)
    preview_up = _normalize(np.cross(insertion_dir, preview_side))
    if np.linalg.norm(preview_up) < 1e-12:
        preview_up = np.array([0.0, 1.0, 0.0], dtype=float)
    preview_head_base = force_arrow_anchor + (MAGNETIC_FORCE_ARROW_LENGTH_MM - MAGNETIC_FORCE_ARROW_HEAD_LENGTH_MM) * insertion_dir
    arrow_points = np.asarray([
        force_arrow_anchor,
        force_arrow_anchor + MAGNETIC_FORCE_ARROW_LENGTH_MM * insertion_dir,
        preview_head_base + MAGNETIC_FORCE_ARROW_HEAD_WIDTH_MM * preview_side,
        preview_head_base - MAGNETIC_FORCE_ARROW_HEAD_WIDTH_MM * preview_side,
        preview_head_base + MAGNETIC_FORCE_ARROW_HEAD_WIDTH_MM * preview_up,
        preview_head_base - MAGNETIC_FORCE_ARROW_HEAD_WIDTH_MM * preview_up,
    ], dtype=float)
    force_arrow = root.addChild('MagneticForceArrow')
    force_arrow.addObject('MechanicalObject', name='dofs', template='Vec3d', position=arrow_points.tolist(), showObject=False)
    force_arrow.addObject('EdgeSetTopologyContainer', name='topo', edges=[[0, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
    force_arrow_vis = force_arrow.addChild('Visual')
    force_arrow_vis.addObject('OglModel', name='vis', src='@../topo', color='1.0 0.55 0.05 1.0', lineWidth=8)
    force_arrow_vis.addObject('IdentityMapping', input='@../dofs', output='@vis')

    scene_points = np.vstack((scene_points_base, arrow_points))
    scene_min = np.min(scene_points, axis=0) - np.array([20.0, 20.0, 20.0], dtype=float)
    scene_max = np.max(scene_points, axis=0) + np.array([20.0, 20.0, 20.0], dtype=float)
    scene_base_center = 0.5 * (scene_base_min + scene_base_max)
    route_focus_index = int(np.clip(round(DEFAULT_CAMERA_ROUTE_FOCUS_FRACTION * (centerline.shape[0] - 1)), 0, centerline.shape[0] - 1))
    route_focus_point = centerline[route_focus_index, :3]
    default_camera_focus = (
        (1.0 - DEFAULT_CAMERA_ROUTE_FOCUS_WEIGHT) * scene_base_center
        + DEFAULT_CAMERA_ROUTE_FOCUS_WEIGHT * route_focus_point
    )
    # The native elasticrod backend keeps a long extra-vascular shaft outside
    # the vessel. If we focus only on the route midpoint, the visible magnetic
    # tip can start well outside the initial camera frustum and the scene looks
    # like it contains no guidewire at all.
    camera_focus_center = (
        init_rigid[-1, :3]
        if (ENABLE_CAMERA_FOLLOW or GUIDEWIRE_BACKEND == 'elasticrod')
        else default_camera_focus
    )
    camera, camera_follow_offset = _add_camera(
        root,
        camera_focus_center,
        max(float(np.linalg.norm(scene_base_max - scene_base_min)), 100.0),
    )

    root.addObject(
        backend['controller_cls'](
            name='GuidewireNavigationController',
            root_node=root,
            constraint_solver=root.getObject('constraintSolver'),
            wire_mech=guidewire.getObject('dofs'),
            proximal_push_ff=backend['proximal_push_ff'],
            tip_torque_ff=backend['tip_torque_ff'],
            rod_model=backend['rod_model'],
            native_mass=backend['native_mass'],
            physics_rod_state_mech=backend.get('physics_rod_state_mech'),
            native_axial_assist_ff=backend.get('native_axial_assist_ff'),
            native_axial_assist_indices=backend.get('native_axial_assist_indices', []),
            backend_name=backend['backend_name'],
            magnetic_force_field=backend['magnetic_force'],
            centerline_points=centerline.tolist(),
            push_force_target_speed_mm_s=(BEAM_ACTIVE_PUSH_SPEED_MM_S if backend['backend_name'] == 'beam' else PUSH_FORCE_TARGET_SPEED_MM_S),
            insertion_direction=insertion_dir.tolist(),
            max_push_mm=float(centerline_cum[-1]) + 20.0,
            node_initial_path_s_mm=init_signed_s.tolist(),
            vessel_vertices=controller_vessel_vertices.tolist(),
            vessel_faces=controller_vessel_faces.tolist(),
            vessel_surface_query_face_candidate_count=(
                controller_vessel_query_count
                if backend['backend_name'] == 'elasticrod' and backend.get('strict_native_windows')
                else (128 if backend['backend_name'] == 'beam' else 384)
            ),
            enable_vessel_lumen_constraint=(
                ENABLE_LUMEN_SAFETY_PROJECTION
                if backend['backend_name'] == 'beam'
                else ELASTICROD_ENABLE_LUMEN_SAFETY_PROJECTION
            ),
            # beam 录屏路径恢复到短入口导向/虚拟鞘管版本；elasticrod 仍走自己的 native support 语义。
            enable_virtual_sheath=(ENABLE_VIRTUAL_SHEATH if backend['backend_name'] == 'beam' else False),
            target_marker_mech=target_marker.getObject('dofs'),
            force_arrow_mech=force_arrow.getObject('dofs'),
            force_arrow_anchor=force_arrow_anchor.tolist(),
            navigation_mode=NAVIGATION_MODE,
            drive_node_count=backend['drive_node_count'],
            entry_push_indices=backend.get('entry_push_indices', []),
            native_support_indices=backend.get('native_support_indices', []),
            native_drive_window_indices=backend.get('native_drive_window_indices', []),
            tail_push_indices=backend.get('tail_push_indices', []),
            external_support_length_mm=(
                float(ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM)
                if backend['backend_name'] == 'elasticrod' and backend.get('strict_native_windows')
                else 0.0
            ),
            external_support_radius_mm=(
                float(external_support_radius_mm)
                if backend['backend_name'] == 'elasticrod' and backend.get('strict_native_windows')
                else 0.0
            ),
            camera_object=camera if ENABLE_CAMERA_FOLLOW else None,
            camera_follow_offset=camera_follow_offset.tolist() if (ENABLE_CAMERA_FOLLOW and camera_follow_offset is not None) else None,
            use_python_magnetic_fallback=backend['use_python_magnetic_fallback'],
            python_magnetic_field_strength=BEAM_PYTHON_MAGNETIC_FIELD_STRENGTH,
            python_magnetic_moment=BEAM_PYTHON_MAGNETIC_MOMENT,
            python_magnetic_force_gain=BEAM_PYTHON_MAGNETIC_FORCE_GAIN,
            native_virtual_sheath_target_mech=backend.get('virtual_sheath_target'),
            native_virtual_sheath_indices=backend.get('virtual_sheath_indices', []),
            native_virtual_sheath_offsets_mm=backend.get('virtual_sheath_offsets_mm', []),
            native_virtual_sheath_stiffnesses=backend.get('virtual_sheath_stiffnesses', []),
            enable_native_virtual_sheath=(
                bool(ELASTICROD_ENABLE_VIRTUAL_SHEATH and backend.get('virtual_sheath_target') is not None)
                if backend['backend_name'] == 'elasticrod'
                else False
            ),
            enable_native_thrust_limit=ELASTICROD_ENABLE_THRUST_LIMIT if backend['backend_name'] == 'elasticrod' else False,
            native_thrust_force_n=ELASTICROD_THRUST_FORCE_N if backend['backend_name'] == 'elasticrod' else 0.0,
        )
    )

    root.bbox = np.stack((scene_min, scene_max))
    root.bbox.value = [scene_min.tolist(), scene_max.tolist()]

    print('=' * 72)
    print(f'[guidewire_navigation] {backend["backend_name"]} scene loaded')
    print(f'  plugin: {loaded_plugin}')
    print(f'  option file: {OPTION_TXT}')
    print(f'  vessel surface: {VESSEL_OBJ}')
    print(f'  vessel visual mesh: {VESSEL_VISUAL_OBJ}')
    print(f'  scene background: {SCENE_BACKGROUND_RGBA}')
    print(f'  scene autoplay default: {SCENE_AUTOPLAY}')
    print(f'  route name: {SELECTED_ROUTE_NAME}')
    print(f'  route summary: {ROUTE_DESCRIPTIONS.get(SELECTED_ROUTE_NAME, "n/a")}')
    print(f'  centerline: {centerline_path} (n={centerline.shape[0]}, L={float(centerline_cum[-1]):.1f} mm)')
    guidewire_length_mm = WIRE_TOTAL_LENGTH_MM if backend['backend_name'] == 'beam' else NATIVE_WIRE_TOTAL_LENGTH_MM
    magnetic_edge_count = ELASTICROD_MAGNETIC_HEAD_EDGES if backend['backend_name'] == 'elasticrod' else MAGNETIC_HEAD_EDGES
    print(f'  guidewire length: {guidewire_length_mm:.1f} mm, nodes: {wire_node_count}, magnetic edges: {magnetic_edge_count}')
    for line in option_parameter_lines(backend['backend_name']):
        print(line)
    if backend['backend_name'] == 'beam':
        print(f'  beam rayleigh(k/m): {BEAM_RAYLEIGH_STIFFNESS:.3f} / {BEAM_RAYLEIGH_MASS:.3f}')
        print(f'  beam magnetic source: {"ExternalMagneticForceField" if backend["magnetic_force"] is not None else "Python fallback"}')
        print(f'  beam runtime profile: {BEAM_RUNTIME_PROFILE}')
        print(f'  beam kinematic insertion: {BEAM_USE_KINEMATIC_INSERTION}')
        print(f'  beam legacy drive constraint: {BEAM_ENABLE_LEGACY_DRIVE_CONSTRAINT}')
    else:
        print(f'  elasticrod rayleigh(k/m): {ELASTICROD_RAYLEIGH_STIFFNESS:.3f} / {ELASTICROD_RAYLEIGH_MASS:.3f}')
        print(f'  elasticrod stabilization mode: {ELASTICROD_STABILIZATION_MODE}')
        print(f'  elasticrod runtime profile: {ELASTICROD_RUNTIME_PROFILE}')
        print(
            f'  elasticrod material(profile/contact/core): '
            f'{ELASTICROD_MATERIAL_PROFILE} / {ELASTICROD_CONTACT_OUTER_RADIUS_MM:.3f} / {NATIVE_WIRE_MECHANICAL_CORE_RADIUS_MM:.3f} mm'
        )
        print(
            f'  elasticrod metal Young(body/head): '
            f'{NATIVE_WIRE_BODY_YOUNG_MODULUS_PA:.3e} / {NATIVE_WIRE_HEAD_YOUNG_MODULUS_PA:.3e} Pa ; '
            f'density={NATIVE_WIRE_MASS_DENSITY:.1f} kg/m^3 ; softTipEdges={NATIVE_WIRE_SOFT_TIP_EDGE_COUNT}'
        )
        print(
            f'  elasticrod push model: {ELASTICROD_PUSH_MODEL} '
            f'(kinematicSheathDriver={ELASTICROD_USE_KINEMATIC_SHEATH_DRIVER}, introducer={ELASTICROD_INTRODUCER_LENGTH_MM:.1f} mm)'
        )
        print(f'  elasticrod lumen safety projection: {ELASTICROD_ENABLE_LUMEN_SAFETY_PROJECTION}')
        print(f'  elasticrod self collision: {ELASTICROD_ENABLE_SELF_COLLISION}')
        if backend.get('strict_native_windows'):
            print(
                f'  elasticrod external support: '
                f'node={"yes" if external_support_node is not None else "no"} '
                f'(length={ELASTICROD_STRICT_EXTERNAL_SUPPORT_EFFECTIVE_LENGTH_MM:.3f} mm, radius={external_support_radius_mm:.3f} mm)'
            )
            if backend.get("strict_native_boundary_driver_disabled", False):
                print(
                    f'  elasticrod strict insertion: external push(total/max/nodes)= '
                    f'{ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_N:.3f} N / '
                    f'{ELASTICROD_STRICT_EXTERNAL_PUSH_FORCE_MAX_N:.3f} N / '
                    f'{ELASTICROD_STRICT_EXTERNAL_PUSH_MAX_NODE_COUNT} ; '
                    f'nativeBoundaryDriverDisabled=True'
                )
            else:
                print(
                    '  elasticrod strict insertion: native boundary driver enabled '
                    f'(dynamicSupportWindow=True, kinematicSupportBlock={ELASTICROD_USE_KINEMATIC_SHEATH_DRIVER}, '
                    f'externalPushFallbackDisabled=True)'
                )
        else:
            print(f'  elasticrod introducer: {ELASTICROD_ENABLE_INTRODUCER} (node={"yes" if introducer_node is not None else "no"})')
        print(f'  elasticrod initial tip insertion: {ELASTICROD_INITIAL_TIP_INSERTION_MM:.3f} mm')
        print(
            f'  elasticrod virtual sheath: {ELASTICROD_ENABLE_VIRTUAL_SHEATH} '
            f'(length={ELASTICROD_SHEATH_LENGTH_MM:.3f} mm, stiffness={ELASTICROD_SHEATH_STIFFNESS_N_PER_M:.3f} N/m)'
        )
        if not backend.get('strict_native_windows'):
            print(f'  elasticrod native drive window nodes(initial): {backend.get("native_drive_window_indices", [])}')
            print(f'  elasticrod tail push nodes(initial): {backend.get("tail_push_indices", backend.get("entry_push_indices", []))}')
            print(f'  elasticrod native kinematic sheath driver: {ELASTICROD_USE_KINEMATIC_SHEATH_DRIVER}')
            print(f'  elasticrod thrust limit: {ELASTICROD_ENABLE_THRUST_LIMIT} ({ELASTICROD_THRUST_FORCE_N:.3f} N)')
        print(f'  elasticrod field gradient enabled: {ELASTICROD_ENABLE_FIELD_GRADIENT}')
        print(
            '  elasticrod contact(mu/alarm/contact): '
            f'{ELASTICROD_CONTACT_MANAGER_RESPONSE_PARAMS} / {ELASTICROD_CONTACT_ALARM_DISTANCE_MM:.3f} / {ELASTICROD_CONTACT_DISTANCE_MM:.3f}'
        )
        print('  elasticrod fallback policy: disabled')
    print('=' * 72)
    return root
