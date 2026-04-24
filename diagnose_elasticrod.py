# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _preparse_runtime_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--profile', default='no-push-no-mag')
    parser.add_argument('--sleep', type=float, default=0.0)
    return parser.parse_known_args(argv)[0]


def _load_runtime(profile: str) -> SimpleNamespace:
    if profile == 'gui-benchmark':
        os.environ['GUIDEWIRE_ELASTICROD_GUI_WALLCLOCK'] = '1'

    from guidewire_scene.config import (  # noqa: E402
        ELASTICROD_FAILFAST_MAX_STRETCH,
        ELASTICROD_SAFE_RECOVERY_ANGULAR_SPEED_RAD_S,
        ELASTICROD_SAFE_RECOVERY_LINEAR_SPEED_MM_S,
        ELASTICROD_STRICT_GLOBAL_STRETCH_HARD_LIMIT,
        ELASTICROD_STRICT_HEAD_STRETCH_LIMIT,
    )
    from guidewire_scene.runtime import ensure_sofa  # noqa: E402

    Sofa = ensure_sofa()
    import Sofa.Simulation  # noqa: E402,F401

    import guidewire_scene.controller as ctrlmod  # noqa: E402
    from guidewire_scene.scene import createScene  # noqa: E402

    return SimpleNamespace(
        sofa=Sofa,
        ctrlmod=ctrlmod,
        createScene=createScene,
        abort_on_stretch_default=float(ELASTICROD_FAILFAST_MAX_STRETCH),
        safe_recovery_angular_speed_rad_s=float(ELASTICROD_SAFE_RECOVERY_ANGULAR_SPEED_RAD_S),
        safe_recovery_linear_speed_mm_s=float(ELASTICROD_SAFE_RECOVERY_LINEAR_SPEED_MM_S),
        strict_global_stretch_hard_limit=float(ELASTICROD_STRICT_GLOBAL_STRETCH_HARD_LIMIT),
        strict_head_stretch_limit=float(ELASTICROD_STRICT_HEAD_STRETCH_LIMIT),
    )


def _parse_args(argv: list[str] | None = None, *, abort_on_stretch_default: float) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a single lightweight elasticrod diagnostic case.')
    parser.add_argument(
        '--profile',
        choices=('free-space', 'no-push-no-mag', 'push-only', 'full', 'gui-benchmark'),
        default='no-push-no-mag',
        help='Diagnostic profile to run.',
    )
    parser.add_argument('--steps', type=int, default=200, help='Maximum simulation steps.')
    parser.add_argument('--timeout-s', type=float, default=600.0, help='Wall-clock timeout in seconds.')
    parser.add_argument(
        '--abort-on-stretch',
        type=float,
        default=float(abort_on_stretch_default),
        help='Abort if max absolute stretch reaches this threshold.',
    )
    parser.add_argument('--print-every', type=int, default=10, help='Print a progress line every N steps.')
    parser.add_argument('--sleep', type=float, default=0.0, help='Optional wall-clock sleep after each animation step.')
    return parser.parse_args(argv)


def _maybe_remove_child(root, name: str) -> None:
    try:
        child = root.getChild(name)
    except Exception:
        child = None
    if child is None:
        return
    try:
        root.removeChild(child)
        print(f'[INFO] Removed child node for diagnostic profile: {name}')
    except Exception as exc:
        print(f'[WARN] Failed to remove child node {name}: {exc}')


def _read_array(data) -> np.ndarray:
    try:
        return np.asarray(data.value, dtype=float)
    except Exception:
        return np.zeros(0, dtype=float)


def _read_scalar(data, default: float = 0.0) -> float:
    try:
        return float(data.value)
    except Exception:
        return float(default)


def _safe_child(node, name: str):
    try:
        return node.getChild(name)
    except Exception:
        return None


def _safe_object(node, name: str):
    if node is None:
        return None
    try:
        return node.getObject(name)
    except Exception:
        return None


def _position_array(obj) -> np.ndarray:
    if obj is None:
        return np.zeros((0, 3), dtype=float)
    try:
        return np.asarray(obj.findData('position').value, dtype=float)
    except Exception:
        return np.zeros((0, 3), dtype=float)


def _visual_sync_error_mm(root) -> float:
    guidewire = _safe_child(root, 'Guidewire')
    physics = _safe_child(guidewire, 'Physics')
    rod_state = _safe_object(physics, 'rodState')
    rigid = _safe_object(guidewire, 'dofs')
    body = _safe_child(guidewire, 'BodyVisual')
    body_model = _safe_child(body, 'Model')
    head = _safe_child(guidewire, 'MagneticHead')
    head_visual = _safe_child(head, 'Visual')

    rigid_pos = _position_array(rigid)
    body_pos = _position_array(_safe_object(body, 'dofs'))
    body_vis_pos = _position_array(_safe_object(body_model, 'vis'))
    head_pos = _position_array(_safe_object(head, 'dofs'))
    head_vis_pos = _position_array(_safe_object(head_visual, 'vis'))
    rod_pos = _position_array(rod_state)

    errors: list[float] = []
    if rod_pos.ndim == 2 and rod_pos.shape[0] > 0 and rigid_pos.ndim == 2 and rigid_pos.shape[0] >= rod_pos.shape[0]:
        n = int(rod_pos.shape[0])
        errors.append(float(np.max(np.linalg.norm(1000.0 * rod_pos[:n, :3] - rigid_pos[:n, :3], axis=1))))

    for child, child_pos, child_vis_pos in ((body, body_pos, body_vis_pos), (head, head_pos, head_vis_pos)):
        if child_pos.ndim == 2 and child_vis_pos.ndim == 2 and child_pos.shape[0] > 0 and child_vis_pos.shape[0] > 0:
            count = min(int(child_pos.shape[0]), int(child_vis_pos.shape[0]))
            errors.append(float(np.max(np.linalg.norm(child_pos[:count, :3] - child_vis_pos[:count, :3], axis=1))))

    return max(errors) if errors else float('inf')


def _configure_profile(root, profile: str, runtime: SimpleNamespace) -> tuple[object, object, object, object]:
    profile_mode = 'full' if profile == 'gui-benchmark' else profile
    if profile_mode == 'free-space':
        _maybe_remove_child(root, 'Vessel')
        _maybe_remove_child(root, 'Introducer')
        _maybe_remove_child(root, 'ExternalSupport')

    runtime.sofa.Simulation.init(root)
    controller = root.getObject('GuidewireNavigationController')
    guidewire = root.getChild('Guidewire')
    dofs = guidewire.getObject('dofs')
    try:
        physics = guidewire.getChild('Physics')
    except Exception:
        physics = guidewire
    rod = physics.getObject('elasticRodGuidewireModel')
    magnetic = physics.getObject('externalMagneticForce')

    if profile_mode in {'free-space', 'no-push-no-mag'}:
        controller.push_force_target_speed_mm_s = 0.0
        controller.commanded_push_mm = 0.0
        rod.findData('commandedInsertion').value = 0.0
        if magnetic is not None:
            magnetic.findData('brVector').value = [0.0, 0.0, 0.0]
        if hasattr(controller, '_native_nominal_br_vector'):
            controller._native_nominal_br_vector = np.zeros(3, dtype=float)
    elif profile_mode == 'push-only':
        if magnetic is not None:
            magnetic.findData('brVector').value = [0.0, 0.0, 0.0]
        if hasattr(controller, '_native_nominal_br_vector'):
            controller._native_nominal_br_vector = np.zeros(3, dtype=float)

    return controller, dofs, rod, magnetic


def run_case(args: argparse.Namespace, runtime: SimpleNamespace) -> int:
    runtime.ctrlmod.DEBUG_PRINT_EVERY = 0
    runtime.ctrlmod.ELASTICROD_DIAGNOSTIC_PRINT_EVERY = 0
    runtime.ctrlmod.ELASTICROD_DIAGNOSTIC_STEP_WINDOW = 0

    root = runtime.sofa.Core.Node(f'diag_{args.profile}')
    runtime.createScene(root)
    controller, dofs, rod, magnetic = _configure_profile(root, args.profile, runtime)
    initial_positions = np.asarray(dofs.position.value, dtype=float)
    initial_centers = initial_positions[:, :3].copy() if initial_positions.ndim == 2 and initial_positions.shape[1] >= 3 else np.zeros((0, 3), dtype=float)
    insertion_dir = np.asarray(getattr(controller, 'insertion_direction', np.array([0.0, 0.0, 1.0], dtype=float)), dtype=float).reshape(3)
    insertion_dir_norm = float(np.linalg.norm(insertion_dir))
    if insertion_dir_norm > 1.0e-12:
        insertion_dir = insertion_dir / insertion_dir_norm
    else:
        insertion_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    node_count = int(initial_centers.shape[0])
    base_index = 0
    mid_index = max(node_count // 2, 0)
    visual_sync_max_error_mm = 0.0

    if getattr(controller, 'is_native_strict', False):
        print(
            f'[DIAG] externalPushNodes={getattr(controller, "_active_external_push_indices", [])} '
            f'guiWallclockControl={getattr(controller, "use_native_gui_wallclock_control", False)}'
        )
    else:
        print(
            f'[DIAG] nativeDriveWindowNodes={getattr(controller, "native_drive_window_indices", [])} '
            f'tailPushNodes={getattr(controller, "tail_push_indices", getattr(controller, "entry_push_indices", []))} '
            f'initialEntrySupportNodes={getattr(controller, "native_support_indices", [])} '
            f'guiWallclockControl={getattr(controller, "use_native_gui_wallclock_control", False)}'
        )

    stretch_data = rod.findData('debugStretch')
    edge_len_data = rod.findData('debugEdgeLengthMm')
    abnormal_idx_data = rod.findData('debugAbnormalEdgeIndex')
    abnormal_len_data = rod.findData('debugAbnormalEdgeLengthMm')
    abnormal_ref_data = rod.findData('debugAbnormalEdgeRefLengthMm')
    axial_err_data = rod.findData('debugMaxAxialBoundaryErrorMm')
    lateral_err_data = rod.findData('debugMaxLateralBoundaryErrorMm')
    min_clearance_data = rod.findData('debugMinLumenClearanceMm')
    barrier_nodes_data = rod.findData('debugBarrierActiveNodeCount')
    head_stretch_data = rod.findData('debugMaxHeadStretch')
    torque_sin_data = magnetic.findData('debugTorqueSin') if magnetic is not None else None
    outward_assist_data = magnetic.findData('debugOutwardAssistComponentN') if magnetic is not None else None

    t0 = time.perf_counter()
    prev_base_progress_mm = 0.0
    prev_mid_progress_mm = 0.0
    prev_tip_progress_mm = 0.0
    for step in range(1, max(int(args.steps), 1) + 1):
        elapsed = time.perf_counter() - t0
        if elapsed >= float(args.timeout_s):
            print(f'[TIMEOUT] profile={args.profile} step={step-1} elapsed={elapsed:.2f}s')
            return 2

        dt = float(root.dt.value)
        runtime.sofa.Simulation.animate(root, dt)
        if args.sleep > 0.0:
            time.sleep(float(args.sleep))

        velocities = np.asarray(dofs.velocity.value, dtype=float)
        lin = np.linalg.norm(velocities[:, :3], axis=1) if velocities.ndim == 2 and velocities.shape[1] >= 3 else np.zeros(0, dtype=float)
        ang = np.linalg.norm(velocities[:, 3:], axis=1) if velocities.ndim == 2 and velocities.shape[1] >= 6 else np.zeros(0, dtype=float)
        max_lin = float(np.max(lin)) if lin.size else 0.0
        max_ang = float(np.max(ang)) if ang.size else 0.0
        max_stretch = float(np.max(np.abs(_read_array(stretch_data)))) if stretch_data is not None else 0.0
        edge_lengths = _read_array(edge_len_data)
        abnormal_edge = int(_read_scalar(abnormal_idx_data, default=-1.0))
        abnormal_len = _read_scalar(abnormal_len_data)
        abnormal_ref = _read_scalar(abnormal_ref_data)
        axial_err = _read_scalar(axial_err_data)
        lateral_err = _read_scalar(lateral_err_data)
        min_clearance = _read_scalar(min_clearance_data, default=float('inf'))
        barrier_nodes = int(_read_scalar(barrier_nodes_data, default=0.0))
        max_head_stretch = _read_scalar(head_stretch_data)
        torque_sin = _read_scalar(torque_sin_data)
        outward_assist = _read_scalar(outward_assist_data)
        commanded_push = float(getattr(controller, 'commanded_push_mm', 0.0))
        drive_push = float(getattr(controller, 'drive_push_mm', 0.0))
        tip_progress = float(getattr(controller, 'tip_progress_raw_mm', 0.0))
        tip_axial_progress_mm = float(getattr(controller, 'tip_axial_progress_mm', 0.0))
        centers = np.asarray(dofs.position.value, dtype=float)
        centers = centers[:, :3] if centers.ndim == 2 and centers.shape[1] >= 3 else np.zeros((0, 3), dtype=float)
        base_progress_mm = 0.0
        mid_progress_mm = 0.0
        if centers.shape == initial_centers.shape and centers.shape[0] > 0:
            base_progress_mm = float(np.dot(centers[base_index] - initial_centers[base_index], insertion_dir))
            mid_progress_mm = float(np.dot(centers[mid_index] - initial_centers[mid_index], insertion_dir))
        if step <= 3:
            visual_sync_max_error_mm = max(visual_sync_max_error_mm, _visual_sync_error_mm(root))
        finite = bool(np.isfinite(np.asarray(dofs.position.value, dtype=float)).all() and np.isfinite(velocities).all())
        if args.profile == 'push-only':
            monotonic_tol_mm = 0.05
            if (
                base_progress_mm + monotonic_tol_mm < prev_base_progress_mm
                or mid_progress_mm + monotonic_tol_mm < prev_mid_progress_mm
                or tip_progress + monotonic_tol_mm < prev_tip_progress_mm
            ):
                print(
                    f'[FAIL] profile={args.profile} non-monotonic whole-rod progress: '
                    f'baseProgressMm={base_progress_mm:.4f} prevBase={prev_base_progress_mm:.4f} '
                    f'midProgressMm={mid_progress_mm:.4f} prevMid={prev_mid_progress_mm:.4f} '
                    f'tipPathProgress={tip_progress:.4f} prevTipPath={prev_tip_progress_mm:.4f}'
                )
                return 1
            prev_base_progress_mm = max(prev_base_progress_mm, base_progress_mm)
            prev_mid_progress_mm = max(prev_mid_progress_mm, mid_progress_mm)
            prev_tip_progress_mm = max(prev_tip_progress_mm, tip_progress)

        strict_mode = bool(getattr(controller, 'is_native_strict', False))
        wall_contact_clearance_mm = float(getattr(controller, 'wall_contact_clearance_mm', float('inf')))
        wall_contact_active = bool(getattr(controller, 'wall_contact_active', False))
        surface_wall_clearance_mm = float(getattr(controller, 'surface_wall_contact_clearance_mm', float('inf')))
        native_wall_gap_mm = (
            float(controller._native_strict_actual_wall_gap_mm())
            if strict_mode and hasattr(controller, '_native_strict_actual_wall_gap_mm')
            else float('inf')
        )
        physical_wall_clearance_mm = (
            float(controller._native_strict_physical_contact_clearance_mm())
            if strict_mode and hasattr(controller, '_native_strict_physical_contact_clearance_mm')
            else wall_contact_clearance_mm
        )
        effective_min_clearance = float(min_clearance)
        if strict_mode:
            strict_clearance_candidates = [
                float(v)
                for v in (physical_wall_clearance_mm, surface_wall_clearance_mm)
                if np.isfinite(v)
            ]
            if strict_clearance_candidates:
                effective_min_clearance = float(min(strict_clearance_candidates))
            elif np.isfinite(native_wall_gap_mm):
                effective_min_clearance = float(native_wall_gap_mm)
        else:
            safe_clearance_candidates = [
                float(v)
                for v in (surface_wall_clearance_mm, wall_contact_clearance_mm)
                if np.isfinite(v)
            ]
            if safe_clearance_candidates:
                # Safe mode uses the profile barrier as a stabilizing internal
                # guide. It can become much more conservative than the exact
                # vessel surface near branches, so benchmark pass/fail should be
                # based on the exact wall clearance instead of the barrier's
                # internal early-warning metric.
                effective_min_clearance = float(min(safe_clearance_candidates))

        if args.print_every > 0 and (step == 1 or step % int(args.print_every) == 0):
            print(
                f'[STEP] profile={args.profile} step={step} elapsed={elapsed:.2f}s '
                f'finite={finite} maxLin={max_lin:.4f} mm/s maxAng={max_ang:.4f} rad/s '
                f'maxStretch={max_stretch:.4e} abnormalEdge={abnormal_edge} '
                f'axialErr={axial_err:.4f} mm lateralErr={lateral_err:.4f} mm '
                f'minClearance={effective_min_clearance:.4f} mm barrierNodes={barrier_nodes} '
                f'headStretch={max_head_stretch:.4e} torqueSin={torque_sin:.4f} outwardAssist={outward_assist:.4e} N '
                f'wallContact={wall_contact_active} wallClearance={wall_contact_clearance_mm:.4f} mm '
                f'surfaceClearance={surface_wall_clearance_mm:.4f} mm nativeGap={native_wall_gap_mm:.4f} mm '
                f'physicalClearance={physical_wall_clearance_mm:.4f} mm '
                f'commandedPush={commanded_push:.4f} mm drivePush={drive_push:.4f} mm '
                f'baseProgressMm={base_progress_mm:.4f} midProgressMm={mid_progress_mm:.4f} '
                f'tipProgress={tip_axial_progress_mm:.4f} mm tipPathProgress={tip_progress:.4f} mm '
                f'visualSyncMaxError={visual_sync_max_error_mm:.4e} mm'
            )

        speed_threshold_hit = (
            max_lin >= runtime.safe_recovery_linear_speed_mm_s
            or max_ang >= runtime.safe_recovery_angular_speed_rad_s
        )
        safe_recovered_this_step = bool(
            (not strict_mode)
            and int(getattr(controller, '_native_safe_last_recovery_step', -1)) == int(step)
        )
        if strict_mode and speed_threshold_hit:
            print(
                f'[WARN] profile={args.profile} step={step} speed spike observed without physical failure: '
                f'maxLin={max_lin:.4f} mm/s maxAng={max_ang:.4f} rad/s '
                f'maxStretch={max_stretch:.4e} headStretch={max_head_stretch:.4e}'
            )

        strict_surface_fail = strict_mode and (
            (wall_contact_active and physical_wall_clearance_mm < -0.02)
            or (surface_wall_clearance_mm < -0.02)
            or (native_wall_gap_mm < -0.02)
            or (effective_min_clearance < -0.02)
        )
        fail = (
            (not finite)
            or (abnormal_edge >= 0)
            or (max_stretch > runtime.strict_global_stretch_hard_limit)
            or (max_head_stretch > runtime.strict_head_stretch_limit)
            or (max_stretch >= float(args.abort_on_stretch))
            or ((not strict_mode) and speed_threshold_hit)
            or ((not strict_mode) and effective_min_clearance < -0.02)
            or strict_surface_fail
            or (step <= 3 and not np.isfinite(visual_sync_max_error_mm))
        )
        if fail and not safe_recovered_this_step:
            head = np.round(edge_lengths[: min(4, edge_lengths.size)], 4).tolist() if edge_lengths.size else []
            tail = np.round(edge_lengths[max(0, edge_lengths.size - 4):], 4).tolist() if edge_lengths.size else []
            print(
                f'[FAIL] profile={args.profile} step={step} elapsed={elapsed:.2f}s '
                f'finite={finite} maxLin={max_lin:.4f} mm/s maxAng={max_ang:.4f} rad/s '
                f'maxStretch={max_stretch:.4e} abnormalEdge={abnormal_edge} '
                f'abnormalEdgeLen={abnormal_len:.4f} mm abnormalEdgeRef={abnormal_ref:.4f} mm '
                f'axialErr={axial_err:.4f} mm lateralErr={lateral_err:.4f} mm '
                f'minClearance={effective_min_clearance:.4f} mm barrierNodes={barrier_nodes} '
                f'headStretch={max_head_stretch:.4e} torqueSin={torque_sin:.4f} outwardAssist={outward_assist:.4e} N '
                f'wallContact={wall_contact_active} wallClearance={wall_contact_clearance_mm:.4f} mm '
                f'surfaceClearance={surface_wall_clearance_mm:.4f} mm nativeGap={native_wall_gap_mm:.4f} mm '
                f'physicalClearance={physical_wall_clearance_mm:.4f} mm '
                f'commandedPush={commanded_push:.4f} mm drivePush={drive_push:.4f} mm '
                f'baseProgressMm={base_progress_mm:.4f} midProgressMm={mid_progress_mm:.4f} '
                f'tipProgress={tip_axial_progress_mm:.4f} mm tipPathProgress={tip_progress:.4f} mm '
                f'visualSyncMaxError={visual_sync_max_error_mm:.4e} mm '
                f'edgeLenHead={head} edgeLenTail={tail}'
            )
            return 1
        if safe_recovered_this_step:
            print(
                f'[INFO] profile={args.profile} step={step} safe recovery applied; '
                'skip fail-fast evaluation for this step and continue.'
            )

    elapsed = time.perf_counter() - t0
    if args.profile == 'gui-benchmark':
        commanded_push = float(getattr(controller, 'commanded_push_mm', 0.0))
        drive_push = float(getattr(controller, 'drive_push_mm', 0.0))
        tip_progress = float(getattr(controller, 'tip_progress_raw_mm', 0.0))
        route_length_mm = max(float(getattr(controller, 'max_push_mm', 0.0)), float(getattr(controller, 'path_len', 0.0)), 1.0)
        projected_wallclock_min = (
            float('inf')
            if tip_progress <= 1.0e-6 or elapsed <= 1.0e-9
            else (elapsed * route_length_mm / tip_progress) / 60.0
        )
        backlog_mm = commanded_push - tip_progress
        print(
            f'[BENCHMARK] profile={args.profile} steps={args.steps} elapsed={elapsed:.2f}s '
            f'commandedPush={commanded_push:.4f} mm drivePush={drive_push:.4f} mm tipProgress={tip_progress:.4f} mm '
            f'backlog={backlog_mm:.4f} mm projectedTraversalWallclockMin={projected_wallclock_min:.4f}'
        )
        if not np.isfinite(projected_wallclock_min) or projected_wallclock_min > 10.0:
            print(f'[FAIL] profile={args.profile} projectedTraversalWallclockMin={projected_wallclock_min:.4f} exceeds 10.0')
            return 1

    final_positions = np.asarray(dofs.position.value, dtype=float)
    final_centers = final_positions[:, :3] if final_positions.ndim == 2 and final_positions.shape[1] >= 3 else np.zeros((0, 3), dtype=float)
    final_base_progress_mm = 0.0
    final_mid_progress_mm = 0.0
    final_tip_progress_mm = float(getattr(controller, 'tip_progress_raw_mm', 0.0))
    final_tip_axial_progress_mm = float(getattr(controller, 'tip_axial_progress_mm', 0.0))
    if final_centers.shape == initial_centers.shape and final_centers.shape[0] > 0:
        final_base_progress_mm = float(np.dot(final_centers[base_index] - initial_centers[base_index], insertion_dir))
        final_mid_progress_mm = float(np.dot(final_centers[mid_index] - initial_centers[mid_index], insertion_dir))

    profile_mode = args.profile
    if profile_mode in {'push-only', 'full'} and int(args.steps) >= 120:
        if final_base_progress_mm <= 0.5 or final_mid_progress_mm <= 0.5 or final_tip_progress_mm < 1.5:
            print(
                f'[FAIL] profile={args.profile} whole-rod motion insufficient: '
                f'baseProgressMm={final_base_progress_mm:.4f} midProgressMm={final_mid_progress_mm:.4f} '
                f'tipProgress={final_tip_axial_progress_mm:.4f} mm tipPathProgress={final_tip_progress_mm:.4f} mm'
            )
            return 1
        if profile_mode == 'push-only':
            spread_mm = max(final_base_progress_mm, final_mid_progress_mm) - min(
                final_base_progress_mm,
                final_mid_progress_mm,
            )
            if spread_mm > 0.5:
                print(
                    f'[FAIL] profile={args.profile} whole-rod co-translation lost: '
                    f'baseProgressMm={final_base_progress_mm:.4f} midProgressMm={final_mid_progress_mm:.4f} '
                    f'tipProgress={final_tip_axial_progress_mm:.4f} mm tipPathProgress={final_tip_progress_mm:.4f} mm '
                    f'spread={spread_mm:.4f} mm'
                )
                return 1
        if not np.isfinite(visual_sync_max_error_mm) or visual_sync_max_error_mm > 1.0e-6:
            print(
                f'[FAIL] profile={args.profile} visual sync drift detected: '
                f'visualSyncMaxError={visual_sync_max_error_mm:.4e} mm'
            )
            return 1
        if profile_mode == 'full':
            final_torque_sin = _read_scalar(torque_sin_data)
            if final_torque_sin <= 1.0e-6:
                print(
                    f'[FAIL] profile={args.profile} magnetic torque vanished: '
                    f'torqueSin={final_torque_sin:.4e}'
                )
                return 1
            if final_tip_progress_mm > 3.0 and min(final_base_progress_mm, final_mid_progress_mm) < 1.0:
                print(
                    f'[FAIL] profile={args.profile} tip outran proximal body: '
                    f'baseProgressMm={final_base_progress_mm:.4f} midProgressMm={final_mid_progress_mm:.4f} '
                    f'tipProgress={final_tip_axial_progress_mm:.4f} mm tipPathProgress={final_tip_progress_mm:.4f} mm'
                )
                return 1

    print(
        f'[PASS] profile={args.profile} steps={args.steps} elapsed={elapsed:.2f}s '
        f'baseProgressMm={final_base_progress_mm:.4f} midProgressMm={final_mid_progress_mm:.4f} '
        f'tipProgress={final_tip_axial_progress_mm:.4f} mm tipPathProgress={final_tip_progress_mm:.4f} mm '
        f'visualSyncMaxError={visual_sync_max_error_mm:.4e} mm'
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    runtime_pre = _preparse_runtime_args(argv)
    runtime = _load_runtime(str(runtime_pre.profile))
    args = _parse_args(argv, abort_on_stretch_default=runtime.abort_on_stretch_default)
    return run_case(args, runtime)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
