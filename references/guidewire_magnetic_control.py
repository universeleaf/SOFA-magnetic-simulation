# -*- coding: utf-8 -*-
"""
guidewire_magnetic_control.py
============================
导丝磁场控制模块。

设计目标：
1) 显式维护“全局匀强磁场”状态，而不是在控制器里散落若干磁场公式。
2) 对磁性头段仅施加匀强磁场下的磁力矩 m x B，不施加净平移力。
3) 允许将磁场向量同步到真正的 SOFA MagneticField 组件；若环境没有该组件，
   仍可通过等效力矩保持相同的物理控制接口。

注意：
- 这里的“真实磁场”是指：显式建模 B 向量，并由 m x B 推出磁力矩。
- 如果运行环境未安装 MagneticField / MagneticDipole 组件，本模块仍可工作，
  但力矩最终由上层场景写入 ConstantForceField。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def normalize(v: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    if n < eps:
        return np.zeros(3, dtype=float)
    return arr / n


def set_magnetic_field_vector(mag_field_obj, vector_xyz: Sequence[float]) -> None:
    """兼容不同 MagneticField 组件的数据字段名字。"""
    vec = [float(vector_xyz[0]), float(vector_xyz[1]), float(vector_xyz[2])]
    for field_name in ("field", "magneticField", "B", "value", "direction"):
        try:
            data = getattr(mag_field_obj, field_name)
            data.value = vec
            return
        except Exception:
            continue
    raise RuntimeError("未找到 MagneticField 向量数据字段")


@dataclass
class UniformMagneticFieldState:
    """全局匀强磁场状态。"""

    strength: float
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))

    def __post_init__(self) -> None:
        self.direction = normalize(self.direction)
        if np.linalg.norm(self.direction) < 1e-12:
            self.direction = np.array([0.0, 0.0, 1.0], dtype=float)

    @property
    def vector(self) -> np.ndarray:
        return self.direction * float(self.strength)

    def set_direction(self, direction: Sequence[float]) -> np.ndarray:
        d = normalize(direction)
        if np.linalg.norm(d) > 1e-12:
            self.direction = d
        return self.direction.copy()


@dataclass
class DistalMagneticHeadModel:
    """磁性头段参数，按 elasticRod 的思路只让最后 5 条边磁化。"""

    moment_magnitude: float
    head_edge_count: int = 5
    distribution_exp: float = 1.8

    def node_count(self) -> int:
        return self.head_edge_count + 1

    def distribution_weights(self, count: int | None = None) -> np.ndarray:
        n = int(max(1, count if count is not None else self.node_count()))
        raw = np.array([(i + 1) ** float(self.distribution_exp) for i in range(n)], dtype=float)
        return raw / float(np.sum(raw))

    def head_direction(self, rigid_positions: np.ndarray) -> np.ndarray:
        """
        用磁头段平均切向定义磁偶极方向。
        这比单独拿最后一个节点的局部坐标轴更符合“头段磁化”的物理含义。
        """
        pts = np.asarray(rigid_positions, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return np.array([0.0, 0.0, 1.0], dtype=float)

        head_n = int(min(self.node_count(), pts.shape[0]))
        head_pts = pts[-head_n:, :3]
        if head_pts.shape[0] < 2:
            return normalize(pts[-1, :3] - pts[-2, :3])

        segs = np.diff(head_pts, axis=0)
        acc = np.zeros(3, dtype=float)
        for i, seg in enumerate(segs):
            acc += float(i + 1) * normalize(seg)
        d = normalize(acc)
        if np.linalg.norm(d) > 1e-12:
            return d
        return normalize(head_pts[-1] - head_pts[0])


class UniformMagneticFieldController:
    """
    负责三件事：
    1) 由“当前磁偶极方向 -> 目标方向”反解匀强磁场方向。
    2) 保持磁场方向平滑，避免帧间剧烈抖动。
    3) 由 B 与 m 计算磁力矩，并按头段节点分配到 6D wrench。
    """

    def __init__(
        self,
        strength: float,
        moment_magnitude: float,
        smoothing_alpha: float = 0.4,
        min_torque_sin: float = 0.22,
        parallel_gain: float = 0.25,
        initial_direction: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> None:
        self.field = UniformMagneticFieldState(strength=float(strength), direction=np.asarray(initial_direction, dtype=float))
        self.moment_magnitude = float(moment_magnitude)
        self.smoothing_alpha = float(smoothing_alpha)
        self.min_torque_sin = float(min_torque_sin)
        self.parallel_gain = float(parallel_gain)
        self.filtered_direction = self.field.direction.copy()

    def align_field_to_target(self, moment_direction: Sequence[float], target_direction: Sequence[float]) -> np.ndarray:
        desired = normalize(target_direction)
        if np.linalg.norm(desired) < 1e-12:
            desired = self.filtered_direction.copy()

        m_dir = normalize(moment_direction)
        if np.linalg.norm(m_dir) < 1e-12:
            m_dir = desired

        tau_dir = normalize(np.cross(m_dir, desired))
        if np.linalg.norm(tau_dir) < 1e-12:
            fallback = np.array([0.0, 0.0, 1.0], dtype=float)
            tau_dir = normalize(np.cross(m_dir, fallback))
        if np.linalg.norm(tau_dir) < 1e-12:
            tau_dir = np.array([0.0, 0.0, 1.0], dtype=float)

        # 令 B 同时包含：
        # 1) 垂直于 m 的分量，用于产生转向力矩；
        # 2) 少量平行于 m 的分量，用于稳定方向，避免数值抖动。
        b_perp = normalize(np.cross(tau_dir, m_dir))
        b_candidate = normalize(b_perp + self.parallel_gain * m_dir)

        torque_sin = float(np.linalg.norm(np.cross(m_dir, b_candidate)))
        if torque_sin < self.min_torque_sin:
            perp2 = normalize(np.cross(m_dir, tau_dir))
            if np.linalg.norm(perp2) > 1e-12:
                b_candidate = normalize(b_candidate + (self.min_torque_sin - torque_sin) * perp2)

        alpha = float(np.clip(self.smoothing_alpha, 0.0, 1.0))
        self.filtered_direction = normalize((1.0 - alpha) * self.filtered_direction + alpha * b_candidate)
        if np.linalg.norm(self.filtered_direction) < 1e-12:
            self.filtered_direction = b_candidate

        self.field.set_direction(self.filtered_direction)
        return self.field.vector.copy()

    def magnetic_torque(self, moment_direction: Sequence[float], boost: float = 1.0) -> np.ndarray:
        m_dir = normalize(moment_direction)
        if np.linalg.norm(m_dir) < 1e-12:
            return np.zeros(3, dtype=float)
        return np.cross(m_dir, self.field.direction) * (self.moment_magnitude * self.field.strength * float(boost))

    def magnetic_force(
        self,
        moment_direction: Sequence[float],
        target_direction: Sequence[float],
        gradient_gain: float,
        boost: float = 1.0,
        forward_gain: float = 0.05,
    ) -> np.ndarray:
        """
        ?????????????????????????????????
        ???????????????? desired ?????
        """
        desired = normalize(target_direction)
        if np.linalg.norm(desired) < 1e-12 or gradient_gain <= 0.0:
            return np.zeros(3, dtype=float)

        m_dir = normalize(moment_direction)
        if np.linalg.norm(m_dir) < 1e-12:
            m_dir = desired

        sin_err = float(np.linalg.norm(np.cross(m_dir, desired)))
        alignment_gain = max(0.25, sin_err)
        magnitude = float(gradient_gain) * float(boost) * alignment_gain
        return desired * magnitude

    def distribute_wrench_as_wrenches(
        self,
        force_world: Sequence[float],
        torque_world: Sequence[float],
        weights: Sequence[float],
    ) -> list[list[float]]:
        force = np.asarray(force_world, dtype=float).reshape(3)
        torque = np.asarray(torque_world, dtype=float).reshape(3)
        out: list[list[float]] = []
        for w in np.asarray(weights, dtype=float):
            wf = force * float(w)
            wt = torque * float(w)
            out.append([float(wf[0]), float(wf[1]), float(wf[2]), float(wt[0]), float(wt[1]), float(wt[2])])
        return out

    def distribute_torque_as_wrenches(self, torque_world: Sequence[float], weights: Sequence[float]) -> list[list[float]]:
        return self.distribute_wrench_as_wrenches([0.0, 0.0, 0.0], torque_world, weights)

    def sync_to_sofa_field(self, mag_field_obj) -> None:
        if mag_field_obj is None:
            return
        set_magnetic_field_vector(mag_field_obj, self.field.vector.tolist())


class CylindricalMagnetFieldController(UniformMagneticFieldController):
    """
    Approximate a cylindrical permanent magnet with a softened dipole field.

    The goal here is not exact magnetostatics at every point of the cylinder,
    but a stable, physically motivated field direction plus a bounded radial
    pull that can recenter the distal guidewire segment inside the lumen.
    """

    def __init__(
        self,
        strength: float,
        moment_magnitude: float,
        smoothing_alpha: float = 0.25,
        min_torque_sin: float = 0.16,
        parallel_gain: float = 0.15,
        initial_direction: Sequence[float] = (0.0, 0.0, 1.0),
        magnet_radius_mm: float = 8.0,
        magnet_length_mm: float = 24.0,
        magnet_standoff_mm: float = 18.0,
        softening_mm: float = 6.0,
        max_field_strength: float | None = None,
        max_force_n: float = 0.06,
        forward_force_gain: float = 0.0,
    ) -> None:
        super().__init__(
            strength=strength,
            moment_magnitude=moment_magnitude,
            smoothing_alpha=smoothing_alpha,
            min_torque_sin=min_torque_sin,
            parallel_gain=parallel_gain,
            initial_direction=initial_direction,
        )
        self.magnet_radius_mm = float(max(magnet_radius_mm, 1.0e-3))
        self.magnet_length_mm = float(max(magnet_length_mm, 1.0e-3))
        self.magnet_standoff_mm = float(max(magnet_standoff_mm, 0.0))
        self.softening_mm = float(max(softening_mm, 1.0e-3))
        self.max_field_strength = float(max_field_strength) if max_field_strength is not None else float(strength)
        self.max_force_n = float(max(max_force_n, 0.0))
        self.forward_force_gain = float(max(forward_force_gain, 0.0))
        radius_m = 1.0e-3 * self.magnet_radius_mm
        length_m = 1.0e-3 * self.magnet_length_mm
        self._volume_m3 = float(np.pi * radius_m * radius_m * length_m)

    def _steering_candidate(self, moment_direction: Sequence[float], target_direction: Sequence[float]) -> np.ndarray:
        desired = normalize(target_direction)
        if np.linalg.norm(desired) < 1.0e-12:
            desired = self.filtered_direction.copy()

        m_dir = normalize(moment_direction)
        if np.linalg.norm(m_dir) < 1.0e-12:
            m_dir = desired

        tau_dir = normalize(np.cross(m_dir, desired))
        if np.linalg.norm(tau_dir) < 1.0e-12:
            tau_dir = normalize(np.cross(m_dir, np.array([0.0, 0.0, 1.0], dtype=float)))
        if np.linalg.norm(tau_dir) < 1.0e-12:
            tau_dir = np.array([0.0, 0.0, 1.0], dtype=float)

        b_perp = normalize(np.cross(tau_dir, m_dir))
        candidate = normalize(b_perp + self.parallel_gain * m_dir)
        if np.linalg.norm(candidate) < 1.0e-12:
            candidate = desired
        return candidate

    def _softened_dipole_field(
        self,
        sample_position_mm: Sequence[float],
        magnet_center_mm: Sequence[float],
        magnet_axis: Sequence[float],
    ) -> tuple[np.ndarray, float]:
        axis = normalize(magnet_axis)
        if np.linalg.norm(axis) < 1.0e-12:
            axis = self.filtered_direction.copy()

        sample_m = 1.0e-3 * np.asarray(sample_position_mm, dtype=float).reshape(3)
        center_m = 1.0e-3 * np.asarray(magnet_center_mm, dtype=float).reshape(3)
        r = sample_m - center_m
        eps_m = 1.0e-3 * self.softening_mm
        r2 = float(np.dot(r, r) + eps_m * eps_m)
        rnorm = float(np.sqrt(max(r2, 1.0e-18)))

        # Equivalent dipole moment aligned with the cylindrical magnet axis.
        dipole_moment = axis * max(self._volume_m3, 1.0e-18)
        mdotr = float(np.dot(dipole_moment, r))
        raw_field = 3.0 * r * mdotr / max(rnorm ** 5, 1.0e-18) - dipole_moment / max(rnorm ** 3, 1.0e-18)
        field_dir = normalize(raw_field)
        if np.linalg.norm(field_dir) < 1.0e-12:
            field_dir = axis

        char_len_mm = max(self.magnet_radius_mm + 0.5 * self.magnet_length_mm, 1.0e-3)
        distance_mm = float(np.linalg.norm(np.asarray(sample_position_mm, dtype=float).reshape(3) - np.asarray(magnet_center_mm, dtype=float).reshape(3)))
        decay = (char_len_mm / max(distance_mm + self.softening_mm, 1.0e-6)) ** 3
        magnitude = float(np.clip(self.field.strength * max(0.25, decay), 0.0, self.max_field_strength))
        return field_dir, magnitude

    def command_cylindrical_magnet(
        self,
        sample_position_mm: Sequence[float],
        moment_direction: Sequence[float],
        target_point_mm: Sequence[float],
        target_direction: Sequence[float],
        axis_point_mm: Sequence[float] | None = None,
        radial_force_gain: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_dir = normalize(target_direction)
        if np.linalg.norm(target_dir) < 1.0e-12:
            target_dir = self.filtered_direction.copy()

        target_point = np.asarray(target_point_mm, dtype=float).reshape(3)
        magnet_center = target_point - self.magnet_standoff_mm * target_dir
        field_dir, magnitude = self._softened_dipole_field(
            sample_position_mm=sample_position_mm,
            magnet_center_mm=magnet_center,
            magnet_axis=target_dir,
        )

        m_dir = normalize(moment_direction)
        if np.linalg.norm(m_dir) > 1.0e-12 and self.min_torque_sin > 0.0:
            torque_sin = float(np.linalg.norm(np.cross(m_dir, field_dir)))
            if torque_sin < self.min_torque_sin:
                steer_dir = self._steering_candidate(m_dir, target_dir)
                blend = float(np.clip((self.min_torque_sin - torque_sin) / max(self.min_torque_sin, 1.0e-6), 0.0, 1.0))
                field_dir = normalize((1.0 - blend) * field_dir + blend * steer_dir)
                if np.linalg.norm(field_dir) < 1.0e-12:
                    field_dir = steer_dir

        alpha = float(np.clip(self.smoothing_alpha, 0.0, 1.0))
        self.filtered_direction = normalize((1.0 - alpha) * self.filtered_direction + alpha * field_dir)
        if np.linalg.norm(self.filtered_direction) < 1.0e-12:
            self.filtered_direction = field_dir

        self.field.strength = float(np.clip(magnitude, 0.0, self.max_field_strength))
        self.field.set_direction(self.filtered_direction)

        torque = self.magnetic_torque(moment_direction)

        sample = np.asarray(sample_position_mm, dtype=float).reshape(3)
        axis_anchor = (
            np.asarray(axis_point_mm, dtype=float).reshape(3)
            if axis_point_mm is not None
            else target_point
        )
        axial = float(np.dot(sample - axis_anchor, target_dir))
        closest_axis_point = axis_anchor + axial * target_dir
        radial_vec = closest_axis_point - sample
        radial_dist = float(np.linalg.norm(radial_vec))
        force = np.zeros(3, dtype=float)
        if radial_dist > 1.0e-9 and radial_force_gain > 0.0 and self.max_force_n > 0.0:
            radial_dir = radial_vec / radial_dist
            radial_gate = float(np.tanh(radial_dist / max(self.softening_mm, 1.0e-6)))
            force_mag = min(
                self.max_force_n,
                float(radial_force_gain) * float(self.field.strength) * radial_gate,
            )
            force += force_mag * radial_dir
        if self.forward_force_gain > 0.0 and self.max_force_n > 0.0:
            forward_mag = min(self.max_force_n, self.forward_force_gain * float(self.field.strength))
            force += forward_mag * target_dir

        return self.field.vector.copy(), torque, force
