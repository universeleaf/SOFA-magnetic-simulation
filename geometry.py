# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np

from .config import (
    CENTERLINE_FILE,
    CENTERLINE_CANDIDATES,
    CENTERLINE_LUMEN_NEAREST_K,
    CENTERLINE_LUMEN_RADIUS_PERCENTILE,
    INITIAL_ENTRY_ADVANCE_MM,
    INITIAL_TIP_ARC_MM,
    ROUTE_REVISIT_MIN_ARC_MM,
    ROUTE_REVISIT_MIN_INDEX_GAP,
    ROUTE_REVISIT_TOLERANCE_MM,
    WIRE_RADIUS_MM,
    WIRE_TOTAL_LENGTH_MM,
)
from .math_utils import _cumlen, _interp, _normalize, _parallel_transport, _quat_from_basis, _quat_from_z_to, _tangent


def _load_obj_vertices_faces(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v' and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f' and len(parts) >= 4:
                faces.append([int(chunk.split('/')[0]) - 1 for chunk in parts[1:4]])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def _drop_nonlocal_revisits(points: np.ndarray) -> np.ndarray:
    """
    路径提取结果里有时会出现“沿主干走上去，再绕一圈回到同一点”的回折段。

    这种数据在屏幕上看起来就像路径中途长出分叉，但本质上并不是两条合法通路，
    而是一条 polyline 自己回到了先前已经经过的位置。

    这里的清理原则很保守：
    - 只在两个非相邻点几乎重合时才裁掉中间整段；
    - 而且要求被裁掉的中间弧长明显大于端点间直线距离；
    - 因此不会去“拍扁”正常弯曲，只会消掉明显的折返闭环。
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < ROUTE_REVISIT_MIN_INDEX_GAP + 2:
        return pts

    keep = np.ones(pts.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(pts[:, :3], axis=0), axis=1) > 1e-6
    pts = pts[keep]

    removed_loops = 0
    while pts.shape[0] >= ROUTE_REVISIT_MIN_INDEX_GAP + 2:
        cum = _cumlen(pts[:, :3])
        best = None
        for i in range(pts.shape[0] - ROUTE_REVISIT_MIN_INDEX_GAP - 1):
            for j in range(i + ROUTE_REVISIT_MIN_INDEX_GAP, pts.shape[0]):
                distance = float(np.linalg.norm(pts[i, :3] - pts[j, :3]))
                if distance > ROUTE_REVISIT_TOLERANCE_MM:
                    continue
                arc = float(cum[j] - cum[i])
                if arc < ROUTE_REVISIT_MIN_ARC_MM:
                    continue
                if best is None or arc > best[0]:
                    best = (arc, i, j, distance)
        if best is None:
            break
        _, i, j, _ = best
        pts = np.vstack((pts[: i + 1], pts[j:]))
        keep = np.ones(pts.shape[0], dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(pts[:, :3], axis=0), axis=1) > 1e-6
        pts = pts[keep]
        removed_loops += 1

    if removed_loops > 0:
        print(
            f'[INFO] Route cleanup removed {removed_loops} folded revisit segment(s): '
            f'tol={ROUTE_REVISIT_TOLERANCE_MM:.3f} mm, minArc={ROUTE_REVISIT_MIN_ARC_MM:.1f} mm'
        )
    return pts


def _load_centerline() -> Tuple[np.ndarray, Path]:
    """
    中心线加载优先走显式指定的目标分支文件。

    这样“导丝要走哪一条血管”不再依赖文件存在顺序的隐式回退；
    如果后续你要切到别的分支，只需要在 config.py 里改 `CENTERLINE_FILE` 即可。
    """
    path = CENTERLINE_FILE if CENTERLINE_FILE.exists() else next((p for p in CENTERLINE_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError('未找到中心线 npy 文件。')
    pts = np.asarray(np.load(path), dtype=float)
    pts = np.atleast_2d(pts)
    if pts.shape[1] > 3:
        pts = pts[:, :3]
    # 路线文件统一按“入口在下、出口在上”解释。
    # 这样导丝入口、初始插入方向和后续导航都只围绕一套语义工作，
    # 不会因为某个 npy 文件恰好是反向存储就把整条路径倒过来。
    if pts.shape[0] >= 2 and float(pts[0, 1]) > float(pts[-1, 1]):
        pts = pts[::-1].copy()
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.ones(pts.shape[0], dtype=bool)
    keep[1:] = diffs > 1e-6
    pts = pts[keep]
    pts = _drop_nonlocal_revisits(pts)
    if pts.shape[0] < 2:
        raise ValueError('中心线至少需要 2 个点。')
    cum = _cumlen(pts)
    total = float(cum[-1])
    # 把原始中心线按约 2 mm 重采样，后续最近段查找和腔内半径估计会更稳定。
    count = max(2, int(math.ceil(total / 2.0)) + 1)
    resampled = np.asarray([_interp(pts, cum, s) for s in np.linspace(0.0, total, count)], dtype=float)
    return resampled, path


def _closest_point_on_triangle_fast(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, 1.0e-12)
        return a + v * ab

    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, 1.0e-12)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        bc = c - b
        w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1.0e-12)
        return b + w * bc

    denom = max(va + vb + vc, 1.0e-12)
    v = vb / denom
    w = vc / denom
    return a + ab * v + ac * w


def _closest_point_on_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return _closest_point_on_triangle_fast(
        np.asarray(point, dtype=float).reshape(3),
        np.asarray(a, dtype=float).reshape(3),
        np.asarray(b, dtype=float).reshape(3),
        np.asarray(c, dtype=float).reshape(3),
    )


class _NearestSurface:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray | None = None, face_candidate_count: int = 96):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces if faces is not None else [], dtype=int).reshape(-1, 3)
        self.face_candidate_count = int(max(face_candidate_count, 8))
        self.vertex_candidate_count = int(min(max(self.face_candidate_count // 2, 24), 128))
        self.face_centroids = np.zeros((0, 3), dtype=float)
        self.face_vertices = np.zeros((0, 3, 3), dtype=float)
        self.face_normals = np.zeros((0, 3), dtype=float)
        self.vertex_face_ids: list[np.ndarray] = []
        if self.faces.shape[0] > 0 and self.vertices.shape[0] > 0:
            valid = np.all((self.faces >= 0) & (self.faces < self.vertices.shape[0]), axis=1)
            self.faces = self.faces[valid]
            if self.faces.shape[0] > 0:
                self.face_vertices = self.vertices[self.faces]
                self.face_centroids = np.mean(self.face_vertices, axis=1)
                raw_normals = np.cross(
                    self.face_vertices[:, 1] - self.face_vertices[:, 0],
                    self.face_vertices[:, 2] - self.face_vertices[:, 0],
                )
                self.face_normals = np.zeros_like(raw_normals)
                normal_norms = np.linalg.norm(raw_normals, axis=1)
                valid_normals = normal_norms > 1.0e-12
                if np.any(valid_normals):
                    self.face_normals[valid_normals] = (
                        raw_normals[valid_normals]
                        / normal_norms[valid_normals].reshape(-1, 1)
                    )
                adjacency: list[list[int]] = [[] for _ in range(self.vertices.shape[0])]
                for face_id, tri in enumerate(self.faces.tolist()):
                    adjacency[int(tri[0])].append(face_id)
                    adjacency[int(tri[1])].append(face_id)
                    adjacency[int(tri[2])].append(face_id)
                self.vertex_face_ids = [np.asarray(ids, dtype=int) for ids in adjacency]

    def query(self, point) -> tuple[float, np.ndarray, np.ndarray]:
        p = np.asarray(point, dtype=float).reshape(3)
        if self.face_vertices.shape[0] > 0:
            candidate_id_parts: list[np.ndarray] = []
            d2 = np.sum((self.face_centroids - p.reshape(1, 3)) ** 2, axis=1)
            centroid_candidate_count = min(self.face_candidate_count, d2.shape[0])
            if centroid_candidate_count > 0:
                candidate_id_parts.append(np.argpartition(d2, centroid_candidate_count - 1)[:centroid_candidate_count])

            if self.vertices.ndim == 2 and self.vertices.shape[0] > 0 and self.vertex_face_ids:
                vertex_d2 = np.sum((self.vertices[:, :3] - p.reshape(1, 3)) ** 2, axis=1)
                vertex_candidate_count = min(self.vertex_candidate_count, vertex_d2.shape[0])
                if vertex_candidate_count > 0:
                    vertex_ids = np.argpartition(vertex_d2, vertex_candidate_count - 1)[:vertex_candidate_count]
                    face_sets = [self.vertex_face_ids[int(vertex_id)] for vertex_id in vertex_ids.tolist()]
                    face_sets = [face_ids for face_ids in face_sets if face_ids.size > 0]
                    if face_sets:
                        candidate_id_parts.append(np.unique(np.concatenate(face_sets)))

            if candidate_id_parts:
                candidate_ids = np.unique(np.concatenate(candidate_id_parts))
            else:
                candidate_ids = np.arange(self.face_vertices.shape[0], dtype=int)
            best_distance = float('inf')
            best_point = self.face_vertices[int(candidate_ids[0]), 0].copy()
            best_normal = np.array([0.0, 0.0, 1.0], dtype=float)
            for face_id in candidate_ids:
                tri = self.face_vertices[face_id]
                q = _closest_point_on_triangle_fast(p, tri[0], tri[1], tri[2])
                delta = p - q
                dist = float(np.linalg.norm(delta))
                if dist < best_distance:
                    best_distance = dist
                    best_point = q
                    normal = self.face_normals[face_id]
                    if float(np.dot(normal, normal)) > 1.0e-24:
                        best_normal = normal
                    elif dist > 1.0e-12:
                        best_normal = delta / dist
            return best_distance, best_point, best_normal

        p = np.asarray(point, dtype=float).reshape(3)
        if self.vertices.ndim != 2 or self.vertices.shape[0] == 0:
            return float('inf'), p.copy(), np.array([0.0, 0.0, 1.0], dtype=float)
        delta = self.vertices[:, :3] - p.reshape(1, 3)
        d = np.linalg.norm(delta, axis=1)
        idx = int(np.argmin(d))
        dist = float(d[idx])
        normal = delta[idx] / dist if dist > 1.0e-12 else np.array([0.0, 0.0, 1.0], dtype=float)
        return dist, self.vertices[idx, :3].copy(), normal

    def distance(self, point) -> float:
        distance, _, _ = self.query(point)
        return float(distance)



def _opening_radius(vertices: np.ndarray, entry: np.ndarray, direction: np.ndarray) -> float:
    rel = np.asarray(vertices, dtype=float)[:, :3] - np.asarray(entry, dtype=float).reshape(1, 3)
    axis = _normalize(direction)
    axial = rel @ axis.reshape(3, 1)
    sample = rel[np.abs(axial.reshape(-1)) <= 6.0]
    if sample.shape[0] < 8:
        sample = rel
    radial = sample - np.outer(sample @ axis, axis)
    return float(np.clip(np.percentile(np.linalg.norm(radial, axis=1), 35.0), 1.8, 6.0))



def _lumen_profile(
    centerline: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray | None = None,
    face_candidate_count: int = 1024,
) -> np.ndarray:
    pts = np.asarray(centerline, dtype=float)[:, :3]
    verts = np.asarray(vertices, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or verts.ndim != 2 or verts.shape[0] == 0:
        return np.zeros(0, dtype=float)

    k = max(8, min(CENTERLINE_LUMEN_NEAREST_K, verts.shape[0]))
    approx_radii = []
    for p in pts:
        d = np.linalg.norm(verts[:, :3] - p.reshape(1, 3), axis=1)
        near = np.partition(d, k - 1)[:k]
        approx_radii.append(float(np.percentile(near, CENTERLINE_LUMEN_RADIUS_PERCENTILE)))
    approx = np.asarray(approx_radii, dtype=float)

    face_array = np.asarray(faces if faces is not None else [], dtype=int).reshape(-1, 3)
    if face_array.shape[0] <= 0:
        return np.clip(approx, 0.75, 5.5)

    surface = _NearestSurface(verts[:, :3], face_array, face_candidate_count=face_candidate_count)
    exact = np.asarray([surface.distance(p) for p in pts], dtype=float)
    radii = np.minimum(approx, exact)
    return np.clip(radii, 0.75, 5.5)



def _initial_wire_state(
    centerline: np.ndarray,
    n_nodes: int,
    insertion_dir: np.ndarray,
    vessel_query: _NearestSurface,
    max_external_length_mm: float | None = None,
    initial_tip_insertion_mm: float | None = None,
    total_length_mm: float = WIRE_TOTAL_LENGTH_MM,
    wire_radius_mm: float = WIRE_RADIUS_MM,
    smooth_entry_transition: bool = False,
    entry_blend_length_mm: float = 0.0,
    initial_axis_hold_mm: float = 0.0,
):
    """
    生成初始直导丝状态。

    支持两种初始化语义：
    - `initial_tip_insertion_mm`: 直接指定导丝头相对入口沿中心线进入多深；
    - `max_external_length_mm`: 旧语义，按体外保留长度反推头部弧长。

    native strict 默认走第一种语义，确保画面打开时“导丝头只刚进入口一点”。
    """
    cum = _cumlen(centerline)
    total = float(cum[-1])
    total_length_mm = float(max(total_length_mm, 0.0))
    ds = total_length_mm / max(n_nodes - 1, 1)
    blend_len = float(max(entry_blend_length_mm, 0.0)) if smooth_entry_transition else 0.0
    axis_hold_len = float(max(initial_axis_hold_mm, 0.0))
    min_tip_arc = INITIAL_TIP_ARC_MM + INITIAL_ENTRY_ADVANCE_MM
    tip_arc_floor = float(INITIAL_ENTRY_ADVANCE_MM)
    explicit_tip_insertion = initial_tip_insertion_mm is not None
    if explicit_tip_insertion:
        # Strict elasticrod uses an explicit "tip just entered" initialization.
        # Do not clamp it back up to the shared beam pre-entry advance, otherwise
        # the tip starts beyond the first bend and immediately kinks on release.
        tip_arc_floor = 0.0
        tip_arc = max(float(initial_tip_insertion_mm), 0.0)
    elif max_external_length_mm is not None:
        desired_tip_arc = total_length_mm - float(max_external_length_mm)
        tip_arc = max(min_tip_arc, desired_tip_arc)
    else:
        tip_arc = min_tip_arc
    tip_arc = min(float(tip_arc), total)

    while tip_arc >= tip_arc_floor - 1e-9:
        centers, signed_s = [], []
        for i in range(n_nodes):
            arc_to_tip = (n_nodes - 1 - i) * ds
            s_i = tip_arc - arc_to_tip
            signed_s.append(float(s_i))
            if s_i >= 0.0:
                if explicit_tip_insertion:
                    # For explicit native insertion depth, the in-lumen segment must
                    # already lie on the vessel centerline. Keeping the first
                    # inserted millimeters on the external insertion axis injects a
                    # large entry misalignment, so the tip contacts the wall almost
                    # immediately and appears to "bend in place" instead of moving
                    # forward.
                    pos = _interp(centerline, cum, s_i)
                elif axis_hold_len > 1.0e-9 and s_i <= axis_hold_len:
                    pos = centerline[0, :3] + s_i * insertion_dir
                elif axis_hold_len > 1.0e-9 and smooth_entry_transition and blend_len > axis_hold_len + 1.0e-9 and s_i < blend_len:
                    transition_span = max(blend_len - axis_hold_len, 1.0e-9)
                    p0 = centerline[0, :3] + axis_hold_len * insertion_dir
                    p1 = _interp(centerline, cum, blend_len)
                    m0 = insertion_dir * transition_span
                    m1 = _tangent(centerline, cum, blend_len) * transition_span
                    u = float(np.clip((s_i - axis_hold_len) / transition_span, 0.0, 1.0))
                    u2 = u * u
                    u3 = u2 * u
                    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
                    h10 = u3 - 2.0 * u2 + u
                    h01 = -2.0 * u3 + 3.0 * u2
                    h11 = u3 - u2
                    pos = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
                elif smooth_entry_transition and blend_len > 1.0e-9 and s_i < blend_len:
                    p0 = centerline[0, :3] - blend_len * insertion_dir
                    p1 = _interp(centerline, cum, blend_len)
                    m0 = insertion_dir * (2.0 * blend_len)
                    m1 = _tangent(centerline, cum, blend_len) * (2.0 * blend_len)
                    u = float(np.clip((s_i + blend_len) / (2.0 * blend_len), 0.0, 1.0))
                    u2 = u * u
                    u3 = u2 * u
                    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
                    h10 = u3 - 2.0 * u2 + u
                    h01 = -2.0 * u3 + 3.0 * u2
                    h11 = u3 - u2
                    pos = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
                else:
                    pos = _interp(centerline, cum, s_i)
            else:
                if explicit_tip_insertion:
                    # Outside the vessel the guidewire remains straight along the
                    # insertion axis; the introducer/support corridor takes over
                    # after scene construction.
                    pos = centerline[0, :3] + s_i * insertion_dir
                elif axis_hold_len > 1.0e-9 and s_i > -blend_len:
                    pos = centerline[0, :3] + s_i * insertion_dir
                elif smooth_entry_transition and blend_len > 1.0e-9 and s_i > -blend_len:
                    p0 = centerline[0, :3] - blend_len * insertion_dir
                    p1 = _interp(centerline, cum, blend_len)
                    m0 = insertion_dir * (2.0 * blend_len)
                    m1 = _tangent(centerline, cum, blend_len) * (2.0 * blend_len)
                    u = float(np.clip((s_i + blend_len) / (2.0 * blend_len), 0.0, 1.0))
                    u2 = u * u
                    u3 = u2 * u
                    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
                    h10 = u3 - 2.0 * u2 + u
                    h01 = -2.0 * u3 + 3.0 * u2
                    h11 = u3 - u2
                    pos = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
                else:
                    pos = centerline[0, :3] + s_i * insertion_dir
            centers.append(pos)
        centers_arr = np.asarray(centers, dtype=float)
        edge_tangents = []
        for i in range(max(n_nodes - 1, 0)):
            edge = centers_arr[i + 1, :3] - centers_arr[i, :3]
            tan = _normalize(edge)
            if np.linalg.norm(tan) < 1.0e-12:
                tan = insertion_dir.copy()
            edge_tangents.append(tan)

        quats = []
        if edge_tangents:
            d1 = np.cross(edge_tangents[0], np.array([0.0, 0.0, -1.0], dtype=float))
            if np.linalg.norm(d1) < 1.0e-12:
                d1 = np.cross(edge_tangents[0], np.array([0.0, 1.0, 0.0], dtype=float))
            d1 = _normalize(d1)
            d2 = _normalize(np.cross(edge_tangents[0], d1))
            d1 = _normalize(np.cross(d2, edge_tangents[0]))
            quats.append(_quat_from_basis(d1, d2, edge_tangents[0]))
            for i in range(1, len(edge_tangents)):
                d1 = _parallel_transport(d1, edge_tangents[i - 1], edge_tangents[i])
                d2 = _normalize(np.cross(edge_tangents[i], d1))
                d1 = _normalize(np.cross(d2, edge_tangents[i]))
                quats.append(_quat_from_basis(d1, d2, edge_tangents[i]))
            quats.append(quats[-1].copy())
        else:
            quats.append(_quat_from_z_to(insertion_dir))

        rigid_arr = np.column_stack((centers_arr, np.asarray(quats, dtype=float)))
        signed_s_arr = np.asarray(signed_s, dtype=float)
        inside = signed_s_arr >= 0.0
        if np.any(inside):
            clearance = min(vessel_query.distance(p) - float(wire_radius_mm) for p in rigid_arr[inside, :3])
        else:
            clearance = float('inf')
        if clearance >= float(wire_radius_mm) + 0.05:
            return rigid_arr, signed_s_arr, float(tip_arc), float(clearance), int(np.sum(inside))
        tip_arc -= 1.0

    raise RuntimeError('初始导丝无法在入口前送后保持安全净间隙，请检查血管入口网格或中心线。')


def _build_open_cylinder_shell(
    center: np.ndarray,
    axis: np.ndarray,
    length_mm: float,
    radius_mm: float,
    radial_segments: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    axis = _normalize(np.asarray(axis, dtype=float).reshape(3))
    center = np.asarray(center, dtype=float).reshape(3)
    length_mm = float(max(length_mm, 1.0e-6))
    radius_mm = float(max(radius_mm, 1.0e-6))
    radial_segments = int(max(radial_segments, 6))

    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(axis, ref))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = _normalize(np.cross(axis, ref))
    v = _normalize(np.cross(axis, u))

    p0 = center - 0.5 * length_mm * axis
    p1 = center + 0.5 * length_mm * axis
    verts = []
    faces = []
    for ring_center in (p0, p1):
        for k in range(radial_segments):
            theta = 2.0 * math.pi * float(k) / float(radial_segments)
            radial = math.cos(theta) * u + math.sin(theta) * v
            verts.append(ring_center + radius_mm * radial)

    for k in range(radial_segments):
        a0 = k
        a1 = (k + 1) % radial_segments
        b0 = radial_segments + k
        b1 = radial_segments + (k + 1) % radial_segments
        faces.append([a0, b0, a1])
        faces.append([a1, b0, b1])

    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)
