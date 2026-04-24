from __future__ import annotations

import heapq
import json
import shutil
import struct
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT_DIR / 'guidewire_scene'
ASSETS_DIR = PACKAGE_DIR / 'assets'
CENTERLINE_DIR = ASSETS_DIR / 'centerline'
EXTRACTED_DIR = CENTERLINE_DIR / 'extracted_paths'
ROUTE_DIR = EXTRACTED_DIR / 'routes'

EXTERNAL_SOURCE_VESSEL_MESH = ROOT_DIR / 'vessel_final_4_0108.stl'
EXTERNAL_SOURCE_CENTERLINE_POINTS = ROOT_DIR / 'vessel_centerline_4_0108.npy'

PROJECT_VESSEL_MESH = ASSETS_DIR / 'vessel_final_4_0108.stl'
PROJECT_RUNTIME_VESSEL_MESH = ASSETS_DIR / 'vessel_final_4_0108_runtime.obj'
PROJECT_RUNTIME_MESH_INFO = ASSETS_DIR / 'vessel_final_4_0108_runtime_info.json'
PROJECT_CENTERLINE_POINTS = CENTERLINE_DIR / 'vessel_centerline_4_0108.npy'

ROUTE_NAME = 'vessel_4_0108_full'
ROUTE_FILE = ROUTE_DIR / f'route_{ROUTE_NAME}.npy'
ALIGNED_POINTS_FILE = CENTERLINE_DIR / 'vessel_centerline_4_0108_aligned.npy'
ALIGNMENT_INFO_FILE = CENTERLINE_DIR / 'vessel_centerline_4_0108_alignment.json'
SELECTED_ROUTE_FILE = EXTRACTED_DIR / 'selected_route.json'
ROUTE_CATALOG_FILE = EXTRACTED_DIR / 'route_catalog.json'
PREFERRED_ROUTE_START_INDEX = 532
PREFERRED_ROUTE_END_INDEX = 7177

LOWER_ENDPOINT_BAND_RAW = 40.0
UPPER_ENDPOINT_BAND_RAW = 60.0
RUNTIME_MESH_QUANTIZATION_MM = 0.7
ROUTE_RESAMPLE_STEP_MM = 1.0
ROUTE_SMOOTH_PASSES_PRE = 4
ROUTE_SMOOTH_PASSES_POST = 3
ROUTE_CENTERING_RADIUS_MM = 5.5
ROUTE_CENTERING_BLEND = 0.40
ROUTE_CENTERING_MIN_NEIGHBORS = 8
ROUTE_CENTERING_MAX_SHIFT_MM = 1.50
ROUTE_SURFACE_CENTERING_PASSES = 3
ROUTE_SURFACE_CENTERING_BLEND = 0.72
ROUTE_SURFACE_CENTERING_SEARCH_RADII_MM = (1.20, 0.60, 0.28)
ROUTE_SURFACE_CENTERING_MAX_SHIFT_MM = 2.20
ROUTE_SURFACE_CENTERING_GUIDE_PENALTY = 0.16
ROUTE_SURFACE_CENTERING_SMOOTH_PENALTY = 0.24
ROUTE_SURFACE_CENTERING_FACE_CANDIDATES = 160
ROUTE_SURFACE_CENTERING_SMOOTH_PASSES = 1
ROUTE_SURFACE_CENTERING_MESH_QUANTIZATION_MM = 1.40
ROUTE_MIN_SEGMENT_MM = 0.25
ROUTE_ENTRY_AXIS_DIRECTION_SAMPLE_MM = 6.0
ROUTE_ENTRY_AXIS_LOCK_LENGTH_MM = 8.0
ROUTE_ENTRY_AXIS_BLEND_OUT_LENGTH_MM = 14.0
ROUTE_ENTRY_AXIS_MAX_SHIFT_MM = 4.0
ROUTE_ENTRY_AXIS_STATS_SAMPLE_MM = 12.0
RUNTIME_OPEN_INLET_ENABLED = True
RUNTIME_OPEN_INLET_BACKOFF_MM = 0.0
RUNTIME_OPEN_PLANE_EPS_MM = 1.0e-6
RUNTIME_INLET_BOUNDARY_MIN_COMPONENT_SIZE = 12


def _resolve_existing_path(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_text = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(f'Unable to locate {label}. Checked: {candidate_text}')


def _copy_if_needed(source: Path, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return False
    shutil.copy2(source, destination)
    return True


def _load_ascii_stl_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    with path.open('r', encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == 'vertex':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    verts = np.asarray(vertices, dtype=np.float32)
    if verts.shape[0] == 0 or verts.shape[0] % 3 != 0:
        raise ValueError(f'ASCII STL parse failed or triangle count is invalid: {path}')
    faces = np.arange(verts.shape[0], dtype=np.int32).reshape(-1, 3)
    return verts, faces


def _load_stl_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open('rb') as handle:
        handle.read(80)
        tri_count_bytes = handle.read(4)
        if len(tri_count_bytes) != 4:
            raise ValueError(f'Invalid STL header: {path}')
        tri_count = struct.unpack('<I', tri_count_bytes)[0]

    expected_size = 84 + tri_count * 50
    if path.stat().st_size == expected_size:
        dtype = np.dtype(
            [
                ('normal', '<f4', (3,)),
                ('vertices', '<f4', (3, 3)),
                ('attr', '<u2'),
            ]
        )
        data = np.fromfile(path, dtype=dtype, count=tri_count, offset=84)
        vertices = np.ascontiguousarray(data['vertices'].reshape(-1, 3), dtype=np.float32)
        faces = np.arange(vertices.shape[0], dtype=np.int32).reshape(-1, 3)
        return vertices, faces

    return _load_ascii_stl_vertices_faces(path)


def _closest_point_on_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=float).reshape(3)
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    c = np.asarray(c, dtype=float).reshape(3)

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


class _NearestSurface:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, face_candidate_count: int = 96):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        self.face_candidate_count = int(max(face_candidate_count, 16))
        self.vertex_candidate_count = int(min(max(self.face_candidate_count // 2, 24), 128))
        self.face_vertices = np.zeros((0, 3, 3), dtype=float)
        self.face_centroids = np.zeros((0, 3), dtype=float)
        self.vertex_face_ids: list[np.ndarray] = []
        if self.vertices.shape[0] == 0 or self.faces.shape[0] == 0:
            return
        valid = np.all((self.faces >= 0) & (self.faces < self.vertices.shape[0]), axis=1)
        self.faces = self.faces[valid]
        if self.faces.shape[0] == 0:
            return
        self.face_vertices = self.vertices[self.faces]
        self.face_centroids = np.mean(self.face_vertices, axis=1)
        adjacency: list[list[int]] = [[] for _ in range(self.vertices.shape[0])]
        for face_id, tri in enumerate(self.faces.tolist()):
            adjacency[int(tri[0])].append(face_id)
            adjacency[int(tri[1])].append(face_id)
            adjacency[int(tri[2])].append(face_id)
        self.vertex_face_ids = [np.asarray(ids, dtype=np.int32) for ids in adjacency]

    def candidate_faces(self, point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=float).reshape(3)
        if self.face_vertices.shape[0] == 0:
            return np.zeros(0, dtype=np.int32)

        candidate_parts: list[np.ndarray] = []
        centroid_d2 = np.sum((self.face_centroids - p.reshape(1, 3)) ** 2, axis=1)
        centroid_count = min(self.face_candidate_count, centroid_d2.shape[0])
        if centroid_count > 0:
            candidate_parts.append(np.argpartition(centroid_d2, centroid_count - 1)[:centroid_count])

        if self.vertex_face_ids:
            vertex_d2 = np.sum((self.vertices - p.reshape(1, 3)) ** 2, axis=1)
            vertex_count = min(self.vertex_candidate_count, vertex_d2.shape[0])
            if vertex_count > 0:
                vertex_ids = np.argpartition(vertex_d2, vertex_count - 1)[:vertex_count]
                face_sets = [self.vertex_face_ids[int(vertex_id)] for vertex_id in vertex_ids.tolist()]
                face_sets = [face_ids for face_ids in face_sets if face_ids.size > 0]
                if face_sets:
                    candidate_parts.append(np.unique(np.concatenate(face_sets)))

        return (
            np.unique(np.concatenate(candidate_parts))
            if candidate_parts
            else np.arange(self.face_vertices.shape[0], dtype=np.int32)
        )

    def distance_to_faces(self, point: np.ndarray, face_ids: np.ndarray | None = None) -> float:
        p = np.asarray(point, dtype=float).reshape(3)
        if self.face_vertices.shape[0] == 0:
            if self.vertices.shape[0] == 0:
                return float('inf')
            delta = self.vertices - p.reshape(1, 3)
            return float(np.min(np.linalg.norm(delta, axis=1)))
        candidate_ids = (
            np.asarray(face_ids, dtype=np.int32).reshape(-1)
            if face_ids is not None and np.asarray(face_ids).size > 0
            else np.arange(self.face_vertices.shape[0], dtype=np.int32)
        )
        best_distance = float('inf')
        for face_id in candidate_ids.tolist():
            tri = self.face_vertices[int(face_id)]
            q = _closest_point_on_triangle(p, tri[0], tri[1], tri[2])
            best_distance = min(best_distance, float(np.linalg.norm(p - q)))
        return best_distance

    def distance(self, point: np.ndarray) -> float:
        return self.distance_to_faces(point, self.candidate_faces(point))


def _structured_view_int3(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=np.int32)
    return contiguous.view([('x', np.int32), ('y', np.int32), ('z', np.int32)]).reshape(-1)


def _build_runtime_mesh(vertices: np.ndarray, faces: np.ndarray, quantization_mm: float) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    quantized = np.rint(np.asarray(vertices, dtype=np.float64) / float(quantization_mm)).astype(np.int32)
    unique_view, inverse = np.unique(_structured_view_int3(quantized), return_inverse=True)
    unique_grid = unique_view.view(np.int32).reshape(-1, 3)

    counts = np.bincount(inverse, minlength=unique_grid.shape[0]).astype(np.float64)
    reduced_vertices = np.zeros((unique_grid.shape[0], 3), dtype=np.float64)
    for axis in range(3):
        reduced_vertices[:, axis] = np.bincount(
            inverse,
            weights=np.asarray(vertices[:, axis], dtype=np.float64),
            minlength=unique_grid.shape[0],
        ) / np.maximum(counts, 1.0)

    reduced_faces = inverse[np.asarray(faces, dtype=np.int32)]
    valid = (
        (reduced_faces[:, 0] != reduced_faces[:, 1])
        & (reduced_faces[:, 1] != reduced_faces[:, 2])
        & (reduced_faces[:, 0] != reduced_faces[:, 2])
    )
    reduced_faces = reduced_faces[valid]

    reduced_face_keys = np.sort(reduced_faces, axis=1)
    _, unique_face_indices = np.unique(_structured_view_int3(reduced_face_keys), return_index=True)
    reduced_faces = reduced_faces[np.sort(unique_face_indices)]

    used_vertices = np.unique(reduced_faces.reshape(-1))
    remap = np.full(reduced_vertices.shape[0], -1, dtype=np.int32)
    remap[used_vertices] = np.arange(used_vertices.shape[0], dtype=np.int32)
    reduced_vertices = reduced_vertices[used_vertices]
    reduced_faces = remap[reduced_faces]

    stats = {
        'quantization_mm': float(quantization_mm),
        'input_vertex_count': int(vertices.shape[0]),
        'input_face_count': int(faces.shape[0]),
        'cluster_vertex_count': int(unique_grid.shape[0]),
        'output_vertex_count': int(reduced_vertices.shape[0]),
        'output_face_count': int(reduced_faces.shape[0]),
    }
    return np.asarray(reduced_vertices, dtype=float), np.asarray(reduced_faces, dtype=np.int32), stats


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write('# Auto-generated runtime vessel mesh for SOFA beam collision\n')
        for vertex in np.asarray(vertices, dtype=float):
            handle.write(f'v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n')
        for face in np.asarray(faces, dtype=np.int32):
            a, b, c = face + 1
            handle.write(f'f {a} {b} {c}\n')


def _compact_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=float)
    tri = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if tri.shape[0] == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int32)
    used = np.unique(tri.reshape(-1))
    remap = np.full(verts.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)
    return np.asarray(verts[used], dtype=float), np.asarray(remap[tri], dtype=np.int32)


def _cut_mesh_with_plane(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    *,
    keep_positive_side: bool,
    eps_mm: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | list[float]]]:
    verts = np.asarray(vertices, dtype=float)
    tri = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    point = np.asarray(plane_point, dtype=float).reshape(3)
    normal = np.asarray(plane_normal, dtype=float).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1.0e-12:
        raise RuntimeError('Plane normal is degenerate while cutting runtime vessel mesh.')
    normal = normal / normal_norm

    signed = (verts - point.reshape(1, 3)) @ normal.reshape(3, 1)
    signed = signed.reshape(-1)
    if not keep_positive_side:
        signed = -signed

    face_signed = signed[tri]
    keep_faces = np.all(face_signed >= -float(eps_mm), axis=1)
    cut_faces = tri[keep_faces]
    cut_vertices, cut_faces = _compact_mesh(verts, cut_faces)
    stats = {
        'input_vertex_count': int(verts.shape[0]),
        'input_face_count': int(tri.shape[0]),
        'output_vertex_count': int(cut_vertices.shape[0]),
        'output_face_count': int(cut_faces.shape[0]),
        'removed_face_count': int(tri.shape[0] - cut_faces.shape[0]),
        'plane_point_mm': point.tolist(),
        'plane_normal': normal.tolist(),
        'keep_positive_side': bool(keep_positive_side),
    }
    return cut_vertices, cut_faces, stats


def _open_runtime_mesh_inlet(
    vertices: np.ndarray,
    faces: np.ndarray,
    route: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool | list[float]]]:
    verts = np.asarray(vertices, dtype=float)
    tri = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    path = np.asarray(route, dtype=float)
    if (not bool(RUNTIME_OPEN_INLET_ENABLED)) or path.shape[0] < 2:
        return verts, tri, {
            'enabled': bool(RUNTIME_OPEN_INLET_ENABLED),
            'applied': False,
            'reason': 'disabled_or_route_too_short',
        }

    tangent = np.asarray(path[1, :3] - path[0, :3], dtype=float).reshape(3)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1.0e-12:
        return verts, tri, {
            'enabled': True,
            'applied': False,
            'reason': 'degenerate_route_tangent',
        }
    tangent /= tangent_norm
    plane_point = np.asarray(path[0, :3], dtype=float).reshape(3) - float(RUNTIME_OPEN_INLET_BACKOFF_MM) * tangent
    cut_vertices, cut_faces, cut_stats = _cut_mesh_with_plane(
        verts,
        tri,
        plane_point,
        tangent,
        keep_positive_side=True,
        eps_mm=float(RUNTIME_OPEN_PLANE_EPS_MM),
    )
    return cut_vertices, cut_faces, {
        'enabled': True,
        'applied': True,
        'backoff_mm': float(RUNTIME_OPEN_INLET_BACKOFF_MM),
        'route_start_mm': np.asarray(path[0, :3], dtype=float).tolist(),
        'route_tangent': tangent.tolist(),
        **cut_stats,
    }


def _boundary_components(faces: np.ndarray) -> list[np.ndarray]:
    tri = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if tri.shape[0] == 0:
        return []

    edge_counts: dict[tuple[int, int], int] = {}
    for a, b, c in tri.tolist():
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    adjacency: dict[int, list[int]] = {}
    for (u, v), count in edge_counts.items():
        if count != 1:
            continue
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    components: list[np.ndarray] = []
    visited: set[int] = set()
    for seed in adjacency:
        if seed in visited:
            continue
        stack = [seed]
        vertices_in_component: list[int] = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            vertices_in_component.append(node)
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(np.asarray(sorted(vertices_in_component), dtype=np.int32))
    return components


def _build_adjacency(points: np.ndarray) -> tuple[list[list[tuple[int, float]]], np.ndarray]:
    points_int = np.asarray(points, dtype=int)
    point_to_index = {tuple(point.tolist()): idx for idx, point in enumerate(points_int)}
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == dy == dz == 0)
    ]
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(points_int.shape[0])]
    degrees = np.zeros(points_int.shape[0], dtype=int)
    for idx, point in enumerate(points_int):
        x, y, z = point.tolist()
        for dx, dy, dz in offsets:
            neighbor = point_to_index.get((x + dx, y + dy, z + dz))
            if neighbor is None:
                continue
            weight = float(np.linalg.norm(points[neighbor] - points[idx]))
            adjacency[idx].append((neighbor, weight))
        degrees[idx] = len(adjacency[idx])
    return adjacency, degrees


def _dijkstra(adjacency: list[list[tuple[int, float]]], start: int) -> tuple[np.ndarray, np.ndarray]:
    distance = np.full(len(adjacency), np.inf, dtype=float)
    previous = np.full(len(adjacency), -1, dtype=int)
    distance[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance != distance[node]:
            continue
        for neighbor, weight in adjacency[node]:
            new_distance = current_distance + weight
            if new_distance < distance[neighbor]:
                distance[neighbor] = new_distance
                previous[neighbor] = node
                heapq.heappush(queue, (new_distance, neighbor))
    return distance, previous


def _reconstruct_path(previous: np.ndarray, start: int, end: int) -> np.ndarray:
    path = []
    node = int(end)
    while node != -1:
        path.append(node)
        if node == start:
            break
        node = int(previous[node])
    if not path or path[-1] != start:
        raise RuntimeError(f'Unable to reconstruct a path from endpoint {start} to {end}')
    path.reverse()
    return np.asarray(path, dtype=int)


def _route_turn_statistics(route: np.ndarray) -> tuple[float, float]:
    if route.shape[0] < 3:
        return 0.0, 0.0
    v0 = route[1:-1] - route[:-2]
    v1 = route[2:] - route[1:-1]
    n0 = np.linalg.norm(v0, axis=1)
    n1 = np.linalg.norm(v1, axis=1)
    valid = (n0 > 1.0e-9) & (n1 > 1.0e-9)
    if not np.any(valid):
        return 0.0, 0.0
    cosine = np.sum(v0[valid] * v1[valid], axis=1) / (n0[valid] * n1[valid])
    cosine = np.clip(cosine, -1.0, 1.0)
    turns = np.degrees(np.arccos(cosine))
    return float(turns.mean()), float(turns.max())


def _select_main_route(points: np.ndarray) -> tuple[np.ndarray, dict[str, float | int | list[float]]]:
    adjacency, degrees = _build_adjacency(points)
    endpoints = np.where(degrees == 1)[0]
    if endpoints.size < 2:
        raise RuntimeError('Endpoint extraction failed: fewer than 2 degree-1 endpoints found.')

    min_y = float(np.min(points[:, 1]))
    max_y = float(np.max(points[:, 1]))
    start_candidates = [int(idx) for idx in endpoints.tolist() if float(points[idx, 1]) <= min_y + LOWER_ENDPOINT_BAND_RAW]
    end_candidates = [int(idx) for idx in endpoints.tolist() if float(points[idx, 1]) >= max_y - UPPER_ENDPOINT_BAND_RAW]
    if not start_candidates:
        start_candidates = [int(endpoints[np.argmin(points[endpoints, 1])])]
    if not end_candidates:
        end_candidates = [int(endpoints[np.argmax(points[endpoints, 1])])]

    best_path_indices: np.ndarray | None = None
    best_metrics: dict[str, float | int | list[float]] | None = None

    for start in start_candidates:
        distance, previous = _dijkstra(adjacency, start)
        for end in end_candidates:
            if end == start or not np.isfinite(distance[end]):
                continue
            vertical_span = float(points[end, 1] - points[start, 1])
            if vertical_span <= 0.0:
                continue
            straight_distance = float(np.linalg.norm(points[end] - points[start]))
            tortuosity = float(distance[end] / max(straight_distance, 1.0e-9))
            path_indices = _reconstruct_path(previous, start, end)
            route = np.asarray(points[path_indices], dtype=float)
            mean_turn_deg, max_turn_deg = _route_turn_statistics(route)
            metrics = {
                'start_index': int(start),
                'end_index': int(end),
                'vertical_span_raw': vertical_span,
                'geodesic_length_raw': float(distance[end]),
                'straight_length_raw': straight_distance,
                'tortuosity': tortuosity,
                'mean_turn_deg': mean_turn_deg,
                'max_turn_deg': max_turn_deg,
                'path_point_count': int(route.shape[0]),
                'start_point_raw': route[0].tolist(),
                'end_point_raw': route[-1].tolist(),
            }
            if best_metrics is None:
                best_path_indices = path_indices
                best_metrics = metrics
                continue
            if vertical_span > float(best_metrics['vertical_span_raw']) + 1.0e-6:
                best_path_indices = path_indices
                best_metrics = metrics
                continue
            if abs(vertical_span - float(best_metrics['vertical_span_raw'])) <= 1.0e-6:
                if tortuosity < float(best_metrics['tortuosity']) - 1.0e-6:
                    best_path_indices = path_indices
                    best_metrics = metrics
                    continue
                if abs(tortuosity - float(best_metrics['tortuosity'])) <= 1.0e-6:
                    if float(distance[end]) > float(best_metrics['geodesic_length_raw']) + 1.0e-6:
                        best_path_indices = path_indices
                        best_metrics = metrics

    if best_path_indices is None or best_metrics is None:
        raise RuntimeError('Failed to select a valid lower-inlet to upper-outlet route.')
    return np.asarray(points[best_path_indices], dtype=float), best_metrics


def _select_route_between_endpoints(
    points: np.ndarray,
    start_index: int,
    end_index: int,
) -> tuple[np.ndarray, dict[str, float | int | list[float] | str]]:
    pts = np.asarray(points, dtype=float)
    if start_index < 0 or start_index >= pts.shape[0] or end_index < 0 or end_index >= pts.shape[0]:
        raise IndexError('Preferred route endpoint index is out of bounds.')

    adjacency, _ = _build_adjacency(pts)
    distance, previous = _dijkstra(adjacency, int(start_index))
    if not np.isfinite(distance[int(end_index)]):
        raise RuntimeError('Preferred route endpoints are not connected.')

    path_indices = _reconstruct_path(previous, int(start_index), int(end_index))
    route = np.asarray(pts[path_indices], dtype=float)
    mean_turn_deg, max_turn_deg = _route_turn_statistics(route)
    straight_distance = float(np.linalg.norm(route[-1] - route[0]))
    return route, {
        'selection_mode': 'preferred_endpoints',
        'start_index': int(start_index),
        'end_index': int(end_index),
        'vertical_span_raw': float(route[-1, 1] - route[0, 1]),
        'geodesic_length_raw': float(distance[int(end_index)]),
        'straight_length_raw': straight_distance,
        'tortuosity': float(distance[int(end_index)] / max(straight_distance, 1.0e-9)),
        'mean_turn_deg': mean_turn_deg,
        'max_turn_deg': max_turn_deg,
        'path_point_count': int(route.shape[0]),
        'start_point_raw': route[0].tolist(),
        'end_point_raw': route[-1].tolist(),
    }


def _build_bbox_alignment(points: np.ndarray, mesh_vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_min = np.min(points, axis=0)
    points_max = np.max(points, axis=0)
    mesh_min = np.min(mesh_vertices, axis=0)
    mesh_max = np.max(mesh_vertices, axis=0)
    scale = (mesh_max - mesh_min) / np.maximum(points_max - points_min, 1.0e-9)
    translation = mesh_min - points_min * scale
    return np.asarray(scale, dtype=float), np.asarray(translation, dtype=float)


def _cumlen(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] <= 1:
        return np.zeros(pts.shape[0], dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def _interp_polyline(points: np.ndarray, cumulative: np.ndarray, arc_mm: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if arc_mm <= 0.0:
        return pts[0].copy()
    if arc_mm >= float(cumulative[-1]):
        return pts[-1].copy()
    idx = int(np.searchsorted(cumulative, arc_mm, side='right') - 1)
    idx = max(0, min(idx, pts.shape[0] - 2))
    seg = float(cumulative[idx + 1] - cumulative[idx])
    if seg <= 1.0e-12:
        return pts[idx].copy()
    alpha = (arc_mm - cumulative[idx]) / seg
    return (1.0 - alpha) * pts[idx] + alpha * pts[idx + 1]


def _resample_polyline(points: np.ndarray, step_mm: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return pts.copy()
    cumulative = _cumlen(pts)
    total = float(cumulative[-1])
    count = max(2, int(np.ceil(total / float(step_mm))) + 1)
    samples = np.linspace(0.0, total, count)
    return np.asarray([_interp_polyline(pts, cumulative, arc) for arc in samples], dtype=float)


def _smooth_polyline(points: np.ndarray, passes: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    if pts.shape[0] < 5 or passes <= 0:
        return pts
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
    kernel /= np.sum(kernel)
    for _ in range(int(passes)):
        out = pts.copy()
        for idx in range(2, pts.shape[0] - 2):
            out[idx] = (
                kernel[0] * pts[idx - 2]
                + kernel[1] * pts[idx - 1]
                + kernel[2] * pts[idx]
                + kernel[3] * pts[idx + 1]
                + kernel[4] * pts[idx + 2]
            )
        out[0] = pts[0]
        out[-1] = pts[-1]
        out[1] = 0.75 * pts[1] + 0.25 * out[2]
        out[-2] = 0.75 * pts[-2] + 0.25 * out[-3]
        pts = out
    return pts


def _normalize(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-12:
        return np.zeros_like(vec)
    return vec / norm


def _plane_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = _normalize(direction)
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(tangent, reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    u = _normalize(np.cross(tangent, reference))
    if float(np.linalg.norm(u)) <= 1.0e-12:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
        u = _normalize(np.cross(tangent, reference))
    v = _normalize(np.cross(tangent, u))
    return u, v


def _project_to_plane(point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=float).reshape(3)
    origin = np.asarray(plane_point, dtype=float).reshape(3)
    normal = _normalize(np.asarray(plane_normal, dtype=float).reshape(3))
    if float(np.linalg.norm(normal)) <= 1.0e-12:
        return p.copy()
    offset = p - origin
    return p - normal * float(np.dot(offset, normal))


def _clip_displacement(point: np.ndarray, anchor: np.ndarray, max_distance_mm: float) -> np.ndarray:
    p = np.asarray(point, dtype=float).reshape(3)
    center = np.asarray(anchor, dtype=float).reshape(3)
    delta = p - center
    norm = float(np.linalg.norm(delta))
    if norm <= float(max_distance_mm) + 1.0e-12:
        return p
    return center + delta * (float(max_distance_mm) / max(norm, 1.0e-12))


def _route_surface_clearance_stats(route: np.ndarray, surface: _NearestSurface) -> dict[str, float]:
    pts = np.asarray(route, dtype=float)
    if pts.shape[0] == 0:
        return {
            'min_mm': 0.0,
            'mean_mm': 0.0,
            'max_mm': 0.0,
            'tail_min_mm': 0.0,
            'tail_mean_mm': 0.0,
        }
    distances = np.asarray([surface.distance(point) for point in pts], dtype=float)
    tail = distances[-min(12, distances.shape[0]):]
    return {
        'min_mm': float(np.min(distances)),
        'mean_mm': float(np.mean(distances)),
        'max_mm': float(np.max(distances)),
        'tail_min_mm': float(np.min(tail)) if tail.size else 0.0,
        'tail_mean_mm': float(np.mean(tail)) if tail.size else 0.0,
    }


def _route_summary(route: np.ndarray) -> dict[str, float | int]:
    pts = np.asarray(route, dtype=float)
    return {
        'point_count': int(pts.shape[0]),
        'length_mm': float(_cumlen(pts)[-1]) if pts.shape[0] >= 2 else 0.0,
        'mean_turn_deg': float(_route_turn_statistics(pts)[0]),
        'max_turn_deg': float(_route_turn_statistics(pts)[1]),
    }


def _estimate_route_entry_axis(route: np.ndarray, sample_mm: float) -> np.ndarray:
    pts = np.asarray(route, dtype=float)
    if pts.shape[0] < 2:
        return np.zeros(3, dtype=float)
    cumulative = _cumlen(pts)
    sample_arc = min(float(sample_mm), float(cumulative[-1]))
    target = _interp_polyline(pts, cumulative, sample_arc)
    axis = _normalize(target - pts[0])
    if float(np.linalg.norm(axis)) > 1.0e-12:
        return axis
    return _normalize(pts[1] - pts[0])


def _axis_distances(points: np.ndarray, axis_origin: np.ndarray, axis_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    origin = np.asarray(axis_origin, dtype=float).reshape(1, 3)
    direction = _normalize(np.asarray(axis_direction, dtype=float).reshape(3))
    delta = pts - origin
    axial = delta @ direction.reshape(3, 1)
    axial = axial.reshape(-1)
    radial = np.linalg.norm(delta - axial[:, None] * direction.reshape(1, 3), axis=1)
    return axial, radial


def _entry_axis_stats(
    route: np.ndarray,
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
    sample_mm: float,
) -> dict[str, float]:
    pts = np.asarray(route, dtype=float)
    if pts.shape[0] == 0:
        return {
            'sample_length_mm': float(sample_mm),
            'mean_radial_mm': 0.0,
            'max_radial_mm': 0.0,
            'start_radial_mm': 0.0,
            'end_radial_mm': 0.0,
        }
    cumulative = _cumlen(pts)
    mask = cumulative <= float(sample_mm) + 1.0e-9
    if not np.any(mask):
        mask[0] = True
    _, radial = _axis_distances(pts[mask], axis_origin, axis_direction)
    return {
        'sample_length_mm': float(sample_mm),
        'mean_radial_mm': float(np.mean(radial)) if radial.size else 0.0,
        'max_radial_mm': float(np.max(radial)) if radial.size else 0.0,
        'start_radial_mm': float(radial[0]) if radial.size else 0.0,
        'end_radial_mm': float(radial[-1]) if radial.size else 0.0,
    }


def _detect_runtime_inlet_opening(
    vertices: np.ndarray,
    faces: np.ndarray,
    route: np.ndarray,
) -> dict[str, float | int | bool | list[float] | str]:
    verts = np.asarray(vertices, dtype=float)
    tri = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    path = np.asarray(route, dtype=float)
    if verts.shape[0] == 0 or tri.shape[0] == 0 or path.shape[0] < 2:
        return {
            'found': False,
            'reason': 'invalid_runtime_mesh_or_route',
        }

    axis_direction = _estimate_route_entry_axis(path, ROUTE_ENTRY_AXIS_DIRECTION_SAMPLE_MM)
    if float(np.linalg.norm(axis_direction)) <= 1.0e-12:
        return {
            'found': False,
            'reason': 'degenerate_entry_axis',
        }
    plane_point = np.asarray(path[0, :3], dtype=float).reshape(3)

    components = _boundary_components(tri)
    if not components:
        return {
            'found': False,
            'reason': 'no_boundary_components_found',
        }

    best_component: dict[str, float | int | bool | list[float]] | None = None
    best_score = float('inf')
    for component in components:
        component_pts = verts[component]
        if component_pts.shape[0] == 0:
            continue
        centroid = np.mean(component_pts, axis=0)
        centroid_axial, centroid_radial = _axis_distances(
            centroid.reshape(1, 3),
            plane_point,
            axis_direction,
        )
        plane_residual = np.abs((component_pts - plane_point.reshape(1, 3)) @ axis_direction.reshape(3, 1)).reshape(-1)
        radius = np.linalg.norm(component_pts - centroid.reshape(1, 3), axis=1)
        size_penalty = 0.0 if component_pts.shape[0] >= int(RUNTIME_INLET_BOUNDARY_MIN_COMPONENT_SIZE) else 50.0
        score = (
            float(centroid_radial[0])
            + 0.35 * abs(float(centroid_axial[0]))
            + 0.25 * float(np.mean(plane_residual))
            + size_penalty
        )
        candidate = {
            'found': True,
            'component_vertex_count': int(component_pts.shape[0]),
            'center_mm': centroid.tolist(),
            'mean_radius_mm': float(np.mean(radius)) if radius.size else 0.0,
            'max_radius_mm': float(np.max(radius)) if radius.size else 0.0,
            'axis_radial_offset_mm': float(centroid_radial[0]),
            'axis_axial_offset_mm': float(centroid_axial[0]),
            'plane_residual_mean_mm': float(np.mean(plane_residual)) if plane_residual.size else 0.0,
            'plane_residual_max_mm': float(np.max(plane_residual)) if plane_residual.size else 0.0,
            'entry_axis_direction': axis_direction.tolist(),
        }
        if score < best_score:
            best_score = score
            best_component = candidate

    if best_component is None:
        return {
            'found': False,
            'reason': 'failed_to_select_boundary_component',
        }
    return {
        **best_component,
        'component_count': int(len(components)),
        'selection_score': float(best_score),
    }


def _recenter_route_from_cloud(
    route: np.ndarray,
    cloud: np.ndarray,
    radius_mm: float,
    blend: float,
    min_neighbors: int,
    max_shift_mm: float,
) -> np.ndarray:
    route_pts = np.asarray(route, dtype=float)
    cloud_pts = np.asarray(cloud, dtype=float)
    adjusted = route_pts.copy()
    radius_sq = float(radius_mm) * float(radius_mm)
    gaussian_den = max(2.0 * radius_sq * 0.25, 1.0e-9)

    for idx in range(1, route_pts.shape[0] - 1):
        delta = cloud_pts - route_pts[idx]
        dist_sq = np.einsum('ij,ij->i', delta, delta)
        mask = dist_sq <= radius_sq
        if int(np.count_nonzero(mask)) < int(min_neighbors):
            continue
        local_points = cloud_pts[mask]
        local_dist_sq = dist_sq[mask]
        weights = np.exp(-local_dist_sq / gaussian_den)
        target = np.sum(local_points * weights[:, None], axis=0) / np.maximum(np.sum(weights), 1.0e-9)
        tangent = _normalize(route_pts[min(idx + 1, route_pts.shape[0] - 1)] - route_pts[max(idx - 1, 0)])
        offset = target - route_pts[idx]
        if float(np.linalg.norm(tangent)) > 1.0e-12:
            offset = offset - tangent * float(np.dot(offset, tangent))
        offset_norm = float(np.linalg.norm(offset))
        if offset_norm > float(max_shift_mm):
            offset *= float(max_shift_mm) / offset_norm
        adjusted[idx] = route_pts[idx] + float(blend) * offset

    adjusted[0] = route_pts[0]
    adjusted[-1] = route_pts[-1]
    return adjusted


def _recenter_route_from_surface(
    route: np.ndarray,
    guide_route: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    route_pts = np.asarray(route, dtype=float)
    guide_pts = np.asarray(guide_route, dtype=float)
    if route_pts.shape[0] < 3 or guide_pts.shape[0] != route_pts.shape[0]:
        return route_pts.copy(), {
            'surface_centering_applied': 0,
            'surface_centering_passes': 0,
            'surface_clearance_min_before_mm': 0.0,
            'surface_clearance_min_after_mm': 0.0,
        }

    surface = _NearestSurface(
        np.asarray(surface_vertices, dtype=float),
        np.asarray(surface_faces, dtype=np.int32),
        face_candidate_count=ROUTE_SURFACE_CENTERING_FACE_CANDIDATES,
    )
    clearance_before = _route_surface_clearance_stats(route_pts, surface)
    working = route_pts.copy()

    def score_candidate(
        candidate: np.ndarray,
        guide_anchor: np.ndarray,
        smooth_anchor: np.ndarray,
        local_face_ids: np.ndarray,
    ) -> tuple[float, float]:
        clearance = surface.distance_to_faces(candidate, local_face_ids)
        guide_error = float(np.sum((candidate - guide_anchor) ** 2))
        smooth_error = float(np.sum((candidate - smooth_anchor) ** 2))
        score = (
            clearance
            - float(ROUTE_SURFACE_CENTERING_GUIDE_PENALTY) * guide_error
            - float(ROUTE_SURFACE_CENTERING_SMOOTH_PENALTY) * smooth_error
        )
        return score, clearance

    for _ in range(int(ROUTE_SURFACE_CENTERING_PASSES)):
        updated = working.copy()
        for idx in range(1, working.shape[0] - 1):
            tangent = _normalize(working[idx + 1] - working[idx - 1])
            if float(np.linalg.norm(tangent)) <= 1.0e-12:
                tangent = _normalize(guide_pts[idx + 1] - guide_pts[idx - 1])
            if float(np.linalg.norm(tangent)) <= 1.0e-12:
                continue

            plane_point = working[idx]
            guide_anchor = _project_to_plane(guide_pts[idx], plane_point, tangent)
            smooth_anchor = _project_to_plane(0.5 * (working[idx - 1] + working[idx + 1]), plane_point, tangent)
            u, v = _plane_basis(tangent)
            local_face_ids = np.unique(
                np.concatenate(
                    [
                        surface.candidate_faces(plane_point),
                        surface.candidate_faces(guide_anchor),
                        surface.candidate_faces(smooth_anchor),
                    ]
                )
            )

            initial_candidates = [
                plane_point,
                guide_anchor,
                smooth_anchor,
                0.5 * (guide_anchor + smooth_anchor),
            ]
            best_point = _clip_displacement(initial_candidates[0], guide_anchor, ROUTE_SURFACE_CENTERING_MAX_SHIFT_MM)
            best_score, _ = score_candidate(best_point, guide_anchor, smooth_anchor, local_face_ids)
            for candidate in initial_candidates[1:]:
                clipped = _clip_displacement(candidate, guide_anchor, ROUTE_SURFACE_CENTERING_MAX_SHIFT_MM)
                score, _ = score_candidate(clipped, guide_anchor, smooth_anchor, local_face_ids)
                if score > best_score:
                    best_score = score
                    best_point = clipped

            for radius in ROUTE_SURFACE_CENTERING_SEARCH_RADII_MM:
                offsets = np.linspace(-float(radius), float(radius), 5)
                stage_best = best_point
                stage_best_score = best_score
                for dx in offsets.tolist():
                    for dy in offsets.tolist():
                        if dx * dx + dy * dy > float(radius) * float(radius) + 1.0e-12:
                            continue
                        candidate = best_point + u * float(dx) + v * float(dy)
                        candidate = _clip_displacement(candidate, guide_anchor, ROUTE_SURFACE_CENTERING_MAX_SHIFT_MM)
                        score, _ = score_candidate(candidate, guide_anchor, smooth_anchor, local_face_ids)
                        if score > stage_best_score:
                            stage_best_score = score
                            stage_best = candidate
                best_point = stage_best
                best_score = stage_best_score

            updated[idx] = working[idx] + float(ROUTE_SURFACE_CENTERING_BLEND) * (best_point - working[idx])

        updated = _smooth_polyline(updated, ROUTE_SURFACE_CENTERING_SMOOTH_PASSES)
        updated[0] = route_pts[0]
        updated[-1] = route_pts[-1]
        working = updated

    clearance_after = _route_surface_clearance_stats(working, surface)
    return working, {
        'surface_centering_applied': 1,
        'surface_centering_passes': int(ROUTE_SURFACE_CENTERING_PASSES),
        'surface_centering_blend': float(ROUTE_SURFACE_CENTERING_BLEND),
        'surface_centering_max_shift_mm': float(ROUTE_SURFACE_CENTERING_MAX_SHIFT_MM),
        'surface_centering_guide_penalty': float(ROUTE_SURFACE_CENTERING_GUIDE_PENALTY),
        'surface_centering_smooth_penalty': float(ROUTE_SURFACE_CENTERING_SMOOTH_PENALTY),
        'surface_clearance_min_before_mm': float(clearance_before['min_mm']),
        'surface_clearance_mean_before_mm': float(clearance_before['mean_mm']),
        'surface_clearance_tail_min_before_mm': float(clearance_before['tail_min_mm']),
        'surface_clearance_min_after_mm': float(clearance_after['min_mm']),
        'surface_clearance_mean_after_mm': float(clearance_after['mean_mm']),
        'surface_clearance_tail_min_after_mm': float(clearance_after['tail_min_mm']),
    }


def _recenter_route_entry_to_axis(
    route: np.ndarray,
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | bool | list[float] | str]]:
    route_pts = np.asarray(route, dtype=float)
    origin = np.asarray(axis_origin, dtype=float).reshape(3)
    direction = _normalize(np.asarray(axis_direction, dtype=float).reshape(3))
    if route_pts.shape[0] < 2:
        return route_pts.copy(), {
            'applied': False,
            'reason': 'route_too_short',
        }
    if float(np.linalg.norm(direction)) <= 1.0e-12:
        return route_pts.copy(), {
            'applied': False,
            'reason': 'degenerate_axis_direction',
        }

    cumulative = _cumlen(route_pts)
    lock_length = min(float(ROUTE_ENTRY_AXIS_LOCK_LENGTH_MM), float(cumulative[-1]))
    blend_out_length = min(max(float(ROUTE_ENTRY_AXIS_BLEND_OUT_LENGTH_MM), lock_length), float(cumulative[-1]))
    adjusted = route_pts.copy()

    before_stats = _entry_axis_stats(
        route_pts,
        origin,
        direction,
        sample_mm=float(ROUTE_ENTRY_AXIS_STATS_SAMPLE_MM),
    )

    for idx, arc_mm in enumerate(cumulative):
        if arc_mm > blend_out_length + 1.0e-9:
            continue
        target = origin + direction * float(arc_mm)
        if arc_mm <= lock_length + 1.0e-9:
            adjusted[idx] = target
            continue
        blend_t = (float(arc_mm) - lock_length) / max(blend_out_length - lock_length, 1.0e-9)
        weight = 0.5 * (1.0 + np.cos(np.pi * min(max(blend_t, 0.0), 1.0)))
        shift = target - route_pts[idx]
        shift_norm = float(np.linalg.norm(shift))
        if shift_norm > float(ROUTE_ENTRY_AXIS_MAX_SHIFT_MM):
            shift *= float(ROUTE_ENTRY_AXIS_MAX_SHIFT_MM) / shift_norm
        adjusted[idx] = route_pts[idx] + weight * shift

    adjusted[0] = origin
    after_stats = _entry_axis_stats(
        adjusted,
        origin,
        direction,
        sample_mm=float(ROUTE_ENTRY_AXIS_STATS_SAMPLE_MM),
    )
    point_shift = np.linalg.norm(adjusted - route_pts, axis=1)
    return adjusted, {
        'applied': True,
        'axis_origin_mm': origin.tolist(),
        'axis_direction': direction.tolist(),
        'lock_length_mm': float(lock_length),
        'blend_out_length_mm': float(blend_out_length),
        'max_shift_limit_mm': float(ROUTE_ENTRY_AXIS_MAX_SHIFT_MM),
        'start_shift_mm': float(np.linalg.norm(adjusted[0] - route_pts[0])),
        'max_point_shift_mm': float(np.max(point_shift)) if point_shift.size else 0.0,
        'mean_point_shift_mm': float(np.mean(point_shift)) if point_shift.size else 0.0,
        'axis_radial_before': before_stats,
        'axis_radial_after': after_stats,
    }


def _drop_short_segments(points: np.ndarray, min_segment_mm: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] <= 2:
        return pts.copy()
    keep = np.ones(pts.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) >= float(min_segment_mm)
    filtered = pts[keep]
    if filtered.shape[0] < 2:
        return np.asarray([pts[0], pts[-1]], dtype=float)
    filtered[0] = pts[0]
    filtered[-1] = pts[-1]
    return filtered


def _build_smoothed_route(
    route_aligned: np.ndarray,
    aligned_points: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    base = _resample_polyline(route_aligned, ROUTE_RESAMPLE_STEP_MM)
    smoothed = _smooth_polyline(base, ROUTE_SMOOTH_PASSES_PRE)
    centered = _recenter_route_from_cloud(
        smoothed,
        aligned_points,
        radius_mm=ROUTE_CENTERING_RADIUS_MM,
        blend=ROUTE_CENTERING_BLEND,
        min_neighbors=ROUTE_CENTERING_MIN_NEIGHBORS,
        max_shift_mm=ROUTE_CENTERING_MAX_SHIFT_MM,
    )
    centered, surface_centering_stats = _recenter_route_from_surface(
        centered,
        centered.copy(),
        surface_vertices,
        surface_faces,
    )
    centered = _smooth_polyline(centered, ROUTE_SMOOTH_PASSES_POST)
    centered[0] = route_aligned[0]
    centered[-1] = route_aligned[-1]
    centered = _resample_polyline(centered, ROUTE_RESAMPLE_STEP_MM)
    centered[0] = route_aligned[0]
    centered[-1] = route_aligned[-1]
    centered = _drop_short_segments(centered, ROUTE_MIN_SEGMENT_MM)

    if float(centered[0, 1]) > float(centered[-1, 1]):
        centered = centered[::-1].copy()

    stats = {
        'resample_step_mm': float(ROUTE_RESAMPLE_STEP_MM),
        **_route_summary(centered),
        'centering_radius_mm': float(ROUTE_CENTERING_RADIUS_MM),
        'centering_blend': float(ROUTE_CENTERING_BLEND),
        'centering_max_shift_mm': float(ROUTE_CENTERING_MAX_SHIFT_MM),
        **surface_centering_stats,
    }
    return centered, stats


def main() -> None:
    source_vessel_mesh = _resolve_existing_path(
        [EXTERNAL_SOURCE_VESSEL_MESH, PROJECT_VESSEL_MESH],
        'vessel mesh',
    )
    source_centerline_points = _resolve_existing_path(
        [EXTERNAL_SOURCE_CENTERLINE_POINTS, PROJECT_CENTERLINE_POINTS],
        'centerline point cloud',
    )

    CENTERLINE_DIR.mkdir(parents=True, exist_ok=True)
    ROUTE_DIR.mkdir(parents=True, exist_ok=True)

    copied_vessel = _copy_if_needed(source_vessel_mesh, PROJECT_VESSEL_MESH)
    copied_centerline = _copy_if_needed(source_centerline_points, PROJECT_CENTERLINE_POINTS)

    raw_points = np.asarray(np.load(PROJECT_CENTERLINE_POINTS), dtype=float)
    raw_points = np.atleast_2d(raw_points)
    if raw_points.shape[1] > 3:
        raw_points = raw_points[:, :3]
    if raw_points.shape[0] < 2:
        raise RuntimeError('Centerline point cloud must contain at least 2 points.')

    vessel_vertices, vessel_faces = _load_stl_vertices_faces(PROJECT_VESSEL_MESH)
    try:
        route_raw, route_metrics = _select_route_between_endpoints(
            raw_points,
            PREFERRED_ROUTE_START_INDEX,
            PREFERRED_ROUTE_END_INDEX,
        )
    except Exception:
        route_raw, route_metrics = _select_main_route(raw_points)
    scale, translation = _build_bbox_alignment(raw_points, vessel_vertices)
    aligned_points = raw_points * scale + translation
    route_aligned = route_raw * scale + translation

    if float(route_aligned[0, 1]) > float(route_aligned[-1, 1]):
        route_raw = route_raw[::-1].copy()
        route_aligned = route_aligned[::-1].copy()

    route_surface_vertices, route_surface_faces, route_surface_stats = _build_runtime_mesh(
        vessel_vertices,
        vessel_faces,
        quantization_mm=ROUTE_SURFACE_CENTERING_MESH_QUANTIZATION_MM,
    )
    runtime_vertices_raw, runtime_faces_raw, runtime_stats = _build_runtime_mesh(
        vessel_vertices,
        vessel_faces,
        quantization_mm=RUNTIME_MESH_QUANTIZATION_MM,
    )
    original_route_stats = _route_summary(route_aligned)
    smoothed_route, pre_entry_centering_route_stats = _build_smoothed_route(
        route_aligned,
        aligned_points,
        route_surface_vertices,
        route_surface_faces,
    )
    provisional_runtime_vertices, provisional_runtime_faces, provisional_inlet_open_stats = _open_runtime_mesh_inlet(
        runtime_vertices_raw,
        runtime_faces_raw,
        smoothed_route,
    )
    provisional_inlet_boundary = _detect_runtime_inlet_opening(
        provisional_runtime_vertices,
        provisional_runtime_faces,
        smoothed_route,
    )
    route_entry_axis_centering_stats: dict[str, float | bool | list[float] | str] = {
        'applied': False,
        'reason': 'runtime_inlet_boundary_not_found',
    }
    if bool(provisional_inlet_boundary.get('found', False)):
        smoothed_route, route_entry_axis_centering_stats = _recenter_route_entry_to_axis(
            smoothed_route,
            np.asarray(provisional_inlet_boundary['center_mm'], dtype=float),
            np.asarray(provisional_inlet_boundary['entry_axis_direction'], dtype=float),
        )
    smoothed_route = _drop_short_segments(smoothed_route, ROUTE_MIN_SEGMENT_MM)
    final_route_stats = {
        **pre_entry_centering_route_stats,
        **_route_summary(smoothed_route),
        'entry_axis_direction_sample_mm': float(ROUTE_ENTRY_AXIS_DIRECTION_SAMPLE_MM),
        'entry_axis_lock_length_mm': float(ROUTE_ENTRY_AXIS_LOCK_LENGTH_MM),
        'entry_axis_blend_out_length_mm': float(ROUTE_ENTRY_AXIS_BLEND_OUT_LENGTH_MM),
        'entry_axis_max_shift_mm': float(ROUTE_ENTRY_AXIS_MAX_SHIFT_MM),
    }

    runtime_vertices, runtime_faces, inlet_open_stats = _open_runtime_mesh_inlet(
        runtime_vertices_raw,
        runtime_faces_raw,
        smoothed_route,
    )
    inlet_boundary_stats = _detect_runtime_inlet_opening(
        runtime_vertices,
        runtime_faces,
        smoothed_route,
    )
    _write_obj(PROJECT_RUNTIME_VESSEL_MESH, runtime_vertices, runtime_faces)

    np.save(ALIGNED_POINTS_FILE, aligned_points)
    np.save(ROUTE_FILE, smoothed_route)

    alignment_payload = {
        'source_vessel_mesh': str(source_vessel_mesh),
        'project_vessel_mesh': str(PROJECT_VESSEL_MESH),
        'project_runtime_vessel_mesh': str(PROJECT_RUNTIME_VESSEL_MESH),
        'source_centerline_points': str(source_centerline_points),
        'project_centerline_points': str(PROJECT_CENTERLINE_POINTS),
        'copied_vessel_into_project': bool(copied_vessel),
        'copied_centerline_into_project': bool(copied_centerline),
        'route_name': ROUTE_NAME,
        'raw_point_count': int(raw_points.shape[0]),
        'aligned_point_count': int(aligned_points.shape[0]),
        'raw_bbox_min': np.min(raw_points, axis=0).tolist(),
        'raw_bbox_max': np.max(raw_points, axis=0).tolist(),
        'mesh_bbox_min': np.min(vessel_vertices, axis=0).tolist(),
        'mesh_bbox_max': np.max(vessel_vertices, axis=0).tolist(),
        'scale_xyz': scale.tolist(),
        'translation_xyz': translation.tolist(),
        'selected_route_metrics': route_metrics,
        'original_route_stats': original_route_stats,
        'smoothed_route_stats': final_route_stats,
        'pre_entry_centering_route_stats': pre_entry_centering_route_stats,
        'aligned_route_start_mm': smoothed_route[0].tolist(),
        'aligned_route_end_mm': smoothed_route[-1].tolist(),
        'aligned_route_vertical_span_mm': float(smoothed_route[-1, 1] - smoothed_route[0, 1]),
        'runtime_mesh_stats': runtime_stats,
        'route_surface_mesh_stats': route_surface_stats,
        'route_entry_axis_centering': route_entry_axis_centering_stats,
        'provisional_runtime_inlet_opening': provisional_inlet_open_stats,
        'provisional_runtime_inlet_boundary': provisional_inlet_boundary,
        'runtime_inlet_opening': inlet_open_stats,
        'runtime_inlet_boundary': inlet_boundary_stats,
    }
    ALIGNMENT_INFO_FILE.write_text(json.dumps(alignment_payload, indent=2, ensure_ascii=False), encoding='utf-8')

    runtime_info_payload = {
        'source_mesh': str(PROJECT_VESSEL_MESH),
        'runtime_mesh': str(PROJECT_RUNTIME_VESSEL_MESH),
        'stats': runtime_stats,
        'inlet_opening': inlet_open_stats,
        'inlet_boundary': inlet_boundary_stats,
        'bbox_min_mm': np.min(runtime_vertices, axis=0).tolist(),
        'bbox_max_mm': np.max(runtime_vertices, axis=0).tolist(),
    }
    PROJECT_RUNTIME_MESH_INFO.write_text(json.dumps(runtime_info_payload, indent=2, ensure_ascii=False), encoding='utf-8')

    selected_route_payload = {
        'selected_route_name': ROUTE_NAME,
        'selected_route_file': ROUTE_FILE.name,
        'reason': 'auto-extracted, smoothed, and locally recentered full lower-inlet to upper-outlet route from vessel_centerline_4_0108.npy',
        'summary': '完整主通路（下入口 -> 上出口，已平滑并重新居中）',
    }
    SELECTED_ROUTE_FILE.write_text(json.dumps(selected_route_payload, indent=2, ensure_ascii=False), encoding='utf-8')

    catalog_payload = {
        'selected_route_name': ROUTE_NAME,
        'routes': {
            ROUTE_NAME: {
                'file': f'routes/{ROUTE_FILE.name}',
                'summary': '完整主通路（下入口 -> 上出口，已平滑并重新居中）',
                'source': 'auto-extracted from vessel_centerline_4_0108.npy and aligned to vessel_final_4_0108.stl',
            }
        },
    }
    ROUTE_CATALOG_FILE.write_text(json.dumps(catalog_payload, indent=2, ensure_ascii=False), encoding='utf-8')

    print('[prepare_vessel_4_0108] Completed.')
    print(f'  project vessel mesh   : {PROJECT_VESSEL_MESH}')
    print(f'  runtime vessel mesh   : {PROJECT_RUNTIME_VESSEL_MESH}')
    print(f'  project centerline    : {PROJECT_CENTERLINE_POINTS}')
    print(f'  aligned points        : {ALIGNED_POINTS_FILE}')
    print(f'  extracted route       : {ROUTE_FILE}')
    print(f'  route start mm        : {np.round(smoothed_route[0], 3).tolist()}')
    print(f'  route end mm          : {np.round(smoothed_route[-1], 3).tolist()}')
    print(f'  scale xyz             : {np.round(scale, 6).tolist()}')
    print(f'  translation xyz       : {np.round(translation, 6).tolist()}')
    print(
        '  route stats           : '
        f'originalPts={original_route_stats["point_count"]}, '
        f'originalLen={original_route_stats["length_mm"]:.3f} mm, '
        f'originalMeanTurn={original_route_stats["mean_turn_deg"]:.2f} deg, '
        f'smoothedPts={final_route_stats["point_count"]}, '
        f'smoothedLen={final_route_stats["length_mm"]:.3f} mm, '
        f'smoothedMeanTurn={final_route_stats["mean_turn_deg"]:.2f} deg'
    )
    print(
        '  entry centering       : '
        f'applied={route_entry_axis_centering_stats.get("applied", False)}, '
        f'startShift={float(route_entry_axis_centering_stats.get("start_shift_mm", 0.0)):.3f} mm, '
        f'meanRadialBefore={float(route_entry_axis_centering_stats.get("axis_radial_before", {}).get("mean_radial_mm", 0.0)):.3f} mm, '
        f'meanRadialAfter={float(route_entry_axis_centering_stats.get("axis_radial_after", {}).get("mean_radial_mm", 0.0)):.3f} mm'
    )
    print(
        '  inlet boundary        : '
        f'found={inlet_boundary_stats.get("found", False)}, '
        f'componentVertices={int(inlet_boundary_stats.get("component_vertex_count", 0))}, '
        f'axisRadialOffset={float(inlet_boundary_stats.get("axis_radial_offset_mm", 0.0)):.3f} mm'
    )
    print(
        '  runtime mesh          : '
        f'inputFaces={runtime_stats["input_face_count"]}, '
        f'outputFaces={runtime_stats["output_face_count"]}, '
        f'inputVerts={runtime_stats["input_vertex_count"]}, '
        f'outputVerts={runtime_stats["output_vertex_count"]}'
    )


if __name__ == '__main__':
    main()
