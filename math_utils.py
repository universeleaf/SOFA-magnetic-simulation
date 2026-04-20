# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np

from .config import DEFAULT_INSERTION_DIR


@contextmanager
def _writeable(data):
    if hasattr(data, 'writeable'):
        with data.writeable() as arr:
            yield arr
        return
    if hasattr(data, 'writeableArray'):
        with data.writeableArray() as arr:
            yield arr
        return
    arr = np.asarray(data.value, dtype=float)
    yield arr
    data.value = arr.tolist()


def _read(data) -> np.ndarray:
    if hasattr(data, 'array'):
        try:
            return np.asarray(data.array(), dtype=float)
        except Exception:
            pass
    return np.asarray(data.value, dtype=float)


def _normalize(v, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    return np.zeros(3, dtype=float) if n < eps else arr / n


def _cumlen(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return np.zeros(1, dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts[:, :3], axis=0), axis=1))])


def _interp(points: np.ndarray, cum: np.ndarray, s: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    s = float(np.clip(s, 0.0, float(cum[-1])))
    seg = max(0, min(int(np.searchsorted(cum, s, side='right') - 1), pts.shape[0] - 2))
    ds = float(cum[seg + 1] - cum[seg])
    if ds < 1e-12:
        return pts[seg, :3].copy()
    a = float(np.clip((s - cum[seg]) / ds, 0.0, 1.0))
    return (1.0 - a) * pts[seg, :3] + a * pts[seg + 1, :3]


def _tangent(points: np.ndarray, cum: np.ndarray, s: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return DEFAULT_INSERTION_DIR.copy()
    s = float(np.clip(s, 0.0, float(cum[-1])))
    seg = max(0, min(int(np.searchsorted(cum, s, side='right') - 1), pts.shape[0] - 2))
    return _normalize(pts[seg + 1, :3] - pts[seg, :3])


def _quat_from_z_to(direction) -> np.ndarray:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    d = _normalize(direction)
    dot = float(np.clip(np.dot(z_axis, d), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    if dot < -1.0 + 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = _normalize(np.cross(z_axis, d))
    angle = math.acos(dot)
    s = math.sin(0.5 * angle)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle)], dtype=float)


def _parallel_transport(direction, tangent_from, tangent_to) -> np.ndarray:
    d1 = _normalize(direction)
    t1 = _normalize(tangent_from)
    t2 = _normalize(tangent_to)
    b = np.cross(t1, t2)
    if np.linalg.norm(b) < 1.0e-12:
        return d1
    b = _normalize(b)
    tmp = b - np.dot(b, t1) * t1
    b = _normalize(tmp) if np.linalg.norm(tmp) >= 1.0e-12 else b
    tmp = b - np.dot(b, t2) * t2
    b = _normalize(tmp) if np.linalg.norm(tmp) >= 1.0e-12 else b
    n1 = np.cross(t1, b)
    n2 = np.cross(t2, b)
    d2 = np.dot(d1, t1) * t2 + np.dot(d1, n1) * n2 + np.dot(d1, b) * b
    d2 = d2 - np.dot(d2, t2) * t2
    return _normalize(d2)


def _quat_from_basis(x_axis, y_axis, z_axis) -> np.ndarray:
    x = _normalize(x_axis)
    z = _normalize(z_axis)
    y = _normalize(y_axis)
    rot = np.column_stack((x, y, z))
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 1.0e-12)) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 1.0e-12)) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 1.0e-12)) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=float)
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 1.0e-12 else np.array([0.0, 0.0, 0.0, 1.0], dtype=float)


def _quat_rotate(q, v) -> np.ndarray:
    quat = np.asarray(q, dtype=float).reshape(4)
    vec = np.asarray(v, dtype=float).reshape(3)
    uv = np.cross(quat[:3], vec)
    uuv = np.cross(quat[:3], uv)
    return vec + 2.0 * (quat[3] * uv + uuv)


def _marker_points(center, size_mm: float) -> np.ndarray:
    c = np.asarray(center, dtype=float).reshape(3)
    s = float(size_mm)
    return np.asarray([
        c + np.array([s, 0.0, 0.0], dtype=float),
        c + np.array([-s, 0.0, 0.0], dtype=float),
        c + np.array([0.0, s, 0.0], dtype=float),
        c + np.array([0.0, -s, 0.0], dtype=float),
        c + np.array([0.0, 0.0, s], dtype=float),
        c + np.array([0.0, 0.0, -s], dtype=float),
    ], dtype=float)
