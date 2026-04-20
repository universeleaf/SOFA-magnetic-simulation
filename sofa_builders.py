# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import (
    CAMERA_FOLLOW_DIRECTION,
    CAMERA_FOLLOW_DISTANCE_MAX_MM,
    CAMERA_FOLLOW_DISTANCE_MIN_MM,
    CAMERA_FOLLOW_DISTANCE_SCALE,
    CAMERA_FOLLOW_YAW_DEG,
    CONTACT_MANAGER_RESPONSE,
    CONTACT_MANAGER_RESPONSE_PARAMS,
    GUIDEWIRE_CONTACT_STIFFNESS,
    WIRE_RADIUS_MM,
)


def _add_contact_manager(root, response_params: str | None = None, **extra_kwargs):
    params = CONTACT_MANAGER_RESPONSE_PARAMS if response_params is None else str(response_params)
    for name in ('DefaultContactManager', 'CollisionResponse'):
        try:
            return root.addObject(
                name,
                name='contactManager',
                response=CONTACT_MANAGER_RESPONSE,
                responseParams=params,
                **extra_kwargs,
            )
        except Exception:
            pass
    raise RuntimeError('无法创建接触管理器。')


def _add_full_collision(
    guidewire,
    points: np.ndarray,
    edges: Sequence[Sequence[int]],
    self_collision: bool = True,
    wire_radius_mm: float = WIRE_RADIUS_MM,
    contact_stiffness: float = GUIDEWIRE_CONTACT_STIFFNESS,
):
    """
    全导丝统一碰撞节点。

    这里必须覆盖整根导丝的全部节点和全部边，视觉节点与磁头节点都不能复用碰撞模型；
    否则就会再次出现“红色磁头受约束、灰色主体穿墙”的失配问题。
    """
    coll = guidewire.addChild('CollisionNode')
    coll.addObject('EdgeSetTopologyContainer', name='topo', edges=[list(map(int, e[:2])) for e in edges])
    coll.addObject(
        'MechanicalObject',
        name='dofs',
        template='Vec3d',
        position=np.asarray(points, dtype=float).tolist(),
        rest_position=np.asarray(points, dtype=float).tolist(),
        showObject=False,
    )
    coll.addObject(
        'RigidMapping',
        name='collisionMapping',
        template='Rigid3d,Vec3d',
        input='@../dofs',
        output='@dofs',
        rigidIndexPerPoint=list(range(int(np.asarray(points).shape[0]))),
        globalToLocalCoords=True,
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
        f'[INFO] Full guidewire collision enabled: nodes={int(np.asarray(points).shape[0])}, '
        f'edges={len(list(edges))}, mapping=RigidMapping(Rigid3d->Vec3d), '
        f'selfCollision={1 if self_collision else 0}, radius={float(wire_radius_mm):.3f} mm, '
        f'contactStiffness={float(contact_stiffness):.1f}'
    )
    return coll


def _camera_follow_offset(diag: float) -> np.ndarray:
    direction = np.asarray(CAMERA_FOLLOW_DIRECTION, dtype=float).reshape(3)
    yaw = np.deg2rad(float(CAMERA_FOLLOW_YAW_DEG))
    rot_z = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    # 这里只旋转相机视角，不改导丝、血管或任何物理算法。
    # 用户要“朝向我旋转 90 度”，等价于把当前跟随镜头绕世界 Z 轴转 90°。
    direction = rot_z @ direction
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction = np.array([0.0, -1.0, 0.35], dtype=float)
        norm = float(np.linalg.norm(direction))
    distance = float(np.clip(CAMERA_FOLLOW_DISTANCE_SCALE * max(float(diag), 1.0), CAMERA_FOLLOW_DISTANCE_MIN_MM, CAMERA_FOLLOW_DISTANCE_MAX_MM))
    return (distance / norm) * direction


def _add_camera(root, center: np.ndarray, diag: float):
    offset = _camera_follow_offset(diag)
    try:
        pos = np.asarray(center, dtype=float).reshape(3) + offset
        camera = root.addObject(
            'InteractiveCamera',
            name='camera',
            position=pos.tolist(),
            lookAt=np.asarray(center, dtype=float).reshape(3).tolist(),
            zNear=max(0.1, 0.01 * diag),
            zFar=max(1000.0, 5.0 * diag),
        )
        return camera, offset
    except Exception:
        return None, offset
