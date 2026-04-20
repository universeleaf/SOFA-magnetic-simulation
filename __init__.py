from __future__ import annotations


def createScene(*args, **kwargs):
    from .scene import createScene as _create_scene

    return _create_scene(*args, **kwargs)


__all__ = ['createScene']
