# ElasticRodGuidewire

This plugin is the native SOFA backend used by `guidewire_scene` for the `elasticrod` path.
It is compiled as `ElasticRodGuidewire_hotfix.dll` and loaded directly from `guidewire_scene/build/ElasticRodGuidewire`.

## Current components

- `ElasticRodCompatCore`
  - rebuilds the legacy `elasticRod.cpp/.h` reference quantities inside the plugin
  - computes `refLen`, `voronoiLen`, `EA`, `EI`, `GJ`, `kappaBar`, `undeformedTwist`, lumped mass, and lumped rotational inertia
- `ElasticRodGuidewireModel`
  - native rod force field on top of `MechanicalObject<Rigid3d>`
  - includes stretch, bend, twist, and proximal insertion / twist boundary terms
  - exports debug data such as `debugStretch`, `debugKappa`, and `debugTwist`
- `ElasticRodLumpedMass`
  - native lumped translational mass and rotational inertia for the rod state
- `ExternalMagneticForceField`
  - native magnetic coupling aligned with the legacy `externalMagneticForce.cpp/.h` semantics
  - keeps look-ahead display data while applying the actual field direction from the nearest centerline segment tangent

## Build

From the workspace root:

```bat
guidewire_scene\build_plugin.bat
```

Expected output:

- `guidewire_scene/build/ElasticRodGuidewire/ElasticRodGuidewire_hotfix.dll`

## Runtime notes

- `guidewire_scene/config.py` selects the backend through `GUIDEWIRE_BACKEND`.
- `elasticrod` currently uses two stabilization modes:
  - `safe`: enables the post-solve lumen projection fallback used to keep the scene stable with the current rigid vessel surface contact chain
  - `strict`: disables that fallback and leaves only native rod + SOFA contact; this path remains experimental
- The original `elasticRod.cpp/.h` in the workspace root are kept unchanged and serve only as numerical reference.

## Legacy placeholder file

`src/ElasticRodCore.cpp` is no longer the numeric truth source. The active implementation lives in:

- `src/ElasticRodCompatCore.cpp`
- `src/ElasticRodGuidewireModel.cpp`
- `src/ElasticRodLumpedMass.cpp`
- `src/ExternalMagneticForceField.cpp`
