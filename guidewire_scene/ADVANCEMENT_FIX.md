# Guidewire Advancement Fix for Safe Mode

## Problem
After switching to safe mode, the guidewire was not advancing properly:
- Guidewire barely moved forward
- Excessive lateral movement instead of axial advancement
- Appeared to be stuck or sliding sideways

## Root Causes
1. **Thrust force too weak**: `ELASTICROD_THRUST_FORCE_N = 0.25N` was insufficient
2. **Axial assist too weak**: `ELASTICROD_AXIAL_PATH_ASSIST_FORCE_N = 0.15N` was insufficient
3. **Contact stiffness too soft**: `ELASTICROD_GUIDEWIRE_CONTACT_STIFFNESS = 120.0` made contact response weak
4. **Initial push force too low**: `PUSH_FORCE_INITIAL_TOTAL = 0.08` was too conservative
5. **Aggressive force reduction**: Forces dropped to 30-40% when encountering resistance
6. **Push speed scale**: Default 1.0x was too slow

## Solutions Applied

### 1. Increased Thrust and Axial Assist Forces
```python
ELASTICROD_THRUST_FORCE_N = 0.60  # Was 0.25N
ELASTICROD_AXIAL_PATH_ASSIST_FORCE_N = 0.35  # Was 0.15N
```
- 2.4x increase in thrust force limit
- 2.3x increase in axial path assist force
- These forces help the guidewire advance along the vessel centerline

### 2. Increased Contact Stiffness
```python
ELASTICROD_GUIDEWIRE_CONTACT_STIFFNESS = 280.0  # Was 120.0
```
- 2.3x increase in contact stiffness
- Still lower than strict mode (420.0) to maintain numerical stability
- Provides better contact response for effective push transmission

### 3. Increased Push Force Parameters
```python
PUSH_FORCE_INITIAL_TOTAL = 0.20  # Was 0.08
PUSH_FORCE_MIN_TOTAL = 0.005  # Was 0.002
PUSH_FORCE_MAX_TOTAL = 6.00  # Was 4.00
```
- 2.5x increase in initial push force
- 2.5x increase in minimum push force
- 1.5x increase in maximum push force
- Ensures strong initial advancement and maintains minimum push

### 4. Reduced Force Reduction on Resistance
```python
PUSH_FORCE_REDUCED_SCALE_ON_WALL = 0.60  # Was 0.40
PUSH_FORCE_REDUCED_SCALE_ON_STEERING = 0.50  # Was 0.30
PUSH_FORCE_REDUCED_SCALE_ON_STALL = 0.50  # Was 0.30
```
- When hitting walls: maintain 60% force instead of 40%
- When steering: maintain 50% force instead of 30%
- When stalled: maintain 50% force instead of 30%
- Helps overcome resistance instead of backing off too much

### 5. Increased Push Speed Scale
```python
ELASTICROD_GUI_WALLCLOCK_PUSH_SPEED_SCALE = 1.5  # Was 1.0
```
- 1.5x faster push speed in GUI mode
- Helps guidewire advance more quickly

## Expected Results
- Guidewire should now advance smoothly along the vessel
- Reduced lateral sliding/wandering
- Better response to push commands
- Faster overall advancement speed
- Still maintains physical realism (magnetic coupling and collision are 100% real)

## Physics Integrity
All changes maintain the physical realism requirements:
- Magnetic field coupling: 100% real (unchanged)
- Guidewire-vessel collision: 100% real (unchanged)
- Contact stiffness increased but still within physically realistic range
- Forces increased to overcome numerical damping, not to bypass physics

## Testing
Run the simulation and verify:
1. Guidewire advances smoothly when pushing
2. No excessive lateral movement
3. Responds properly to steering commands
4. Performance remains acceptable (target: 2-3 seconds sim time in 8 minutes wallclock)
