# Deep RL Arena - Algorithm & Parameter Reference

> **Quick Reference**: This document provides a unified location for all game mechanics, reward structures, tunable parameters, and their locations in the codebase.

---

## Table of Contents
1. [Game Overview](#game-overview)
2. [State Space (Observation)](#state-space-observation)
3. [Action Spaces](#action-spaces)
4. [Reward Structure](#reward-structure)
5. [Game Mechanics](#game-mechanics)
6. [Entity Parameters](#entity-parameters)
7. [Training Configuration](#training-configuration)
8. [Parameter Tuning Guide](#parameter-tuning-guide)

---

## Game Overview

**Objective**: Destroy all enemy spawners in each phase while surviving enemy attacks. Progress through 5 phases of increasing difficulty.

**Environment Type**: Multi-phase survival arena shooter  
**Learning Task**: Sequential decision-making with sparse rewards and exploration challenges

---

## State Space (Observation)

**Size**: 14-dimensional continuous vector  
**File Location**: [game/arena.py](game/arena.py) - `get_observation()` method  
**Normalization**: All values normalized to [-1, 1] range

### Observation Components

| Index | Component | Range | Description | File |
|-------|-----------|-------|-------------|------|
| 0 | Player X | [0, 1] | Normalized x-coordinate | [game/arena.py](game/arena.py) |
| 1 | Player Y | [0, 1] | Normalized y-coordinate | [game/arena.py](game/arena.py) |
| 2 | Velocity X | [-1, 1] | Horizontal velocity | [game/arena.py](game/arena.py) |
| 3 | Velocity Y | [-1, 1] | Vertical velocity | [game/arena.py](game/arena.py) |
| 4 | Angle | [0, 1] | Player rotation angle | [game/arena.py](game/arena.py) |
| 5 | Health | [0, 1] | Player health percentage | [game/arena.py](game/arena.py) |
| 6 | Enemy Distance | [0, 1] | Distance to nearest enemy | [game/arena.py](game/arena.py) |
| 7 | Enemy Angle | [-1, 1] | Relative angle to nearest enemy | [game/arena.py](game/arena.py) |
| 8 | Spawner Distance | [0, 1] | Distance to nearest spawner | [game/arena.py](game/arena.py) |
| 9 | Spawner Angle | [-1, 1] | Relative angle to nearest spawner | [game/arena.py](game/arena.py) |
| 10 | Phase | [0, 1] | Current game phase | [game/arena.py](game/arena.py) |
| 11 | Enemy Count | [0, 1] | Normalized number of enemies | [game/arena.py](game/arena.py) |
| 12 | Spawner Count | [0, 1] | Normalized number of spawners | [game/arena.py](game/arena.py) |
| 13 | Can Shoot | {0, 1} | Whether player can shoot | [game/arena.py](game/arena.py) |

**Constant**: `OBSERVATION_SIZE = 14` in [game/constants.py](game/constants.py)

---

## Action Spaces

### Rotation Environment (Asteroids-style)
**File**: [envs/rotation_env.py](envs/rotation_env.py)  
**Action Space**: `Discrete(5)`

| Action ID | Name | Effect | Implementation |
|-----------|------|--------|----------------|
| 0 | Nothing | Drift with friction | No action applied |
| 1 | Thrust | Accelerate forward | `player.apply_thrust()` |
| 2 | Rotate Left | Rotate counter-clockwise | `player.rotate_left()` |
| 3 | Rotate Right | Rotate clockwise | `player.rotate_right()` |
| 4 | Shoot | Fire projectile | `arena.player_shoot()` |

### Directional Environment
**File**: [envs/directional_env.py](envs/directional_env.py)  
**Action Space**: `Discrete(6)`

| Action ID | Name | Effect | Implementation |
|-----------|------|--------|----------------|
| 0 | Nothing | Stop moving | `move_direction(0, 0)` |
| 1 | Up | Move upward | `move_direction(0, -1)` |
| 2 | Down | Move downward | `move_direction(0, 1)` |
| 3 | Left | Move left | `move_direction(-1, 0)` |
| 4 | Right | Move right | `move_direction(1, 0)` |
| 5 | Shoot | Fire projectile | `arena.player_shoot()` |

---

## Reward Structure

**File Location**: [game/constants.py](game/constants.py) - `REWARDS` dictionary  
**Implementation**: [game/arena.py](game/arena.py) - `update()` and `_check_collisions()` methods

### Primary Rewards (Main Objectives)

| Event | Reward | Trigger Condition | Location | Tuning Impact |
|-------|--------|-------------------|----------|---------------|
| **Destroy Enemy** | +50.0 | Projectile kills enemy | [game/arena.py](game/arena.py) Line ~221 | ⭐⭐⭐ Encourages combat |
| **Destroy Spawner** | +200.0 | Projectile destroys spawner | [game/arena.py](game/arena.py) Line ~247 | ⭐⭐⭐ Main objective |
| **Phase Complete** | +500.0 | All spawners destroyed | [game/arena.py](game/arena.py) Line ~176 | ⭐⭐⭐ Major milestone |
| **Hit Enemy** | +2.0 | Projectile hits (doesn't kill) | [game/arena.py](game/arena.py) Line ~213 | ⭐⭐ Encourages shooting accuracy |

### Penalties (Survival Signals)

| Event | Reward | Trigger Condition | Location | Tuning Impact |
|-------|--------|-------------------|----------|---------------|
| **Take Damage** | -3.0 | Enemy collision with player | [game/arena.py](game/arena.py) Line ~265 | ⭐⭐ Encourages dodging |
| **Death** | -50.0 | Player health reaches 0 | [game/arena.py](game/arena.py) Line ~169 | ⭐⭐⭐ Terminal penalty |

### Shaping Rewards (Optional Guidance)

| Event | Reward | Trigger Condition | Location | Tuning Impact |
|-------|--------|-------------------|----------|---------------|
| **Survival Bonus** | +0.01 | Every step alive | [game/arena.py](game/arena.py) Line ~189 | ⭐ Time preference |
| **Shot Fired** | +0.5 | Player shoots projectile | [game/arena.py](game/arena.py) Line ~143 | ⭐ Encourages action |
| **Approach Spawner** | +0.1 | Within 300 units of spawner | [game/arena.py](game/arena.py) Line ~160 | ⭐ Objective guidance |
| **Approach Enemy** | 0.0 | Currently disabled | [game/constants.py](game/constants.py) | - Not used |

**Total Possible Reward per Episode**: Depends on phases completed, enemies destroyed, and survival time.

---

## Game Mechanics

### Episode Flow
**File**: [game/arena.py](game/arena.py) - `update()` method

1. **Initialization**: Player spawns at center, 2 spawners at edges
2. **Phase Progression**: Destroy all spawners → advance phase → more spawners spawn
3. **Termination Conditions**:
   - Player dies (`terminated = True`)
   - Time limit reached (`truncated = True`, `MAX_STEPS = 5000`)
   - All phases completed (`truncated = True`)

### Phase System
**File**: [game/arena.py](game/arena.py) - `_advance_phase()` method

| Phase | Spawners | Enemy Health Mult | Enemy Speed Mult | Location |
|-------|----------|-------------------|------------------|----------|
| 1 | 2 | 1.0× | 1.0× | [game/constants.py](game/constants.py) |
| 2 | 3 | 1.2× | 1.1× | Computed dynamically |
| 3 | 4 | 1.44× | 1.21× | Computed dynamically |
| 4 | 5 | 1.73× | 1.33× | Computed dynamically |
| 5 | 6 | 2.07× | 1.46× | Computed dynamically |

**Phase Constants**:
- `INITIAL_SPAWNERS = 2` - Starting spawners
- `SPAWNERS_PER_PHASE = 1` - Additional spawners per phase
- `MAX_PHASE = 5` - Maximum phase number
- `PHASE_ENEMY_HEALTH_MULT = 1.2` - Health multiplier per phase
- `PHASE_ENEMY_SPEED_MULT = 1.1` - Speed multiplier per phase

### Collision Detection
**File**: [game/arena.py](game/arena.py) - `_check_collisions()` method

**Types**:
1. **Player Projectiles ↔ Enemies**: Distance < (projectile_radius + enemy_radius)
2. **Player Projectiles ↔ Spawners**: Distance < (projectile_radius + spawner_radius)
3. **Enemies ↔ Player**: Distance < (enemy_radius + player_radius)

**Effects**:
- Projectile destroyed on hit
- Damage applied to target
- Particles spawned for visual feedback
- Rewards/penalties triggered

---

## Entity Parameters

### Player
**File**: [game/constants.py](game/constants.py) & [game/entities.py](game/entities.py)

| Parameter | Value | Description | Tuning Impact |
|-----------|-------|-------------|---------------|
| `PLAYER_RADIUS` | 20 | Collision radius | ⭐⭐ Affects difficulty |
| `PLAYER_MAX_HEALTH` | 100 | Starting/max health | ⭐⭐⭐ Survival time |
| `PLAYER_SPEED` | 5.0 | Directional movement speed | ⭐⭐ Control responsiveness |
| `PLAYER_THRUST` | 0.3 | Rotation mode acceleration | ⭐⭐ Momentum control |
| `PLAYER_MAX_VELOCITY` | 6.0 | Maximum speed cap | ⭐⭐ Movement dynamics |
| `PLAYER_ROTATION_SPEED` | 5.0 | Degrees per frame | ⭐ Aiming agility |
| `PLAYER_FRICTION` | 0.98 | Velocity decay per frame | ⭐ Momentum feel |
| `PLAYER_SHOOT_COOLDOWN` | 15 frames | Time between shots | ⭐⭐⭐ Fire rate (250ms @ 60fps) |
| `PLAYER_INVINCIBILITY_FRAMES` | 30 frames | Post-hit immunity | ⭐ Survival buffer |

### Enemies
**File**: [game/constants.py](game/constants.py) & [game/entities.py](game/entities.py)

| Parameter | Value | Description | Tuning Impact |
|-----------|-------|-------------|---------------|
| `ENEMY_RADIUS` | 15 | Collision radius | ⭐⭐ Threat level |
| `ENEMY_HEALTH` | 30 | Base health (scales with phase) | ⭐⭐⭐ Time to kill |
| `ENEMY_SPEED` | 2.5 | Movement speed (scales with phase) | ⭐⭐⭐ Evasion difficulty |
| `ENEMY_DAMAGE` | 10 | Damage to player on contact | ⭐⭐ Punishment severity |

### Spawners
**File**: [game/constants.py](game/constants.py) & [game/entities.py](game/entities.py)

| Parameter | Value | Description | Tuning Impact |
|-----------|-------|-------------|---------------|
| `SPAWNER_RADIUS` | 35 | Collision radius | ⭐ Target size |
| `SPAWNER_HEALTH` | 100 | Total health | ⭐⭐⭐ Objective difficulty |
| `SPAWNER_SPAWN_INTERVAL` | 180 frames | Time between enemy spawns | ⭐⭐⭐ Pressure rate (3s @ 60fps) |
| `SPAWNER_MAX_ENEMIES` | 5 | Max concurrent enemies per spawner | ⭐⭐⭐ Enemy density |

### Projectiles
**File**: [game/constants.py](game/constants.py) & [game/entities.py](game/entities.py)

| Parameter | Value | Description | Tuning Impact |
|-----------|-------|-------------|---------------|
| `PROJECTILE_SPEED` | 10.0 | Travel speed | ⭐⭐ Hit probability |
| `PROJECTILE_RADIUS` | 5 | Collision radius | ⭐ Hit detection |
| `PROJECTILE_LIFETIME` | 120 frames | Max duration (2s @ 60fps) | ⭐ Range limit |

---

## Training Configuration

### Algorithm Setup
**File**: [train.py](train.py)

**Current Implementation**:
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=f"./logs/{args.env}_{args.algo}"
)
```

### Default Hyperparameters
**Command Line Arguments**:
- `--env`: Environment type (`rotation` or `directional`)
- `--algo`: Algorithm (`ppo` or `dqn`)
- `--timesteps`: Total training steps (default: 100,000)
- `--lr`: Learning rate (default: 3e-4)

**PPO Defaults** (from Stable-Baselines3):
- Learning rate: 3e-4
- Batch size: 64
- Minibatches: 64
- Epochs: 10
- Discount (γ): 0.99
- GAE λ: 0.95
- Clip range: 0.2

**Training Outputs**:
- Models saved to: `./models/{env}_{algo}_test.zip`
- TensorBoard logs: `./logs/{env}_{algo}/`

---

## Parameter Tuning Guide

### Quick Tuning Matrix

| Goal | Parameters to Adjust | Files | Typical Changes |
|------|---------------------|-------|-----------------|
| **Make learning easier** | ↑ Rewards, ↓ Penalties, ↑ `PLAYER_HEALTH` | [constants.py](game/constants.py) | 2-5× reward multipliers |
| **Improve exploration** | ↑ `shot_fired` reward, ↑ `approach_spawner` | [constants.py](game/constants.py), [arena.py](game/arena.py) | Enable shaping rewards |
| **Speed up training** | ↓ `MAX_STEPS`, ↑ timesteps, ↑ learning rate | [constants.py](game/constants.py), [train.py](train.py) | Shorter episodes |
| **Reduce difficulty** | ↓ Enemy speed/health, ↑ Shoot cooldown | [constants.py](game/constants.py) | 50-80% of current |
| **Encourage combat** | ↑ `hit_enemy`, ↑ `destroy_enemy` rewards | [constants.py](game/constants.py) | Dense feedback |
| **Better aiming** | ↑ `PROJECTILE_SPEED`, ↓ `PLAYER_SHOOT_COOLDOWN` | [constants.py](game/constants.py) | Easier shooting |

### Critical Parameters for First-Time Training

**If agent doesn't learn basic combat:**
1. ↑ `REWARDS['hit_enemy']` from 2.0 → 5.0 or 10.0
2. ↑ `REWARDS['destroy_enemy']` from 50.0 → 100.0
3. Enable shot_fired reward in [arena.py](game/arena.py) (Line 143)
4. ↓ `ENEMY_SPEED` from 2.5 → 1.5

**If agent dies too quickly:**
1. ↑ `PLAYER_MAX_HEALTH` from 100 → 200
2. ↓ `ENEMY_DAMAGE` from 10 → 5
3. ↓ `REWARDS['death']` from -50.0 → -20.0

**If agent doesn't approach spawners:**
1. ↑ `REWARDS['approach_spawner']` from 0.0 → 0.5 or 1.0 in [constants.py](game/constants.py)
2. Uncomment approach logic in [arena.py](game/arena.py) (Line ~150-160)

### Reward Shaping Strategy

**Current Reward Density**: Sparse (only on hits/kills/death)  
**Location**: [game/arena.py](game/arena.py) - `update()` method

**Dense Shaping Options** (add to `update()` method):
```python
# Distance-based spawner reward
if len(living_spawners) > 0:
    min_spawner_dist = min(...)
    reward += 1.0 / (1.0 + min_spawner_dist / 100.0)  # Closer = better

# Damage dealt reward
if enemy_hit and not enemy_killed:
    reward += damage_dealt * 0.1  # Proportional to damage
```

### Advanced Tuning: Phase Difficulty

**File**: [game/constants.py](game/constants.py)

Adjust phase scaling curves:
- `PHASE_ENEMY_HEALTH_MULT = 1.2` → Lower for easier progression
- `PHASE_ENEMY_SPEED_MULT = 1.1` → Keep at 1.0 for constant speed
- `SPAWNERS_PER_PHASE = 1` → Increase for faster difficulty ramp

---

## File Reference Map

### Quick Lookup Table

| What You Want to Change | File to Edit | Lines/Function |
|-------------------------|--------------|----------------|
| Reward values | [game/constants.py](game/constants.py) | `REWARDS` dict (~70-80) |
| Player stats | [game/constants.py](game/constants.py) | Lines 34-42 |
| Enemy stats | [game/constants.py](game/constants.py) | Lines 49-52 |
| Spawner stats | [game/constants.py](game/constants.py) | Lines 54-57 |
| Phase configuration | [game/constants.py](game/constants.py) | Lines 59-64 |
| Episode length | [game/constants.py](game/constants.py) | `MAX_STEPS` (~66) |
| Observation space | [game/arena.py](game/arena.py) | `get_observation()` (~305-370) |
| Action mappings | [envs/rotation_env.py](envs/rotation_env.py) or [envs/directional_env.py](envs/directional_env.py) | `_apply_action()` |
| Collision logic | [game/arena.py](game/arena.py) | `_check_collisions()` (~192-285) |
| Reward computation | [game/arena.py](game/arena.py) | `update()` (~119-191) |
| Training algorithm | [train.py](train.py) | Lines 54-82 |
| Hyperparameters | [train.py](train.py) | PPO initialization (~73-78) |

---

## Evaluation & Debugging

### Monitoring Training
**TensorBoard**:
```bash
tensorboard --logdir=./logs
```

**Key Metrics to Watch**:
- `rollout/ep_rew_mean`: Average episode reward
- `rollout/ep_len_mean`: Average episode length
- `train/value_loss`: Value function convergence
- `train/policy_gradient_loss`: Policy improvement

### Test Trained Model
**File**: [evaluate.py](evaluate.py)
```bash
python3 evaluate.py --model models/rotation_ppo_test.zip --env rotation --episodes 5
```

### Common Training Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Reward stays at ~0.01 | Agent only survives, doesn't fight | ↑ Combat rewards, ↓ survival bonus |
| Agent dies immediately | Too hard, no exploration | ↑ Health, ↓ enemy difficulty |
| Learns to dodge but not shoot | Shooting penalties too high | Check cooldown, ↑ hit rewards |
| Stuck in local optimum | Insufficient exploration | ↑ Entropy coefficient, ↑ timesteps |
| Value loss explodes | Learning rate too high | ↓ Learning rate to 1e-4 |

---

## Version History

- **v1.0** (2026-01-07): Initial algorithm documentation
  - Unified parameter reference
  - Added `hit_enemy` reward constant
  - Comprehensive tuning guide

---

## Notes

- All coordinates use pygame convention: (0,0) at top-left
- Angles in degrees: 0° = right, 90° = down, -90° = up
- Physics updates at 60 FPS (controlled by pygame clock)
- RL agent steps are synced with physics steps (1 action per frame)
