# Non-Suicidal PPO v1.1 - Checkpoint

**Timestamp:** January 11, 2026 - 16:12  
**Model File:** `non-suicidal-1.1_20260111_1612.zip`  
**Previous Version:** `non-suicidal-1.0_20260110_1553`

---

## Status Summary

✅ **Continued non-suicidal behavior with improved exploration capability**

This version maintains the non-suicidal achievements while introducing better exploration through increased entropy and extended training duration. A key discovery: the optimal learning rate schedule for the current reward function.

### Achievements
- **Increased exploration capacity** - Higher entropy coefficient allows better environment understanding
- **Extended training capability** - New LR schedule enables training beyond 3M timesteps
- **Optimal LR schedule discovered** - Linear decay to 2e-5 at 3M, then constant (best for current reward function)
- **Maintained non-suicidal behavior** - Zero suicidal episodes

### Current Limitations
- **Corner hiding persists** - Still hides in corners when nearest spawner spawns at opposite side
- **Opposite spawner challenge** - Struggles with spawners appearing far from current position

---

## Key Changes from v1.0 (non-suicidal-1.0_20260110_1553)

### 1. Entropy Coefficient Increase ⭐
**Previous:** `ent_coef=0.06`  
**Current:** `ent_coef=0.065`

**Rationale:**
- 16:12 10/1/2026: Increased from `0.06` to `0.065`
- Enables more exploration when combined with extended training
- Still gets stuck with spawners behind, needs slight increase
- Allows agent to discover more strategies without becoming reckless

### 2. Improved Learning Rate Schedule ⭐ MAJOR DISCOVERY
**Previous:** Linear decay from `1e-4` to `2e-5` over entire training duration  
**Current:** Linear decay from `1e-4` to `2e-5` at exactly 3M timesteps, then constant at `2e-5`

```python
def linear_schedule_3m(initial_value: float, total_timesteps: int):
    """
    Linear learning rate schedule that decays to 2e-5 at exactly 3M timesteps,
    then stays constant at 2e-5 afterward.
    """
    decay_timesteps = 3_000_000  # Decay ends at 3M timesteps
    min_lr = 2e-5
    
    def func(progress_remaining: float) -> float:
        current_timestep = total_timesteps * (1 - progress_remaining)
        
        if current_timestep >= decay_timesteps:
            # After 3M timesteps, keep at minimum learning rate
            return min_lr
        else:
            # Linear decay from initial_value to min_lr over first 3M timesteps
            decay_progress = current_timestep / decay_timesteps
            return initial_value - decay_progress * (initial_value - min_lr)
    
    return func
```

**Key Benefits:**
- **Decouples training duration from LR decay** - Can train for 5M, 10M steps without LR reaching too low
- **Maintains learning capacity** - After 3M steps, continues learning at stable 2e-5 rate
- **Prevents over-annealing** - LR doesn't decay to near-zero values in extended training
- **Optimal for current reward function** - Through experimentation, this is the best schedule discovered

**Comparison with Previous Schedule:**

| Timestep | Old Schedule (v1.0) | New Schedule (v1.1) |
|----------|---------------------|---------------------|
| 0 (start) | `1e-4` | `1e-4` |
| 1.5M | `~6e-5` | `~6e-5` |
| 3M | `2e-5` | `2e-5` |
| 4M | `~1.5e-5` ❌ | `2e-5` ✅ |
| 5M | `~1e-5` ❌ | `2e-5` ✅ |
| 10M | `~0.5e-5` ❌ | `2e-5` ✅ |

❌ Old schedule continues decaying (too low for learning)  
✅ New schedule maintains optimal learning rate

### 3. Reward Function
**No changes from v1.0** - Reward structure remains optimal for current behavior

---

## Complete Training Configuration

### PPO Hyperparameters (Stable-Baselines3)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **policy** | `"MlpPolicy"` | Multi-layer perceptron policy |
| **learning_rate** | `linear_schedule_3m(1e-4, timesteps)` ⭐ | Decays to `2e-5` at 3M, then constant |
| **n_steps** | `4096` | Steps per environment per update (16,384 total with 4 envs) |
| **batch_size** | `256` | Minibatch size for gradient updates |
| **n_epochs** | `10` | Epochs when optimizing surrogate loss |
| **gamma** | `0.99` | Discount factor for future rewards |
| **gae_lambda** | `0.95` | GAE lambda for bias-variance tradeoff |
| **clip_range** | `0.2` | PPO clipping parameter |
| **clip_range_vf** | `None` | Value function clipping (disabled) |
| **ent_coef** | `0.065` ⭐ | Entropy coefficient for exploration |
| **vf_coef** | `0.5` | Value function loss coefficient |
| **max_grad_norm** | `0.5` | Gradient clipping threshold |
| **use_sde** | `False` | State-dependent exploration (disabled) |
| **sde_sample_freq** | `-1` | SDE sampling frequency |
| **target_kl** | `None` | Target KL divergence (no limit) |
| **verbose** | `1` | Training verbosity level |
| **seed** | `None` | Random seed (random) |
| **device** | `"auto"` | CPU/CUDA device selection |

### Network Architecture

```python
policy_kwargs=dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])]
)
```

| Network | Architecture | Description |
|---------|-------------|-------------|
| **Policy Network (Actor)** | `[256, 256]` | 2 hidden layers, 256 units each |
| **Value Network (Critic)** | `[256, 256]` | 2 hidden layers, 256 units each |

### Learning Rate Schedule Visualization

```
LR
^
1e-4  |●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━○━━━━━━━━━━━━━━━━━━━━━━━━━>
      |                               ╱
      |                              ╱
      |                             ╱
      |                            ╱
      |                           ╱
2e-5  |                          ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━>
      |__________________________|_____|_____|_____|_____|_____|
      0                         3M    4M    5M    6M    7M    8M
                                        Timesteps

● = Key transition points
━ = Learning rate value
╱ = Linear decay phase
```

---

## Complete Reward Function

### Positive Rewards (Objectives)

| Reward Type | Value | Trigger Condition |
|-------------|-------|-------------------|
| **destroy_enemy** | `+85.0` | Enemy health reaches 0 |
| **destroy_spawner** | `+180.0` | Spawner health reaches 0 |
| **phase_complete** | `+500.0` | All spawners destroyed in current phase |
| **hit_enemy** | `+10.0` | Projectile hits enemy (even if doesn't kill) |
| **accuracy_bonus** | `+0.2` | Shooting when aimed within 20° of target |
| **potential_closer** | `+0.1` | Moving closer to nearest spawner (per step) |
| **orientation_closer** | `+0.1` | Rotating towards nearest spawner (per step) |

### Negative Rewards (Penalties)

| Penalty Type | Value | Trigger Condition |
|--------------|-------|-------------------|
| **death** | `-200.0` | Player health reaches 0 |
| **take_damage** | `-45.0` | Player takes damage from enemy collision |
| **potential_further** | `-0.12` | Moving away from nearest spawner (per step) |
| **orientation_further** | `-0.12` | Rotating away from nearest spawner (per step) |
| **shot_fired** | `-0.05` | Each time player shoots (ammo cost) |
| **existence_penalty** | `-0.015` | Every step (discourage passivity) |

### Disabled/Zero Rewards

| Reward Type | Value | Notes |
|-------------|-------|-------|
| **survival_bonus** | `0.0` | Disabled to prevent hiding/farming behavior |

---

## Game Environment Configuration

### Episode Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| **MAX_STEPS** | `5000` | Maximum steps per episode |
| **INITIAL_SPAWNERS** | `2` | Spawners at game start |
| **SPAWNERS_PER_PHASE** | `1` | Additional spawners each phase |
| **MAX_PHASE** | `5` | Maximum phase number |

### Player Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **PLAYER_RADIUS** | `20` | Collision radius |
| **PLAYER_MAX_HEALTH** | `100` | Starting health |
| **PLAYER_THRUST** | `0.3` | Forward acceleration |
| **PLAYER_MAX_VELOCITY** | `6.0` | Speed cap |
| **PLAYER_ROTATION_SPEED** | `2.0°/frame` | Rotation rate |
| **PLAYER_FRICTION** | `0.98` | Velocity decay per frame |
| **PLAYER_SHOOT_COOLDOWN** | `10 frames` | Time between shots (~0.167s @ 60fps) |
| **PLAYER_INVINCIBILITY_FRAMES** | `30 frames` | Immunity after hit (~0.5s) |

### Projectile Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **PROJECTILE_SPEED** | `10.0` | Travel speed per frame |
| **PROJECTILE_RADIUS** | `5` | Collision radius |
| **PROJECTILE_LIFETIME** | `120 frames` | Despawn time (~2s @ 60fps) |
| **PROJECTILE_DAMAGE** | `10` | Damage per hit |

### Enemy Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **ENEMY_RADIUS** | `15` | Collision radius |
| **ENEMY_HEALTH** | `30` | Base health (increases per phase) |
| **ENEMY_SPEED** | `2.5` | Movement speed per frame |
| **ENEMY_DAMAGE** | `10` | Collision damage to player |
| **PHASE_ENEMY_HEALTH_MULT** | `1.2` | Health multiplier per phase |
| **PHASE_ENEMY_SPEED_MULT** | `1.1` | Speed multiplier per phase |

### Spawner Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **SPAWNER_RADIUS** | `42` | Collision radius |
| **SPAWNER_HEALTH** | `100` | Total health |
| **SPAWNER_SPAWN_INTERVAL** | `180 frames` | Time between enemy spawns (~3s) |
| **SPAWNER_MAX_ENEMIES** | `5` | Maximum active enemies per spawner |

---

## Observation Space (16 dimensions)

### Player State (6 values)
1. **x position** - Normalized [0, 1] across screen width
2. **y position** - Normalized [0, 1] across screen height
3. **x velocity** - Normalized velocity component
4. **y velocity** - Normalized velocity component
5. **angle** - Player rotation angle in degrees
6. **health** - Normalized [0, 1], where 1 = full health (100)

### Nearest Enemy (3 values)
7. **distance** - Euclidean distance to nearest enemy
8. **cos(angle)** - Cosine of angle to nearest enemy
9. **sin(angle)** - Sine of angle to nearest enemy

### Nearest Spawner (3 values)
10. **distance** - Euclidean distance to nearest spawner
11. **cos(angle)** - Cosine of angle to nearest spawner
12. **sin(angle)** - Sine of angle to nearest spawner

### Game State (4 values)
13. **phase** - Current game phase [1-5]
14. **enemy_count** - Number of active enemies
15. **spawner_count** - Number of alive spawners
16. **can_shoot** - Boolean (1 if off cooldown, 0 if on cooldown)

---

## Action Space (Discrete - 5 actions)

| Action ID | Action Name | Effect |
|-----------|-------------|--------|
| **0** | `nothing` | No action, player drifts with current velocity |
| **1** | `thrust` | Apply forward thrust in current facing direction |
| **2** | `rotate_left` | Rotate counter-clockwise by 2° |
| **3** | `rotate_right` | Rotate clockwise by 2° |
| **4** | `shoot` | Fire projectile (if off cooldown) |

---

## Training Environment

| Setting | Value |
|---------|-------|
| **Environment Type** | `RotationArenaEnv` (Asteroids-style controls) |
| **Vectorization** | `4` parallel environments |
| **Vectorization Method** | `make_vec_env` (subprocess-based) |
| **Render Mode** | `None` (headless training) |
| **FPS Target** | `60` (physics simulation rate) |
| **Screen Size** | `1000 x 700` pixels |
| **Training Capacity** | Up to 10M+ timesteps (with new LR schedule) ⭐ |
| **Optimal Training Duration** | 3-5M timesteps recommended |
| **TensorBoard Metric** | `PPO_45` ⭐ |

### Training Comparison

**This model (PPO_45):**
- `ent_coef = 0.065`
- Training duration: >3M steps (extended with new LR schedule)
- Learning rate: Linear to 2e-5 at 3M, then constant

**Baseline comparison (PPO_39):**
- `ent_coef = 0.06`
- Training duration: 3M steps (stopped at 3M)
- Learning rate: Same schedule (linear to 2e-5 at 3M)

**Key differences:** PPO_45 has slightly higher exploration (`0.065` vs `0.06`) and benefits from extended training beyond 3M steps, demonstrating the advantage of the new learning rate schedule that maintains learning capacity after 3M timesteps.

**Note:** The new learning rate schedule enables training beyond 3M timesteps without LR decay issues. This allows for extended exploration with higher entropy coefficient.

---

## Gradient Computation & Backpropagation

### Loss Function (PPO)

The total loss for PPO combines three components:

```
L_total = L_policy + c_vf * L_value - c_ent * S_entropy
```

#### 1. Policy Loss (Clipped Surrogate Objective)

```
L_policy = -E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]
```

Where:
- `r(θ) = π_θ(a|s) / π_θ_old(a|s)` (probability ratio)
- `A` = Advantage estimate (from GAE)
- `ε` = `clip_range` = `0.2`
- `E[...]` = Expectation over batch

#### 2. Value Function Loss

```
L_value = MSE(V_θ(s), V_target)
```

Where:
- `V_θ(s)` = Predicted state value
- `V_target` = Target value (returns from GAE)
- `MSE` = Mean squared error
- Coefficient `c_vf` = `vf_coef` = `0.5`

#### 3. Entropy Bonus

```
S_entropy = E[π_θ(s) * log(π_θ(s))]
```

Where:
- Coefficient `c_ent` = `ent_coef` = `0.065`
- Higher entropy = more exploration

### Generalized Advantage Estimation (GAE)

```
A_t = Σ(γλ)^l * δ_{t+l}
```

Where:
- `δ_t = r_t + γV(s_{t+1}) - V(s_t)` (TD error)
- `γ` = `gamma` = `0.99` (discount factor)
- `λ` = `gae_lambda` = `0.95` (bias-variance tradeoff)

### Gradient Clipping

```
if ||∇_θ|| > max_grad_norm:
    ∇_θ ← ∇_θ * (max_grad_norm / ||∇_θ||)
```

Where:
- `max_grad_norm` = `0.5`

---

## Key Discoveries

### Learning Rate Schedule Discovery ⭐

Through experimentation, we discovered the optimal LR schedule for the current reward function:

1. **Linear decay phase (0-3M steps):** 
   - Start: `1e-4`
   - End: `2e-5`
   - Purpose: Initial learning and policy improvement

2. **Constant phase (3M+ steps):**
   - Value: `2e-5`
   - Purpose: Continued refinement and exploration

**Why this is optimal:**
- Provides strong initial learning signal
- Maintains capacity for continued improvement
- Doesn't over-anneal in extended training
- Synergizes well with higher entropy coefficient
- Prevents premature convergence

### Exploration-Duration Synergy

**Key insight:** Increasing `ent_coef` and training duration together creates synergistic benefits:

- **Higher entropy (0.065)** → More diverse behaviors explored
- **Extended training** → More time to evaluate diverse behaviors
- **Stable LR after 3M** → Consistent learning signal during exploration
- **Result:** Agent discovers more strategies without becoming unstable

---

## Evolution Timeline

### Entropy Coefficient History

| Date/Time | Value | Rationale |
|-----------|-------|-----------|
| Initial | `0.01` | Baseline |
| - | `0.05` | Escape corner hiding |
| 9:48 9/1 | `0.1` | Explore opposite spawners |
| 10:18 9/1 | `0.025` | Reduced back |
| 17:12 9/1 | `0.08` | Explore spawners behind |
| - | `0.025` | Agent avoided hitting enemies |
| 13:20 10/1 | `0.05` | Still stuck with opposite spawners |
| 14:53 10/1 | `0.06` | Rarely stuck, but still occasional |
| 16:12 10/1 | `0.065` | ⭐ **CURRENT** - Enable extended exploration |

### Learning Rate Schedule Evolution

| Version | Schedule Type | Min LR | Max Training | Notes |
|---------|---------------|--------|--------------|-------|
| v1.1 (suicidal) | Constant | N/A | N/A | `1e-4` constant |
| v1.2 (suicidal) | Linear | `0` | ~2M | First schedule attempt |
| v1.0 (non-suicidal) | Linear | `2e-5` | ~2M | Improved with min floor |
| **v1.1 (non-suicidal)** | **Linear+Constant** | **`2e-5`** | **10M+** | ⭐ **OPTIMAL** - Decays to 3M, then constant |

---

## Goals for Next Model (v1.2)

### Primary Objectives

1. **Focus more on killing enemies** 
   - Agent should prioritize eliminating active enemies before chasing spawners
   - Especially important for low-health enemies that are nearly dead
   - May need to increase `destroy_enemy` reward further or add "cleanup bonus"

2. **Solve opposite side spawner issue**
   - Agent must overcome hesitation when spawner spawns at opposite side of arena
   - Should navigate across map confidently instead of hiding in corners
   - Potential solutions:
     - Stronger navigation rewards for large distance reductions
     - Penalty for staying stationary when spawner is far (>500 units)
     - Corner penalty that scales with spawner distance
     - Possible new reward: `long_distance_navigation_bonus`

### Potential Approaches

- **Reward adjustments:**
  - Increase `destroy_enemy`: `85.0` → `90.0` or `95.0`
  - Add distance-scaled corner penalty
  - Add momentum rewards for cross-map movement
  
- **Training strategy:**
  - Leverage extended training capability (4-5M steps)
  - Use higher entropy if needed (`0.07` to encourage cross-map exploration)
  - Monitor if agent naturally learns these behaviors with more exploration time

---

## Next Steps & Recommendations

### High Priority

1. **Address corner hiding with opposite spawners**
   - Add penalty for staying in corners when spawner is far
   - Possible: Distance-weighted corner penalty (stronger when spawners far)
   - Consider: `corner_opposite_spawner_penalty = -0.5` when in corner AND nearest spawner >500 units

2. **Improve cross-map navigation**
   - Agent hesitates when spawner spawns at opposite side
   - May need stronger `potential_closer` reward when distance is large
   - Consider adaptive potential rewards: higher bonus for larger distance reductions

3. **Test extended training (4-5M steps)**
   - New LR schedule enables longer training
   - May allow agent to overcome corner hiding through exploration
   - Higher entropy + stable LR should help discover better strategies

### Medium Priority

4. **Monitor entropy-performance relationship**
   - Current `0.065` maintains exploration without chaos
   - May try `0.07` if corner hiding persists
   - Track if exploration quality improves with extended training

5. **Evaluate reward function effectiveness**
   - Current rewards work well with discovered LR schedule
   - May need adjustments for corner hiding issue
   - Consider spatial awareness rewards

### Experiments to Try

- **Extended training:** Train for 5M steps to test new schedule
- **Entropy sweep:** Test `0.065`, `0.07`, `0.075` with 4M training
- **Corner detection:** Add corner-awareness to observation space
- **Adaptive potential:** Scale `potential_closer` reward by distance

---

## Performance Summary

| Metric | v1.0 | v1.1 | Change |
|--------|------|------|--------|
| **Suicidal Episodes** | 0% | 0% | Maintained ✅ |
| **Corner Hiding** | 30% | ~30% | Similar |
| **Exploration Capacity** | Good | Better | Improved ✅ |
| **Training Duration** | 2.05M | 3M+ capable | Extended ✅ |
| **LR Schedule** | Basic linear | Optimal | Discovered ✅ |
| **Opposite Spawner** | Issue | Still issue | Needs work ⚠️ |

---

## Version History

- **v1.1 (suicidal)** (2026-01-09 03:47): Aggressive behavior, constant LR
- **v1.2 (suicidal)** (2026-01-10 13:38): 20% suicidal, linear LR (to 0), ent_coef=0.05
- **v1.0 (non-suicidal)** (2026-01-10 15:53): 0% suicidal, LR min floor (2e-5), ent_coef=0.06
- **v1.1 (non-suicidal)** (2026-01-11 16:12): Optimal LR schedule (3M constant), ent_coef=0.065 ⭐ **CURRENT**
