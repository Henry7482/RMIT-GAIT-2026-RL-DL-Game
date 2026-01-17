# Non-Suicidal PPO v1.0 - Checkpoint

**Timestamp:** January 10, 2026 - 15:53  
**Model File:** `non-suicidal-1.0_20260110_1553.zip`  
**Previous Version:** `suicidal-1.2_20260110_1338`

---

## Status Summary

✅ **Major breakthrough: No suicidal behavior across 10 episodes!**

The agent has successfully overcome suicidal tendencies and demonstrates improved tactical awareness. It now knows how to move forward before spinning to create distance from threats coming from different directions.

### Performance Statistics (10 Episodes)
- **Suicidal behavior:** 0% (eliminated!)
- **Corner hiding:** ~30% of the time (still an issue)
- **Tactical awareness:** Shows understanding of threat management

### Achievements
- **Zero suicidal behavior** - Tested over 10 episodes with no suicidal runs
- **Improved threat response** - Moves forward before spinning to create distance from multi-directional threats
- **Better decision-making** - Agent prioritizes survival while maintaining offensive pressure

### Current Limitations
- **Corner hiding** - Still hides in corners approximately 30% of the time
- **Over-prioritizes spawners** - Will sometimes chase spawners over nearly-dead enemies
- **Enemy dodging** - Could improve evasion tactics

---

## Key Changes from v1.2 (suicidal-1.2_20260110_1338)

### 1. Learning Rate Minimum Floor Adjustment
**Previous:** Linear decay from `1e-4` to `0`  
**Current:** Linear decay from `1e-4` to `2e-5`

```python
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        min_lr = 2e-5  # Increased from 1e-5
        return min_lr + progress_remaining * (initial_value - min_lr)
    return func
```

**Rationale:**
- 14:53 10/1/2026: Changed decay to bottom out at `1e-5` instead of `0`
- 15:41 10/1/2026: Increased minimum from `1e-5` to `2e-5`
- Observing PPO_31, mean reward started rising again (potentially escaping local minima)
- Prevents learning from completely stopping, allowing continued fine-tuning
- Maintains exploration capability throughout training

**Training Duration:**
- **Total timesteps:** `2,500,000` (2.5M instead of 2M)
- **Rationale:** Extra 50k steps (500k total) allowed model to potentially escape local minima
- **Evidence:** Compare PPO_31 vs PPO_32 logs - the additional 500k steps show improved performance

### 2. Entropy Coefficient Fine-Tuning
**Previous:** `ent_coef=0.05`  
**Current:** `ent_coef=0.06`

**Rationale:**
- 14:53 10/1/2026: Increased from `0.05` to `0.06`
- Agent was rarely getting stuck with `0.05`, but still occasionally
- Slight increase to encourage more exploration and prevent getting stuck
- Helps agent discover threat management strategies

### 3. Observation Space Architecture Change ⭐ CRITICAL
**Previous:** Nearest enemy/spawner angles as single normalized value (2 dimensions per target)
**Current:** Angles represented as sin/cos components (3 dimensions per target: distance, cos, sin)

**Observation space expansion: 14D → 16D**

```python
# Previous (v1.2 and earlier)
nearest_enemy_angle = (angle_to_enemy - player.angle) / 180.0  # Single value [-1, 1]
obs = [..., enemy_dist, enemy_angle, spawner_dist, spawner_angle, ...]

# Current (v1.0)
rel_angle_rad = math.radians(angle_to_target - player.angle)
enemy_cos = math.cos(rel_angle_rad)
enemy_sin = math.sin(rel_angle_rad)
obs = [..., enemy_dist, enemy_cos, enemy_sin, spawner_dist, spawner_cos, spawner_sin, ...]
```

**Why this matters for non-suicidal behavior:**

1. **Eliminates angle wrapping discontinuity**
   - Old representation: 179° → -179° caused a jump from +0.99 → -0.99
   - Neural network perceived this as "target teleporting" across observation space
   - Agent couldn't learn smooth rotation policies for targets behind it

2. **Enables continuous spatial reasoning**
   - Sin/cos provides smooth, continuous representation across full 360°
   - Network can learn: "rotate left when sin > 0, rotate right when sin < 0"
   - Cosine naturally encodes "is target in front (cos > 0) or behind (cos < 0)"

3. **Critical for spawner approach behavior**
   - Previous versions showed suicidal rushing because agent couldn't reliably navigate to spawners at arbitrary angles
   - With continuous angle representation, agent learned proper "approach → align → thrust → shoot" sequences
   - Reduced panicked movement toward spawners behind the player

**Impact on learning dynamics:**
- Policy gradient updates became more stable (no discontinuous jumps in observation space)
- Value function could properly generalize across rotation angles
- Agent developed smoother, more deliberate movement patterns instead of reactive thrashing

**Evidence of improvement:**
- Suicidal behavior: 20% (v1.2) → 0% (v1.0)
- This architectural change likely contributed as much as entropy/LR tuning to eliminating suicidal mode

### 4. Reward Function
**No changes from v1.2** - Maintained same reward structure as it was working well

---

## Complete Training Configuration

### PPO Hyperparameters (Stable-Baselines3)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **policy** | `"MlpPolicy"` | Multi-layer perceptron policy |
| **learning_rate** | `linear_schedule(1e-4)` | Decays from `1e-4` to `2e-5` |
| **n_steps** | `4096` | Steps per environment per update (16,384 total with 4 envs) |
| **batch_size** | `256` | Minibatch size for gradient updates |
| **n_epochs** | `10` | Epochs when optimizing surrogate loss |
| **gamma** | `0.99` | Discount factor for future rewards |
| **gae_lambda** | `0.95` | GAE lambda for bias-variance tradeoff |
| **clip_range** | `0.2` | PPO clipping parameter |
| **clip_range_vf** | `None` | Value function clipping (disabled) |
| **ent_coef** | `0.06` | Entropy coefficient for exploration |
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

**Total Parameters:** ~200K-300K (approximate, depends on observation/action space size)

### Learning Rate Schedule

```python
def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule with minimum floor.
    """
    def func(progress_remaining: float) -> float:
        min_lr = 2e-5
        return min_lr + progress_remaining * (initial_value - min_lr)
    return func
```

| Progress | Learning Rate |
|----------|---------------|
| Start (100%) | `1e-4` (0.0001) |
| Mid (50%) | `~6e-5` (0.00006) |
| End (0%) | `2e-5` (0.00002) |

### Optimization Details

| Component | Configuration |
|-----------|---------------|
| **Optimizer** | Adam (default in SB3) |
| **Adam Beta1** | `0.9` (default) |
| **Adam Beta2** | `0.999` (default) |
| **Adam Epsilon** | `1e-5` (default) |
| **Gradient Clipping** | Max norm = `0.5` |
| **Gradient Computation** | Automatic differentiation (PyTorch) |

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

**Total Possible Per Step:** Variable, but destroying spawner + phase complete = `680.0` maximum spike

### Negative Rewards (Penalties)

| Penalty Type | Value | Trigger Condition |
|--------------|-------|-------------------|
| **death** | `-200.0` | Player health reaches 0 |
| **take_damage** | `-45.0` | Player takes damage from enemy collision |
| **potential_further** | `-0.12` | Moving away from nearest spawner (per step) |
| **orientation_further** | `-0.12` | Rotating away from nearest spawner (per step) |
| **shot_fired** | `-0.05` | Each time player shoots (ammo cost) |
| **existence_penalty** | `-0.015` | Every step (discourage passivity) |

**Maximum Penalty Per Step:** `-0.415` (moving away, rotating away, shooting, existing)

### Disabled/Zero Rewards

| Reward Type | Value | Notes |
|-------------|-------|-------|
| **survival_bonus** | `0.0` | Disabled to prevent hiding/farming behavior |

### Reward Shaping Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **accuracy_threshold** | `20.0°` | Angle tolerance for accuracy bonus |
| **spawner_tracking** | Dynamic | Tracks distance/angle to nearest spawner each step |

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
| **Total Training Steps** | `2,050,000` (2.05M) ⭐ |
| **TensorBoard Logs** | `PPO_31` (2M) vs `PPO_32` (2.05M) |

**Note:** Extended training from 2M to 2.5M steps based on observing potential local minima escape in PPO_31. The extra 500k steps allowed the agent to break through a performance plateau and achieve non-suicidal behavior.

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

**Purpose:** Maximize expected advantage while preventing too-large policy updates

#### 2. Value Function Loss

```
L_value = MSE(V_θ(s), V_target)
```

Where:
- `V_θ(s)` = Predicted state value
- `V_target` = Target value (returns from GAE)
- `MSE` = Mean squared error
- Coefficient `c_vf` = `vf_coef` = `0.5`

**Purpose:** Train critic to accurately predict state values

#### 3. Entropy Bonus

```
S_entropy = E[π_θ(s) * log(π_θ(s))]
```

Where:
- Coefficient `c_ent` = `ent_coef` = `0.06`
- Higher entropy = more exploration

**Purpose:** Encourage exploration by preventing premature convergence

### Gradient Flow

```
1. Collect rollouts (n_steps=4096 per env, 16,384 total)
2. Compute advantages using GAE (λ=0.95, γ=0.99)
3. For each epoch (n_epochs=10):
   a. Shuffle rollout data
   b. Split into minibatches (batch_size=256)
   c. For each minibatch:
      - Forward pass through policy and value networks
      - Compute L_total
      - Backward pass: ∇_θ L_total
      - Clip gradients: ||∇_θ|| ≤ max_grad_norm (0.5)
      - Adam optimizer step with current LR
4. Update old policy: π_θ_old ← π_θ
```

### Generalized Advantage Estimation (GAE)

```
A_t = Σ(γλ)^l * δ_{t+l}
```

Where:
- `δ_t = r_t + γV(s_{t+1}) - V(s_t)` (TD error)
- `γ` = `gamma` = `0.99` (discount factor)
- `λ` = `gae_lambda` = `0.95` (bias-variance tradeoff)

**Purpose:** Estimate advantage with reduced variance while maintaining low bias

### Gradient Clipping

```
if ||∇_θ|| > max_grad_norm:
    ∇_θ ← ∇_θ * (max_grad_norm / ||∇_θ||)
```

Where:
- `max_grad_norm` = `0.5`
- `||∇_θ||` = L2 norm of gradient vector

**Purpose:** Prevent exploding gradients and ensure stable training

---

## Training Metrics (TensorBoard)

### Logged Metrics
- `train/learning_rate` - Current learning rate value
- `train/entropy_loss` - Entropy of policy distribution
- `train/policy_gradient_loss` - Policy loss component
- `train/value_loss` - Value function loss
- `train/approx_kl` - Approximate KL divergence
- `train/clip_fraction` - Fraction of samples clipped
- `train/explained_variance` - R² of value function predictions
- `rollout/ep_rew_mean` - Mean episode reward
- `rollout/ep_len_mean` - Mean episode length
- `time/fps` - Training frames per second
- `time/total_timesteps` - Cumulative timesteps

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
| - | `0.025` | Agent avoided hitting enemies at 0.08 |
| 13:20 10/1 | `0.05` | Still stuck with opposite spawners |
| 14:53 10/1 | `0.06` | ⭐ **CURRENT** - Rarely stuck, but still occasional |

### Learning Rate Schedule History

| Version | Schedule | Min LR | Rationale |
|---------|----------|--------|-----------|
| v1.1 | Constant | N/A | Original implementation |
| v1.2 | Linear | `0` | Added LR scheduling |
| **v1.0** | **Linear** | **`2e-5`** | ⭐ **CURRENT** - Prevent complete stop, allow continued fine-tuning |

---

## Next Steps & Recommendations

### High Priority

1. **Reduce corner hiding (30% occurrence)**
   - Consider increasing `existence_penalty` when near corners
   - Add "corner penalty" zone-based reward shaping
   - Possible new reward: `corner_penalty = -0.5` when within 100 units of corner

2. **Improve threat response consistency**
   - Current forward-then-spin behavior is good, reinforce it
   - Consider adding velocity-based rewards for maintaining momentum
   - May need to balance with spawner approach rewards

3. **Continue entropy tuning**
   - Current `0.06` eliminated suicidal behavior
   - Monitor if agent continues to improve or plateaus
   - May try `0.065` if corner hiding persists

### Medium Priority

4. **Enemy dodging enhancement**
   - Add reward shaping for maintaining distance from enemies
   - Possible: `enemy_distance_bonus = 0.05` when 100+ units from nearest enemy
   - Balance with objective pursuit

5. **Spawner vs enemy prioritization**
   - Current balance seems acceptable (no suicidal behavior)
   - May fine-tune if low-health enemies are consistently ignored
   - Consider "cleanup bonus" for clearing all enemies before next spawner

---

## Performance Summary

| Metric | Value | Change from v1.2 |
|--------|-------|------------------|
| **Suicidal Episodes** | 0% | -20% ✅ |
| **Corner Hiding** | 30% | New metric |
| **Phase 4 Reach** | >70% | Similar |
| **Stuck Episodes** | <10% | Similar/Improved |
| **Threat Management** | Good | Improved ✅ |

---

## Version History

- **v1.1** (2026-01-09 03:47): Suicidal PPO - Aggressive behavior, constant learning rate
- **v1.2** (2026-01-10 13:38): Suicidal PPO - 20% suicidal, linear LR schedule (to 0), ent_coef=0.05
- **v1.0** (2026-01-10 15:53): Non-Suicidal PPO - 0% suicidal, LR min floor (2e-5), ent_coef=0.06 ⭐ **CURRENT**
