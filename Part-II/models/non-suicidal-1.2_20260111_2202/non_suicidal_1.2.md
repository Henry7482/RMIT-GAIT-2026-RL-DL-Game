# Non-Suicidal PPO v1.2 - Checkpoint

**Timestamp:** January 11, 2026 - 22:02  
**Model File:** `non-suicidal-1.2_20260111_2202.zip`  
**Previous Version:** `non-suicidal-1.1_20260111_1612`

---

## Status Summary

✅ **Breakthrough: 50% corner escape success with aggressive targeting**

This version achieves a significant behavioral improvement where the agent now has a 50/50 chance of escaping corner-stuck situations by turning and actively shooting at spawners and enemies, rather than passively waiting or hiding.

### Achievements
- **Corner escape capability** - 50% success rate when stuck in corners, actively turns to engage
- **Maintained aggression** - Continues shooting at spawners and enemies even from difficult positions
- **Zero suicidal behavior** - Still maintains safe engagement distance
- **Improved tactical awareness** - Recognizes when to rotate and shoot rather than hide

### Current Limitations
- **50% corner stuck rate remains** - Still gets stuck in corners half the time
- **Inconsistent escape execution** - Success depends on specific corner position and enemy/spawner locations
- **Opposite spawner challenge persists** - Struggles when spawner spawns behind current position

---

## Key Changes from v1.1 (non-suicidal-1.1_20260111_1612)

### 1. Extended Training Duration ⭐
**Previous:** ~3-4M timesteps  
**Current:** Extended training beyond 4M timesteps (exact duration from latest training run)

**Rationale:**
- 22:02 11/1/2026: Extended training leveraging the LR schedule from v1.1
- The constant 2e-5 learning rate after 3M timesteps enables continued learning
- Agent discovered corner escape strategies through extended exploration
- 50/50 corner escape behavior emerged from additional training iterations

### 2. Behavioral Discovery ⭐ MAJOR IMPROVEMENT
**Previous:** Gets stuck in corners, stops shooting  
**Current:** 50% chance to turn and keep shooting when cornered

**Key Behavior Pattern:**
- When cornered, agent now recognizes it's stuck
- Instead of passive waiting, executes rotation maneuvers
- Maintains shooting action while rotating to find spawners/enemies
- Successfully escapes 50% of corner situations
- Other 50% still results in corner-stuck behavior (needs further improvement)

**Example Scenario:**
```
Before (v1.1):
Player in corner → Nearest spawner opposite side → Agent hides → Gets stuck

After (v1.2):
Player in corner → Nearest spawner opposite side → Agent rotates → Shoots while turning → 
  ↓ 50% success                    ↓ 50% failure
Finds target and escapes        Still gets stuck in corner
```

### 3. Reward Function Changes ⭐ SIGNIFICANT ADJUSTMENTS
**Previous (v1.1):** Balanced rewards focused on spawners  
**Current (v1.2):** Heavily increased enemy-kill rewards to encourage finishing enemies

**Key Reward Changes:**
- `destroy_enemy`: **85.0 → 150.0** (+76% increase!) - Strongly encourages killing remaining enemies
- `hit_enemy`: **10.0 → 30.0** (+200% increase!) - Rewards engagement with enemies
- `take_damage`: **-45.0 → -50.0** - Slightly more scary
- `survival_bonus`: **0.0 → -0.05** - Actively discourages passive waiting

**Rationale:**
- 21:22 11/1/2026: Massive increase to `destroy_enemy` (150) and `hit_enemy` (30)
- Goal: Encourage agent to finish remaining enemies before moving to next spawner
- Prevent "ignore enemies, rush spawner" behavior that causes damage
- Early timesteps with high enemy rewards teach that enemies are worthy targets
- Combined with extended training, enabled corner escape + enemy engagement

### 4. Hyperparameters
**No changes from v1.1** - Same optimal configuration
- Same `ent_coef=0.065`
- Same LR schedule (linear to 2e-5 at 3M, then constant)

---

## Complete Training Configuration

### PPO Hyperparameters (Stable-Baselines3)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **policy** | `"MlpPolicy"` | Multi-layer perceptron policy |
| **learning_rate** | `linear_schedule_3m(1e-4, timesteps)` | Decays to `2e-5` at 3M, then constant |
| **n_steps** | `4096` | Steps per environment per update (16,384 total with 4 envs) |
| **batch_size** | `256` | Minibatch size for gradient updates |
| **n_epochs** | `10` | Epochs when optimizing surrogate loss |
| **gamma** | `0.99` | Discount factor for future rewards |
| **gae_lambda** | `0.95` | GAE lambda for bias-variance tradeoff |
| **clip_range** | `0.2` | PPO clipping parameter |
| **clip_range_vf** | `None` | Value function clipping (disabled) |
| **ent_coef** | `0.065` | Entropy coefficient for exploration |
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

### Learning Rate Schedule

```
LR
^
1e-4  |●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━○━━━━━━━━━━━━━━━━━━━━━━━━━>
      |                               ╱
      |                              ╱   ← Extended training
      |                             ╱      discovers corner escape
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

| Reward Type | Value | Trigger Condition | Notes |
|-------------|-------|-------------------|-------|
| **destroy_enemy** | `+150.0` ⭐ | Enemy health reaches 0 | **Increased from 85** (v1.1) - Major boost to encourage enemy clearing |
| **destroy_spawner** | `+180.0` | Spawner health reaches 0 | Unchanged |
| **phase_complete** | `+500.0` | All spawners destroyed in current phase | Unchanged |
| **hit_enemy** | `+30.0` ⭐ | Projectile hits enemy (even if doesn't kill) | **Increased from 10** (v1.1) - Tripled to reward engagement |
| **accuracy_bonus** | `+0.2` | Shooting when aimed within 20° of target | Unchanged |
| **potential_closer** | `+0.1` | Moving closer to nearest spawner (per step) | Unchanged |
| **orientation_closer** | `+0.1` | Rotating towards nearest spawner (per step) | Unchanged |

### Negative Rewards (Penalties)

| Penalty Type | Value | Trigger Condition | Notes |
|--------------|-------|-------------------|-------|
| **death** | `-200.0` | Player health reaches 0 | Unchanged |
| **take_damage** | `-50.0` ⭐ | Player takes damage from enemy collision | **Increased from -45** (v1.1) - More scary to prevent recklessness |
| **potential_further** | `-0.12` | Moving away from nearest spawner (per step) | Unchanged |
| **orientation_further** | `-0.12` | Rotating away from nearest spawner (per step) | Unchanged |
| **shot_fired** | `-0.05` | Each time player shoots (ammo cost) | Unchanged |
| **existence_penalty** | `-0.015` | Every step (discourage passivity) | Unchanged |
| **survival_bonus** | `-0.05` ⭐ | Every step alive | **Changed from 0.0** (v1.1) - Now penalty to prevent passive waiting |

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
| **Training Duration** | >4M timesteps (extended from v1.1) ⭐ |
| **TensorBoard Run** | `PPO_52` (or latest run) ⭐ |

---

## Gradient Computation Details

### Loss Function
```
Total Loss = Policy Loss + vf_coef × Value Loss - ent_coef × Entropy

Where:
- Policy Loss = PPO clipped surrogate objective
- Value Loss = MSE between predicted value and return
- Entropy = -Σ π(a|s) log π(a|s) (exploration bonus)
```

### PPO Clipped Surrogate Objective
```
L^CLIP(θ) = Ê_t [min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t)]

Where:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) (probability ratio)
- Â_t = Advantage estimate (GAE)
- ε = 0.2 (clip_range)
```

### Generalized Advantage Estimation (GAE)
```
Â_t = Σ_(l=0)^∞ (γλ)^l δ_(t+l)

Where:
- δ_t = r_t + γV(s_(t+1)) - V(s_t) (TD residual)
- γ = 0.99 (discount factor)
- λ = 0.95 (gae_lambda)
```

### Gradient Clipping
```
if ||∇L|| > max_grad_norm:
    ∇L ← max_grad_norm × (∇L / ||∇L||)

Where:
- max_grad_norm = 0.5
```

---

## Evolution Timeline

### v1.2 Changes (21:22-22:02 11/1/2026)
- **Major reward restructuring** - Massively increased enemy-kill rewards to encourage engagement
  - `destroy_enemy`: 85 → **150** (+76%)
  - `hit_enemy`: 10 → **30** (+200%)
  - `take_damage`: -45 → **-50** (more scary)
  - `survival_bonus`: 0.0 → **-0.05** (active penalty)
- **Extended training to ~4M timesteps** - Leveraged v1.1's LR schedule for continued learning
- **Corner escape discovered** - 50% success rate emerged from combined reward changes + extended training
- **Behavioral milestone** - First version showing tactical corner escape with aggressive enemy engagement

### v1.1 Changes (16:12 11/1/2026)
- **Increased `ent_coef` to `0.065`** - Better exploration capacity
- **Introduced optimal LR schedule** - Linear decay to 2e-5 at 3M, then constant
- **Enabled extended training** - Can now train beyond 5M+ timesteps effectively

### v1.0 Changes (15:53 10/1/2026)
- **First non-suicidal breakthrough** - Zero suicidal episodes
- **Balanced rewards** - destroy_enemy=85, destroy_spawner=180, take_damage=-45, death=-200
- **Entropy tuning** - `ent_coef=0.06` discovered as sweet spot

### Pre-v1.0 Suicidal Era
- **v1.2 suicidal (13:38 10/1/2026)** - Added LR scheduling, still suicidal
- **v1.1 suicidal (03:47 09/1/2026)** - High rewards, chaotic behavior
- **Earlier versions** - Various reward structures, all resulted in suicidal behavior

---

## Goals for Next Version (v1.3)

### Primary Objective
**Improve corner escape to 75%+ success rate**

### Experiments to Try

1. **Slightly increase entropy coefficient**
   - Try `ent_coef=0.068` or `0.07`
   - Hypothesis: More exploration might discover more reliable corner escape strategies
   - Risk: Too high (>0.075) may cause reckless behavior

2. **Extend training further**
   - Target 6-8M timesteps
   - The 50% escape emerged after 4M, more training may improve consistency
   - Monitor for plateau in TensorBoard metrics

3. **Adjust orientation rewards**
   - Consider slightly increasing `orientation_closer` bonus (currently +0.1)
   - Hypothesis: Stronger rotation rewards might encourage more active turning when stuck
   - Try: `orientation_closer=+0.15`, keep `orientation_further=-0.12`

4. **Add corner detection reward**
   - Implement small bonus for successful corner escapes
   - Requires code changes to detect corner-stuck-then-escape patterns
   - Complexity: Medium effort, high potential impact

5. **Monitor specific behaviors**
   - Track: % of episodes where agent gets cornered
   - Track: % of cornered episodes that result in escape
   - Track: Average time spent in corner before escape/getting stuck
   - Use these metrics to guide next hyperparameter adjustments

### Success Criteria for v1.3
- Corner escape rate: 75%+ (up from 50%)
- Zero suicidal behavior maintained
- Average episode reward: >1000 (currently ~800-900)
- No regression in spawner destruction rate

---

## Performance Summary

### Metrics Comparison (v1.2 vs v1.1)

| Metric | v1.1 (Previous) | v1.2 (Current) | Change |
|--------|-----------------|----------------|--------|
| **Corner Escape Rate** | 0% (always stuck) | 50% | +50% ⭐ |
| **Aggressive Shooting** | Stops when cornered | Continues when cornered | ✓ Improved |
| **Suicidal Episodes** | 0% | 0% | Maintained ✓ |
| **Training Duration** | ~3-4M timesteps | >4M timesteps | Extended |
| **Spawner Destruction** | Moderate | Moderate | Stable |
| **Average Episode Reward** | ~800-900 | ~850-950 (est.) | +5-10% |

### Key Behavioral Improvements

**Corner Situation Handling:**
```
v1.1: Corner → Stop shooting → Hide → Stuck (100%)
v1.2: Corner → Keep shooting → Rotate → Escape (50%) | Stuck (50%)
```

**Enemy Engagement Priority:**
```
v1.1: Ignores remaining enemies, rushes to next spawner (destroy_enemy: 85, hit: 10)
v1.2: Actively hunts enemies before moving (destroy_enemy: 150, hit: 30)
```

**Aggression Maintenance:**
```
v1.1: Passive when stuck
v1.2: Active even when stuck (turns + shoots), heavily rewarded for enemy hits
```

---

## Version History

### Non-Suicidal Era (Current)
1. **v1.2** (22:02 11/1/2026) - **Massive enemy reward boost** (150 destroy, 30 hit), **50% corner escape**, enemy engagement priority
2. **v1.1** (16:12 11/1/2026) - Optimal LR schedule discovered, increased entropy to 0.065
3. **v1.0** (15:53 10/1/2026) - First non-suicidal breakthrough, balanced rewards

### Suicidal Era (Historical)
4. **v1.2 suicidal** (13:38 10/1/2026) - Learning rate scheduling, still suicidal
5. **v1.1 suicidal** (03:47 09/1/2026) - High rewards, chaotic exploration

---

## Technical Notes

### Why 50% Corner Escape Works

The corner escape behavior likely emerged through:

1. **Massively Increased Enemy Engagement Rewards** ⭐
   - `destroy_enemy` (+150) and `hit_enemy` (+30) are now HIGHLY rewarding
   - Agent learns early that shooting at enemies is extremely valuable (30 per hit!)
   - When cornered with enemies nearby, rotation + shooting yields huge expected value
   - Combined with `survival_bonus` penalty (-0.05), agent can't afford to wait passively
   - Reward ratio: hit_enemy (30) vs shot_fired (-0.05) = **600:1** - shooting is almost free!

2. **Extended Policy Exploration**
   - Constant 2e-5 LR after 3M enabled continued learning with new reward structure
   - Agent explored rotation + shooting combinations in corner scenarios
   - Discovered that continuous rotation while shooting can find escape paths
   - High enemy rewards reinforce this aggressive rotation behavior

3. **Reward Signal Accumulation**
   - `hit_enemy` (+30) strongly encourages shooting even when cornered
   - `orientation_closer` (+0.1) rewards rotating towards spawners
   - `shot_fired` (-0.05) is tiny compared to hit rewards (600x difference!)
   - `existence_penalty` (-0.015) + `survival_bonus` (-0.05) = -0.065/step passive cost
   - Combined: Shooting while rotating yields massive net positive expected value

4. **Stochastic Policy Advantage**
   - Policy network outputs probabilistic action distribution
   - In corner situations, high entropy allows varied rotation strategies
   - 50% success suggests two distinct policy modes:
     - Mode A: Rotate + shoot aggressively → escape (learned strategy)
     - Mode B: Minimal rotation → stuck (fallback behavior)

### Remaining Challenge

**Why 50% failure rate persists:**
- Complex state space in corners (position, velocity, enemy/spawner angles)
- Agent hasn't learned to reliably distinguish "escapable" vs "trap" corners
- May need more training data in corner scenarios
- Or additional reward shaping to emphasize corner escape

---

## Training Logs

**TensorBoard Run:** Check latest PPO run in `logs/rotation_ppo/`

**Key Metrics to Monitor:**
- `rollout/ep_rew_mean` - Average episode reward
- `train/entropy_loss` - Exploration level (should stay ~0.065)
- `train/policy_loss` - Policy optimization progress
- `train/value_loss` - Value function accuracy

**Expected Behavior:**
- Episode rewards should be higher than v1.1 due to corner escapes
- Entropy should remain stable (good exploration balance)
- No sudden spikes in `take_damage` penalty rate (maintains safe behavior)

---

## Conclusion

Version 1.2 represents a significant behavioral milestone: the first model to demonstrate tactical corner escape while maintaining non-suicidal behavior. The 50% escape rate emerged from a powerful combination:

1. **Massive enemy engagement rewards** (destroy_enemy: 150, hit_enemy: 30)
2. **Extended training** (~4M timesteps with stable 2e-5 LR)
3. **Active passivity penalty** (survival_bonus: -0.05)

The agent now understands that shooting at enemies is extremely valuable (30 per hit, 150 per kill, only -0.05 to shoot), making aggressive rotation + shooting the optimal strategy even when cornered.

The next step (v1.3) should focus on improving corner escape consistency to 75%+, potentially through slight entropy increases or further training with the current reward structure.

**Key Takeaway:** The massive increase in enemy-kill rewards (150 destroy, 30 hit) fundamentally changed agent behavior from "ignore enemies, rush spawner" to "actively hunt enemies". Combined with extended training at stable LR (2e-5), the agent discovered that aggressive engagement is far more rewarding than passive hiding. This reward restructuring is the breakthrough that enabled corner escape behavior.
