# Non-Suicidal PPO v1.3 - Aggressive Thrust Checkpoint

**Timestamp:** January 12, 2026 - 16:45
**Model File:** `non-suicidal-1.3_20260112_1645.zip`
**Previous Version:** `non-suicidal-1.2_20260111_2202`
**TensorBoard Run:** `PPO_74`

---

## Status Summary

✅ **Breakthrough: Agent learned aggressive thrusting to kill spawners faster**

This version demonstrates a major tactical improvement where the agent now actively thrusts forward toward spawners to close distance and destroy them faster, while maintaining safe engagement distance (non-suicidal). The agent occasionally crashes into spawners but not in an aggressive/suicidal manner - it's a side effect of confident approach, not reckless behavior.

### Key Achievements
- **Aggressive forward thrust** - Agent now uses thrust action effectively to close distance with spawners
- **Faster spawner destruction** - Reduced time to kill spawners due to active approach
- **Maintained safe distance** - Still keeps relatively good distance, not rushing suicidally
- **Zero suicidal behavior** - 0% aggressive death rate maintained
- **Occasional spawner collisions** - Sometimes crashes spawners but not aggressively (navigation side effect)

---

## Critical Remaining Issues ⚠️

Despite the aggressive thrust breakthrough, two major behavioral problems persist:

### Issue #1: Corner/Wall Stuck Behavior
**Problem:** Agent still gets stuck in corners/against walls when spawners are on opposite side
- Symptom: Thrusts into wall repeatedly, making no progress
- Root cause: Agent sees "spawner far away" but doesn't recognize "I'm stuck against obstacle"
- Impact: Wastes time, can't complete phase efficiently
- Frequency: Moderate - happens multiple times per episode

**Why current approach doesn't solve it:**
- `wall_proximity` observation exists but agent hasn't learned to use it for stuck detection
- High entropy (0.1) helps but not enough - needs explicit stuck-escape reward shaping
- Removed penalties allow rotation but don't incentivize it when stuck

### Issue #2: Enemies Seen as Obstacles, Not Health Threats ⚠️ CRITICAL
**Problem:** Agent perceives enemies as "obstacles blocking line of sight to spawner", NOT as "threats that cause health loss"

**Current behavior pattern:**
```
Agent sees: Spawner behind enemies
Agent thinks: "Enemies are blocking my shots at spawner"
Agent does: Try to shoot around enemies or wait for clear line
Agent SHOULD think: "Enemies will damage me (-80 per hit), I need to kill them first"
Agent SHOULD do: Aggressively destroy enemies before engaging spawner
```

**Evidence:**
- Agent ignores enemies that aren't directly in firing path
- Takes avoidable damage by not clearing enemies first
- High `destroy_enemy` reward (170) and `hit_enemy` (30) haven't changed this perception
- Agent rushes to spawners even when surrounded by enemies

**Why high rewards aren't working:**
- Reward structure says "killing enemies is good" (170)
- But agent's value function learned: "killing enemies = removes obstacle to spawner = instrumental goal"
- Agent hasn't learned: "NOT killing enemies = I take damage (-80) = lose health = BAD"
- The association between "enemy presence" → "future health loss" is weak

**Root cause hypothesis:**
- Agent's temporal credit assignment problem: damage happens AFTER ignoring enemy, reward happens BEFORE
- Value function focuses on spawner destruction (180 + 500 phase) as terminal goal
- Enemy kills are seen as means to reach spawner, not as survival necessity
- May need explicit "health loss rate" observation or "enemies within damage range" feature

**Impact:**
- Agent takes unnecessary damage throughout episode
- Health depletes faster than necessary
- Dies in later phases when enemy count/speed increases
- Prevents reaching higher phases consistently

---

## Key Changes from v1.2 (Behavior-Influencing)

### 1. Increased Entropy Coefficient ⭐ PRIMARY CHANGE
**Previous:** `ent_coef = 0.065`
**Current:** `ent_coef = 0.1` (+54% increase)

**Why this matters:**
- Higher entropy = more exploration of thrust action combinations
- Agent discovered that thrusting forward is safe and effective
- Enabled learning aggressive approach patterns without over-committing
- Prevents policy from getting stuck in "cautious movement only" local minima

### 2. Removed Movement/Rotation Penalties ⭐ CRITICAL CHANGE
**Previous:**
- `potential_further = -0.12` (penalty for moving away)
- `orientation_further = -0.12` (penalty for rotating away)

**Current:**
- `potential_further = 0.0` (NO penalty)
- `orientation_further = 0.0` (NO penalty)

**Why this matters:**
- Agent is no longer punished for temporary retreats or rotation adjustments
- Can now thrust forward confidently without fear of "wrong direction" penalties
- Enables more natural approach patterns: thrust → adjust → thrust → engage
- Removes artificial constraints on movement policy gradient
- Agent learns from positive rewards (getting closer) rather than avoiding penalties

### 3. High Enemy Engagement Rewards (Maintained from v1.2)
**Unchanged:**
- `destroy_enemy = 170.0` (very high)
- `hit_enemy = 30.0` (3x shooting cost)
- `destroy_spawner = 180.0`

**Why this matters:**
- High enemy rewards prevent suicidal rushing past enemies
- Agent thrusts forward but still prioritizes clearing threats
- Maintains safe engagement discipline while being aggressive

---

## Complete Training Configuration

### Critical PPO Hyperparameters

| Parameter | Value | Change from v1.2 | Impact on Behavior |
|-----------|-------|------------------|---------------------|
| **ent_coef** | `0.1` | +54% (was 0.065) | ⭐ Enabled thrust exploration |
| **learning_rate** | `linear_schedule_3m(1e-4)` | Unchanged | Stable learning |
| **n_steps** | `4096` | Unchanged | 16,384 steps/update (4 envs) |
| **batch_size** | `256` | Unchanged | - |
| **n_epochs** | `10` | Unchanged | - |
| **gamma** | `0.99` | Unchanged | - |
| **gae_lambda** | `0.95` | Unchanged | - |
| **clip_range** | `0.2` | Unchanged | - |
| **vf_coef** | `0.5` | Unchanged | - |
| **max_grad_norm** | `0.5` | Unchanged | - |

### Network Architecture (Unchanged)
- **Policy Network (Actor):** `[256, 256]` (2 hidden layers)
- **Value Network (Critic):** `[256, 256]` (2 hidden layers)

### Learning Rate Schedule (Unchanged)
```
1e-4 → Linear decay to 2e-5 at 3M timesteps → Constant 2e-5 thereafter
```

---

## Complete Reward Function

### Rewards That Drive Aggressive Thrust Behavior

| Reward | Value | Change from v1.2 | Behavior Impact |
|--------|-------|------------------|-----------------|
| **potential_closer** | `+0.1` | Unchanged | Rewards moving toward spawners |
| **orientation_closer** | `+0.1` | Unchanged | Rewards rotating toward spawners |
| **potential_further** | `0.0` | ⭐ Was -0.12 | No penalty for retreating → confident thrusting |
| **orientation_further** | `0.0` | ⭐ Was -0.12 | No penalty for wrong rotations → natural movement |
| **destroy_spawner** | `+180.0` | Unchanged | High objective value |
| **destroy_enemy** | `+170.0` | Unchanged | Prevents suicidal rushing |
| **hit_enemy** | `+30.0` | Unchanged | Encourages threat clearing |
| **take_damage** | `-80.0` | Unchanged | Discourages reckless collisions |
| **death** | `-200.0` | Unchanged | Strong survival incentive |
| **shot_fired** | `-0.05` | Unchanged | Tiny cost (shooting is cheap) |
| **existence_penalty** | `-0.015` | Unchanged | Time pressure |
| **accuracy_bonus** | `+0.2` | Unchanged | Reward aimed shots (30° threshold) |

### Full Reward Breakdown

**Positive Rewards (Objectives):**
- destroy_enemy: +170.0
- destroy_spawner: +180.0
- phase_complete: +500.0
- hit_enemy: +30.0
- accuracy_bonus: +0.2 (within 30° of target)
- potential_closer: +0.1 (per step moving toward spawner)
- orientation_closer: +0.1 (per step rotating toward spawner)

**Negative Penalties:**
- death: -200.0
- take_damage: -80.0
- shot_fired: -0.05
- existence_penalty: -0.015 (per step)

**Removed Penalties (⭐ KEY CHANGE):**
- potential_further: 0.0 (was -0.12) → No retreat penalty
- orientation_further: 0.0 (was -0.12) → No wrong rotation penalty

---

## Observation Space Configuration (18-dimensional)

**IMPORTANT:** To recreate this model, the observation space MUST match exactly:

### Player State (8 values)
1. **x position** - Normalized by SCREEN_WIDTH (1000)
2. **y position** - Normalized by SCREEN_HEIGHT (700)
3. **x velocity** - Normalized by 10.0
4. **y velocity** - Normalized by 10.0
5. **velocity magnitude** - sqrt(vx² + vy²) / PLAYER_MAX_VELOCITY (6.0)
6. **wall proximity** - min(dist to all 4 walls) / max(SCREEN_WIDTH, SCREEN_HEIGHT)
7. **angle** - Player rotation angle / 360.0
8. **health** - Current health / PLAYER_MAX_HEALTH (100)

### Nearest Enemy (3 values)
9. **distance** - Euclidean distance / max(SCREEN_WIDTH, SCREEN_HEIGHT)
10. **cos(relative_angle)** - Cosine of angle relative to player facing direction
11. **sin(relative_angle)** - Sine of angle relative to player facing direction

### Nearest Spawner (3 values)
12. **distance** - Euclidean distance / max(SCREEN_WIDTH, SCREEN_HEIGHT)
13. **cos(relative_angle)** - Cosine of angle relative to player facing direction
14. **sin(relative_angle)** - Sine of angle relative to player facing direction

### Game State (4 values)
15. **phase** - Current phase / MAX_PHASE (5)
16. **enemy_count** - Number of living enemies / 20.0 (capped at 1.0)
17. **spawner_count** - Number of living spawners / 10.0 (capped at 1.0)
18. **can_shoot** - Boolean (1.0 if off cooldown, 0.0 if on cooldown)

**Observation Space Box:**
```python
spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)
```

---

## Action Space (Discrete - 5 actions)

| Action ID | Action Name | Effect |
|-----------|-------------|--------|
| **0** | `nothing` | No action, drift with current velocity |
| **1** | `thrust` | ⭐ Apply forward thrust (PLAYER_THRUST = 0.3) |
| **2** | `rotate_left` | Rotate counter-clockwise (PLAYER_ROTATION_SPEED = 5.0°/frame) |
| **3** | `rotate_right` | Rotate clockwise (PLAYER_ROTATION_SPEED = 5.0°/frame) |
| **4** | `shoot` | Fire projectile (if off cooldown) |

---

## Game Environment Configuration

### Key Game Mechanics
- **SCREEN_SIZE:** 1000 x 700 pixels
- **PLAYER_THRUST:** 0.3 (forward acceleration per frame)
- **PLAYER_MAX_VELOCITY:** 6.0 (speed cap)
- **PLAYER_ROTATION_SPEED:** 5.0°/frame
- **PLAYER_FRICTION:** 0.98 (velocity decay)
- **PLAYER_SHOOT_COOLDOWN:** 10 frames (~0.167s @ 60fps)
- **MAX_STEPS:** 5000 per episode
- **INITIAL_SPAWNERS:** 2
- **SPAWNER_RADIUS:** 42 (collision radius)
- **ENEMY_RADIUS:** 15
- **ENEMY_SPEED:** 2.5

---

## Why This Version Works: Behavior Analysis

### The Physics of Aggressive Thrust

The removal of movement/rotation penalties fundamentally changed the agent's risk-reward calculation for thrust actions:

**Before (v1.2) - Penalty-Constrained Policy:**
```
Thrust forward decision:
  + positive: potential_closer (+0.1)
  - negative: potential_further (-0.12) if wrong angle
  - negative: orientation_further (-0.12) if wrong rotation
  → Expected value: LOW (penalties dominate if not perfectly aligned)
  → Learned policy: Only thrust when perfectly aligned (rare)
  → Result: Slow, cautious approach
```

**After (v1.3) - Positive-Driven Policy:**
```
Thrust forward decision:
  + positive: potential_closer (+0.1) when moving toward spawner
  + positive: orientation_closer (+0.1) when rotating toward spawner
  - negative: NONE for temporary misalignment
  → Expected value: HIGH (accumulate positive rewards over time)
  → Learned policy: Thrust frequently, adjust angle dynamically
  → Result: Fast, aggressive approach while maintaining safety
```

### Why It's Not Suicidal

Despite aggressive thrusting, the agent maintains safe behavior because:

1. **High enemy engagement rewards (170 destroy, 30 hit)** - Agent prioritizes clearing threats before committing to spawner
2. **High damage penalty (-80)** - Discourages reckless collisions
3. **Death penalty (-200)** - Strong survival incentive
4. **High entropy (0.1)** - Policy remains stochastic, doesn't over-commit to single strategy

### The "Occasional Crash" Phenomenon

The agent sometimes collides with spawners, but this is **navigation error**, not suicidal behavior:

- Agent is closing distance aggressively → high forward velocity
- Rotation speed (5°/frame) has natural turning radius
- Spawner radius (42) is relatively large
- Collision occurs when: thrust momentum + turning radius > safe distance
- This is acceptable because: spawner collision deals 0 damage (spawners don't hurt player)

**Key distinction:**
- Suicidal = intentionally rushing into enemies (takes damage)
- Navigation error = overshooting spawner position (no damage)

---

## Evolution Timeline

### v1.3 Changes (16:45 12/1/2026) ⭐ CURRENT
- **Increased `ent_coef` to 0.1** (+54% from 0.065) → Enabled thrust exploration
- **Removed penalty for moving away** (`potential_further`: -0.12 → 0.0) → Confident forward movement
- **Removed penalty for rotating away** (`orientation_further`: -0.12 → 0.0) → Natural navigation
- **Result:** Agent learned aggressive thrusting to kill spawners faster while maintaining safety

### v1.2 Changes (22:02 11/1/2026)
- Massive enemy reward boost (destroy: 150, hit: 30)
- Extended training (>4M timesteps)
- 50% corner escape success
- Focused on enemy engagement priority

### v1.1 Changes (16:12 11/1/2026)
- Optimal LR schedule (decay to 2e-5 at 3M, then constant)
- Increased entropy to 0.065
- Enabled extended training

### v1.0 Changes (15:53 10/1/2026)
- First non-suicidal breakthrough
- Balanced rewards

---

## Performance Metrics (PPO_74)

**TensorBoard Run:** `logs/rotation_ppo/PPO_74`

### Expected Improvements vs v1.2

| Metric | v1.2 (Previous) | v1.3 (Current) | Change |
|--------|-----------------|----------------|--------|
| **Spawner Kill Speed** | Moderate | Fast ⭐ | +30-40% estimated |
| **Thrust Action Frequency** | Low | High ⭐ | Significantly increased |
| **Suicidal Episodes** | 0% | 0% | Maintained ✓ |
| **Corner Escape Rate** | 50% | TBD | Expected similar or better |
| **Spawner Collisions** | Rare | Occasional | Navigation side effect |
| **Average Episode Reward** | ~850-950 | TBD | Expected +10-15% |

### Key Behavioral Patterns to Monitor

**In TensorBoard, look for:**
1. **Action distribution:** Increased action 1 (thrust) frequency
2. **Episode length:** Potentially shorter (faster spawner kills)
3. **Entropy loss:** Should stay around 0.1 (high exploration maintained)
4. **Value loss:** Should decrease smoothly (learning engaged thrust patterns)
5. **Take damage rate:** Should stay low (maintains safety despite aggression)

---

## Goals for Next Version (v1.4)

### Primary Objectives (Addressing Critical Issues)

1. **Fix stuck behavior** - Agent must learn to detect and escape wall/corner stuck situations
2. **Fix enemy perception** - Agent must learn enemies are health threats, not just obstacles

### Experiments to Address Issue #1: Stuck Behavior

**A. Add explicit stuck-detection reward shaping:**
```python
# In reward function (game/arena.py or constants.py)
if velocity_magnitude < 1.0 and thrust_action_taken:
    reward += REWARDS['stuck_thrust_penalty']  # e.g., -1.0

if previous_velocity < 1.0 and current_velocity > 2.0:
    reward += REWARDS['unstuck_bonus']  # e.g., +2.0
```
- Hypothesis: Explicit penalty for "thrusting but not moving" teaches stuck detection
- Hypothesis: Reward for escaping low-velocity state incentivizes rotation when stuck
- Risk: Low - these are clear signals

**B. Increase wall proximity salience:**
- Add second wall observation: `wall_distance_in_thrust_direction`
- Formula: Distance to wall if agent continues thrusting forward
- Hypothesis: Agent can learn "if wall_ahead < 100 and thrusting → will get stuck → rotate first"
- Complexity: Medium (requires observation space change to 19D)

**C. Curriculum learning for stuck scenarios:**
- Spawn player near walls more frequently during training
- Force agent to practice wall escape scenarios
- Complexity: High (requires training loop modification)

### Experiments to Address Issue #2: Enemy Perception

**A. Add "danger zone" observation ⭐ RECOMMENDED:**
```python
# In get_observation() (game/arena.py)
enemies_in_danger_zone = count_enemies_within_radius(player, radius=150)
obs.append(enemies_in_danger_zone / 5.0)  # Normalize by max expected
```
- Hypothesis: Explicit "enemies nearby" signal helps agent associate proximity with future damage
- Changes observation space to 19D (or 20D with wall_ahead)
- Agent can learn: "danger_zone_count > 2 → prioritize clearing enemies"

**B. Increase take_damage penalty dramatically:**
- Current: `-80` per hit
- Try: `-120` or `-150` per hit
- Hypothesis: Stronger penalty makes value function prioritize damage avoidance
- Risk: May make agent too cautious (test carefully)

**C. Add health-loss-rate reward shaping:**
```python
# Track health change
health_delta = current_health - previous_health
if health_delta < 0:
    # Already taking damage penalty, but add temporal penalty too
    reward += health_delta * 0.5  # Additional penalty proportional to health lost
```
- Hypothesis: Continuous health-loss feedback strengthens temporal credit assignment
- Risk: Low - reinforces existing penalty

**D. Reduce spawner reward temporarily:**
- Current: `destroy_spawner = 180`
- Try: `destroy_spawner = 140` (same as destroy_enemy)
- Hypothesis: Balancing rewards prevents "rush spawner at all costs" behavior
- Agent learns: "spawners and enemies are equally important"
- Risk: Medium - may reduce objective focus

**E. Add "surrounded by enemies" penalty:**
```python
if len(enemies_within_radius(player, 200)) >= 3:
    reward += REWARDS['surrounded_penalty']  # e.g., -5.0 per step
```
- Hypothesis: Being surrounded is predictive of future damage
- Teaches agent to clear local area before advancing
- Risk: Low - clear danger signal

### Recommended Approach for v1.4

**Phase 1: Fix enemy perception first (most critical)**
1. Add "enemies in danger zone" observation (19D space)
2. Increase take_damage to -120
3. Add "surrounded by enemies" penalty (-5.0 per step when 3+ enemies within 200 units)
4. Train 3M timesteps, evaluate

**Phase 2: Fix stuck behavior**
5. Add stuck-detection penalties (stuck_thrust_penalty: -1.0)
6. Add unstuck bonus (+2.0)
7. Train additional 2M timesteps

**Why this order:**
- Enemy perception affects survival in ALL scenarios
- Stuck behavior affects efficiency but not immediate survival
- Easier to test stuck fixes once agent survives longer

### Success Criteria for v1.4
- **Enemy clearing:** Agent kills ≥70% of enemies before engaging spawner
- **Damage reduction:** Average damage taken per episode reduced by 30%
- **Stuck rate:** Reduced stuck episodes by 50%
- **Survival:** Reach phase 3+ in ≥60% of episodes
- **Zero suicidal behavior maintained**
- **Maintain aggressive thrust** (don't regress to cautious movement)

---

## Training Command Used

```bash
cd Part-II
python train.py --env rotation --algo ppo --timesteps 5000000 --lr 1e-4
```

---

## Version History

### Non-Suicidal Era (Current)
1. **v1.3** (16:45 12/1/2026) - ⭐ **Aggressive thrust breakthrough**, removed movement penalties, increased entropy to 0.1
2. **v1.2** (22:02 11/1/2026) - Massive enemy rewards, 50% corner escape
3. **v1.1** (16:12 11/1/2026) - Optimal LR schedule, entropy 0.065
4. **v1.0** (15:53 10/1/2026) - First non-suicidal breakthrough

### Suicidal Era (Historical)
5. **v1.2 suicidal** (13:38 10/1/2026) - LR scheduling, still suicidal
6. **v1.1 suicidal** (03:47 09/1/2026) - High rewards, chaotic

---

## Technical Summary

### Critical Parameters That Define This Model

**To recreate v1.3 behavior, these are essential:**

1. **Entropy coefficient:** `ent_coef = 0.1` (not 0.065!)
2. **Movement penalties:** `potential_further = 0.0` (not -0.12!)
3. **Rotation penalties:** `orientation_further = 0.0` (not -0.12!)
4. **Enemy rewards:** destroy=170, hit=30 (prevent suicidal)
5. **Observation space:** 18-dimensional (see detailed breakdown above)
6. **LR schedule:** Linear decay to 2e-5 at 3M, then constant
7. **Thrust mechanics:** PLAYER_THRUST=0.3, PLAYER_ROTATION_SPEED=5.0°/frame

**Key Insight:** The combination of high entropy (0.1) + no movement penalties (0.0) created a policy that:
- Explores thrust actions frequently (entropy)
- Doesn't fear temporary misalignment (no penalties)
- Accumulates positive rewards from forward progress (potential_closer +0.1)
- Maintains safety through high enemy engagement rewards (170/30)

This is why the agent "learned to thrust to kill spawners faster" - the gradient finally favored aggressive forward movement after removing the penalty constraints.

---

## Conclusion

Version 1.3 represents a major tactical breakthrough in movement efficiency. By removing penalties for temporary retreat/misalignment and increasing entropy, the agent discovered that aggressive forward thrusting is both safe and effective. The occasional spawner collisions are acceptable navigation errors (no damage) and represent the agent's confidence in closing distance quickly.

**Key lesson learned:** **Sometimes removing constraints (penalties) teaches better than adding incentives (rewards).** The agent always "wanted" to thrust forward (potential_closer reward), but penalties held it back. Once freed, it learned the optimal aggressive-but-safe engagement pattern.

### However, Two Critical Issues Remain:

1. **Stuck behavior** - Agent still gets trapped in corners/against walls when spawners are on opposite side
2. **Enemy perception problem** ⚠️ **MOST CRITICAL** - Agent sees enemies as "obstacles blocking spawner shots", NOT as "threats causing health loss"

**Why Issue #2 is critical:**
- Agent's value function learned wrong causal model: "kill enemies → clear path to spawner" (instrumental)
- Should learn: "NOT killing enemies → I take damage → health depletes → death" (survival)
- High enemy rewards (170 destroy, 30 hit) work for instrumental goal, but don't fix perception
- Temporal credit assignment problem: damage penalty happens AFTER ignoring enemies, not during

**Recommended next steps for v1.4:**
1. **Add "enemies in danger zone" observation** (count within 150 units) → explicit threat awareness
2. **Increase take_damage penalty to -120** → stronger damage avoidance signal
3. **Add "surrounded by enemies" penalty** (-5.0 per step when 3+ nearby) → predictive danger signal
4. **Add stuck-detection penalties** (stuck_thrust: -1.0, unstuck_bonus: +2.0) → escape learning

These changes should teach the agent that enemy proximity predicts future damage, shifting perception from "obstacles" to "threats".
