log: directional_dqn/DQN_52

# Pathfinder DQN v1.0 - Checkpoint

**Timestamp:** January 16, 2026 - 14:30  
**Model File:** `pathfinder-1.0_20260116_1430.zip`  
**Original File:** `directional_dqn_best_300000_steps.zip`  
**Training Steps:** 300,000 timesteps  
**Algorithm:** DQN (Deep Q-Network)  
**Environment:** Directional (WASD-style controls)

---

## Status Summary

**Achievements:**
- ✅ Successfully focuses on defeating enemy spawners 

**Limitations:**
- ❌ **Enemy Ignorance**: Completely ignores the presence of enemies 

**Behavior Classification:** "Pathfinder" - The agent knows WHERE to go but not HOW to fight effectively.

---

## Key Observations

### What Works Well
1. **Spatial awareness**: Agent understands spawner locations and navigates directly toward them
2. **Obstacle avoidance**: Sometimes dodges enemies en route to objectives
3. **Rushdown priority**: Will quickly dispatch spawners 
4. **Phase understanding**: Knows to move between spawners after destruction

### What Needs Improvement
1. **Enemy engagement**: Does not see enemies as threats nor targets for elimintation
2. **Resource efficiency**: Doesn't leverage projectile range (PROJECTILE_LIFETIME = 120 frames)
3. **Combat confidence**: Too aggressive, doesn't take calculated risks

---

## Training Configuration

### DQN Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 1e-4 → 2e-5 | Linear decay to 2e-5 at 3M steps |
| Buffer Size | 100,000 | Replay buffer capacity |
| Learning Starts | 10,000 | Steps before training begins |
| Batch Size | 128 | Gradient update batch size |
| Tau | 0.005 | Target network soft update |
| Gamma (γ) | 0.99 | Discount factor |
| Train Frequency | 4 steps | Updates per 4 environment steps |
| Gradient Steps | 1 | Per training call |
| Target Update Interval | 1000 | Steps between target net updates |
| Exploration Fraction | 0.3 | 30% of training for ε decay |
| Initial Epsilon (ε) | 1.0 | Full random exploration |
| Final Epsilon (ε) | 0.05 | 5% random actions |
| Max Gradient Norm | 10.0 | Gradient clipping |
| Network Architecture | [256, 256, 128] | 3-layer MLP |
| Device | auto | CPU/CUDA auto-select |

### Environment Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Max Episode Steps | 5,000 | Timeout limit |
| Screen Size | 1000x700 | Play area dimensions |
| FPS | 60 | Simulation speed |
| Initial Spawners | 2 | Phase 1 starting count |
| Spawners Per Phase | +1 | Difficulty scaling |
| Max Phase | 5 | Win condition |

### Player Stats

| Attribute | Value | Notes |
|-----------|-------|-------|
| Health | 100 HP | Starting health |
| Speed | 5.0 | Directional movement speed |
| Radius | 20 | Collision detection |
| Shoot Cooldown | 10 frames | 0.167 seconds @ 60 FPS |
| Invincibility | 30 frames | Post-damage immunity |

### Projectile Stats

| Attribute | Value | Notes |
|-----------|-------|-------|
| Speed | 10.0 | 2x player speed |
| Radius | 5 | Collision detection |
| Lifetime | 120 frames | **2 seconds @ 60 FPS** |
| Damage | 10 | Per hit |

**Key Insight:** Projectiles last 2 full seconds - the agent should be shooting from distance but doesn't exploit this!

### Enemy Stats

| Attribute | Value | Notes |
|-----------|-------|-------|
| Health | 30 HP | Base health (scales with phase) |
| Speed | 2.5 | 50% of player speed |
| Radius | 15 | Smaller than player |
| Damage | 10 | Per collision |

### Spawner Stats

| Attribute | Value | Notes |
|-----------|-------|-------|
| Health | 100 HP | Primary objective |
| Radius | 42 | Large hitbox |
| Spawn Interval | 180 frames | 3 seconds @ 60 FPS |
| Max Enemies | 5 | Per spawner |

---

## Reward Function (DQN Directional)

### Core Combat Rewards

| Event | Reward | Trigger Condition |
|-------|--------|-------------------|
| `destroy_enemy` | **+170** | Enemy health reaches 0 |
| `destroy_spawner` | **+180** | Spawner health reaches 0 |
| `phase_complete` | **+500** | All spawners destroyed |
| `clean_phase_bonus` | **+300** | No enemies alive at phase end |
| `hit_enemy` | **+30** | Projectile damages enemy |
| `hit_spawner` | **5** | Projectile damages spawner |

### Penalties

| Event | Penalty | Trigger Condition |
|-------|---------|-------------------|
| `take_damage` | **-80** | Player collides with enemy |
| `death` | **-200** | Player health reaches 0 |
| `shot_fired` | **-0.05** | Player shoots (spam prevention) |
| `existence_penalty` | **-0.015** | Every step (time pressure) |

### Navigation Rewards (Potential-Based Shaping)

| Event | Reward | Calculation |
|-------|--------|-------------|
| `potential_closer` | **+0.1** | Distance to nearest spawner decreases |
| `potential_further` | **0** | Distance to nearest spawner increases (no penalty) |
| `orientation_closer` | **+0.1** | Movement direction aligns with spawner |
| `orientation_further` | **0** | Movement direction opposes spawner (no penalty) |

### Accuracy Bonus

| Event | Reward | Condition |
|-------|--------|-----------|
| `accuracy_bonus` | **+0.2** | Shooting within 30° of enemy/spawner |

---

## Observation Space (18-dimensional vector)

### Player State (8 features)
1. **x position** - Normalized [0, 1]
2. **y position** - Normalized [0, 1]
3. **vx velocity** - Normalized [-1, 1]
4. **vy velocity** - Normalized [-1, 1]
5. **velocity magnitude** - Normalized [0, 1]
6. **wall proximity** - Minimum distance to edge [0, 1]
7. **angle** - Movement direction normalized [0, 1]
8. **health** - Normalized [0, 1]

### Nearest Enemy (3 features)
9. **distance** - Normalized by screen diagonal
10. **cos(angle)** - Direction to enemy [-1, 1]
11. **sin(angle)** - Direction to enemy [-1, 1]

### Nearest Spawner (3 features)
12. **distance** - Normalized by screen diagonal
13. **cos(angle)** - Direction to spawner [-1, 1]
14. **sin(angle)** - Direction to spawner [-1, 1]

### Game State (4 features)
15. **phase** - Current phase normalized [0, 1]
16. **enemies_count** - Normalized [0, 1]
17. **spawners_count** - Normalized [0, 1]
18. **can_shoot** - Boolean {0, 1}

---

## Action Space (6 discrete actions)

| Action ID | Action | Effect |
|-----------|--------|--------|
| 0 | `NOTHING` | No action (idle) |
| 1 | `MOVE_UP` | +y direction at speed 5.0 |
| 2 | `MOVE_DOWN` | -y direction at speed 5.0 |
| 3 | `MOVE_LEFT` | -x direction at speed 5.0 |
| 4 | `MOVE_RIGHT` | +x direction at speed 5.0 |
| 5 | `SHOOT` | Fire projectile (respects cooldown) |

---

## Bellman Equation & Gradient Computation

### Q-Learning Update Rule
```
Q(s, a) ← Q(s, a) + α [r + γ · max Q(s', a') - Q(s, a)]
                              a'
```

Where:
- **Q(s, a)**: Action-value for state `s`, action `a`
- **α**: Learning rate (1e-4 decaying to 2e-5)
- **r**: Immediate reward
- **γ**: Discount factor (0.99)
- **s'**: Next state
- **max Q(s', a')**: Maximum Q-value over all actions in next state

### Loss Function (Huber Loss)
```
L(θ) = 𝔼[(r + γ · max Q_target(s', a'; θ⁻) - Q(s, a; θ))²]
                    a'
```

Where:
- **θ**: Current network parameters
- **θ⁻**: Target network parameters (updated every 1000 steps with τ=0.005)
- **Q_target**: Target network (stabilizes training)

### Gradient Clipping
```
∇θ L(θ) → clip(∇θ L(θ), -10.0, +10.0)
```

Prevents exploding gradients by capping at ±10.0.

### Target Network Update (Soft Update)
```
θ⁻ ← τ · θ + (1 - τ) · θ⁻
```

With τ = 0.005, this slowly tracks the main network to reduce instability.

---

## Why "Melee Combat" Behavior Emerged

### Hypothesis: Reward Timing Mismatch

1. **Immediate Penalties Dominate:**
   - `shot_fired: -0.05` → Agent pays IMMEDIATELY when shooting
   - `existence_penalty: -0.015` → Constant pressure every frame
   - `take_damage: -80` → Immediate punishment for enemy contact

2. **Delayed Spawner Rewards:**
   - Agent shoots → 10-20 frames later → projectile hits spawner → `+250` reward
   - With γ=0.99, the shooting action only receives: **250 × 0.99^20 ≈ +204**
   - This is credited, but the temporal gap reduces perceived value

3. **Spatial Credit Assignment:**
   - Agent learns: "Being CLOSE to spawner = good" (navigation rewards)
   - Agent doesn't learn: "Shooting WHILE approaching = better" (too complex)
   - Simplest strategy: Move to spawner first, then shoot (two-phase behavior)

4. **Exploration Insufficient:**
   - At 300k steps, epsilon has decayed significantly
   - Early exploration didn't discover ranged combat advantage
   - Policy solidified around "safe melee" strategy before finding optimal "aggressive ranged"
