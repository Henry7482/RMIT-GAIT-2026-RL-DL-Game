# Suicidal PPO v1.1 - Checkpoint

**Timestamp:** January 9, 2026 - 03:47  
**Model File:** `suicidal_ppo_1.1_20260109_0347.zip`  
**Metric** `PPO_26`

## Status Summary

⚠️ **Still suicidal behavior present, but MUCH less aggressive than previous versions**

The agent has learned to balance risk-taking with survival, showing improved judgment about when to engage enemies and spawners. While it still exhibits some aggressive tendencies, it's significantly more cautious compared to earlier iterations.

---

## PPO Hyperparameters Evolution

### Core Training Settings
- **Learning Rate:** `1e-4` (reduced from `3e-4` to stop KL fluctuations)
- **N Steps:** `4096` (per environment, 16,384 total with 4 parallel envs)
- **Batch Size:** `256` (increased for parallel data)
- **N Epochs:** `10`
- **Gamma:** `0.99`
- **GAE Lambda:** `0.95`
- **Clip Range:** `0.2`

### Entropy Coefficient Changes (Exploration Control)
The `ent_coef` parameter underwent significant experimentation to balance exploration vs exploitation:

1. **Initial:** `0.01` → **Changed to** `0.05` to escape local minimum of hiding in corner
2. **9:48 9/1/2026:** Increased to `0.1` for opposite spawner exploration (agent struggled to find spawners after clearing one)
3. **10:18 9/1/2026:** Decreased back to `0.025`
4. **17:12 9/1/2026:** Increased to `0.08` - agent failed to explore spawners right behind
5. **Final:** Decreased back to `0.025` (agent tried "not to hit enemies" with 0.08)

**Current Value:** `0.025`

### Network Architecture
- **Policy Network:** `[256, 256]`
- **Value Network:** `[256, 256]`
- **Vectorized Environments:** `4` parallel

---

## Reward Function Evolution

### Positive Rewards (Objectives)

#### Destroy Enemy
- **17:30 8/1:** `50.0` (increased from `20.0`)
- **16:46 9/1/2026:** Decreased to `50.0` to balance with destroy spawner
- **16:46 9/1/2026:** Increased to `60.0` to encourage killing enemies first
- **17:12 9/1/2026:** Increased to `80.0` to make killing enemies even more valuable
- **Current:** `80.0`

#### Destroy Spawner
- **Current:** `200.0` (increased from `100.0`)

#### Phase Complete
- **Current:** `500.0` (increased from `200.0`)

#### Hit Enemy
- **17:30 8/1:** `5.0`
- **Current:** `10.0` (increased from `5.0`)

#### Accuracy Bonus
- **Current:** `0.2` (reward for shooting when aimed at target within 20° threshold)

### Negative Rewards (Penalties)

#### Take Damage - Key Anti-Suicidal Parameter
- **17:30 8/1:** `-10.0` (decreased from `-5.0` to make getting hit less scary)
- **Current iteration:** Increased to `-40.0` to make damage MORE scary
- **16:46 9/1/2026:** Decreased to `-35.0` to encourage killing enemies first
- **17:12 9/1/2026:** Increased to `-50.0` to make damage even more scary
- **Final:** Decreased back to `-40.0` (agent got too conservative with `-50.0`)
- **Current:** `-40.0`

#### Death Penalty - Primary Anti-Suicidal Control
- **17:30 8/1:** `-50.0` (decreased from `-100.0` to make death MUCH less scary)
- **Current iteration:** Increased to `-200.0` to make death MUCH MORE scary
- **17:12 9/1/2026:** Increased to `-300.0` to be even more aggressive about losing health
- **Final:** Decreased back to `-200.0` (agent got stuck in corner with `-300.0`)
- **Current:** `-200.0`

#### Other Penalties
- **Shot Fired:** `-0.05` (reduced from `-0.1` to allow more shooting)
- **Existence Penalty:** `-0.015` (small penalty per step to discourage hiding)
- **Potential Further:** `-0.12` (penalty for moving away from objectives)
- **Orientation Further:** `-0.12` (penalty for rotating away from spawners)

### Shaping Rewards
- **Potential Closer:** `0.1` (increased 10x from `0.01` for stronger navigation signal)
- **Orientation Closer:** `0.1` (reward for rotating towards nearest spawner)
- **Survival Bonus:** `0.0` (disabled to prevent hiding/farming)
- **Near Spawner:** `-0.1` (deprecated, was meant to prevent suicidal rushes)

---

## Key Observations

### What Worked
✅ Increasing death penalty from `-50` to `-200` significantly reduced suicidal rushes  
✅ Balancing damage penalty at `-40` (sweet spot between `-35` and `-50`)  
✅ Higher enemy kill reward (`80.0`) encourages clearing enemies before spawners  
✅ Entropy coefficient at `0.025` provides good exploration without erratic behavior

### What Didn't Work
❌ Death penalty of `-300` was too harsh - agent became overly defensive  
❌ Entropy coefficient of `0.08` - agent avoided hitting enemies  
❌ Damage penalty of `-50` - agent got stuck in corners

### Current Behavior
- Agent is **still somewhat aggressive** but shows improved self-preservation
- **Better enemy prioritization** - clears enemies before rushing spawners
- **Less corner camping** compared to overly conservative versions
- **More tactical engagement** - picks fights it can win

---

## Next Steps for Improvement

1. **Fine-tune damage penalty** - Test range between `-40` and `-45`
2. **Experiment with survival bonus** - Small positive value like `0.002` might help
3. **Consider dynamic penalties** - Scale death penalty with health remaining
4. **Test longer training** - Current behavior might stabilize with more timesteps

---

## Training Details

- **Environment:** Rotation Arena (physics-based movement)
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Total Timesteps:** Variable (incremental training sessions)
- **Parallel Environments:** 4
- **Observation Space:** 16 dimensions (player state, enemies, spawners, game state)
- **Action Space:** 5 actions (nothing, thrust, rotate_left, rotate_right, shoot)

---

*Generated on January 9, 2026 at 03:47*
