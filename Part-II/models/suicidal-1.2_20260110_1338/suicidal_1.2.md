# Suicidal PPO v1.2 - Checkpoint

**Timestamp:** January 10, 2026 - 13:38  
**Model File:** `suicidal-1.2_20260110_1338.zip`  
**Previous Version:** `suicidal_ppo_1.1_20260109_0347`

## Status Summary

⚠️ **Still exhibiting suicidal behavior, but with some improvements**

While the agent shows reduced suicidal tendencies compared to v1.1, it has not fully overcome this behavior. Performance is inconsistent with notable variation across episodes.

### Performance Statistics (Observed)
- **Reaches Phase 4:** ~70% of the time
- **Gets stuck in one direction:** ~10% of the time
- **Suicidal behavior:** ~20% of the time
- **First time reaching Phase 5:** Yes, but not consistently

### Achievements
- **Occasional Phase 5 success** - Can reach Phase 5 in best-case scenarios
- **Reduced suicidal tendency** - Down from previous versions, but still present in 20% of runs
- **Stays in middle** - Good positioning (though may need tuning for optimal movement)
- **Almost no corner sticking** - Navigation issues largely resolved

### Current Limitations
- **Inconsistent performance** - High variation between episodes
- **Persistent suicidal behavior** - Still occurs in ~20% of episodes
- **Direction lock** - Gets stuck moving in one direction ~10% of the time
- **Poor enemy dodging** - Agent doesn't evade incoming enemies effectively
- **Over-prioritizes spawners** - Will chase spawners even when enemies are nearly dead
  - Example: If a spawner is destroyed but one enemy remains with only 1 HP, the agent ignores that enemy and rushes to the next spawner
  - This causes unnecessary health loss from easily killable enemies

---

## Key Changes from v1.1 (suicidal_ppo_1.1_20260109_0347)

### 1. Learning Rate Scheduling ⭐ NEW FEATURE
**Previous:** Constant learning rate of `1e-4` throughout training  
**Current:** Linear learning rate decay from `1e-4` to `0` over training duration

```python
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

learning_rate=linear_schedule(1e-4)  # Was: learning_rate=1e-4
```

**Rationale:** 
- Learning rate scheduling helps stabilize training in later stages
- Linear decay allows aggressive learning early, fine-tuning later
- Commonly used best practice in deep RL to improve convergence
- Addresses user concern about constant learning rate
- Adam optimizer's adaptive per-parameter rates work on top of this base schedule

### 2. Entropy Coefficient Adjustment
**Previous:** `ent_coef=0.025`  
**Current:** `ent_coef=0.05`

**Rationale:**
- Increased from `0.025` to `0.05` to encourage more exploration
- Previous value caused agent to get stuck with opposite spawners
- Agent needs more exploration to discover better enemy engagement strategies
- Timestamp: 13:20 10/1/2026

### 3. Reward Function Refinements

#### Destroy Enemy (increased)
**Previous:** `80.0`  
**Current:** `85.0`

**Rationale:**
- Incremental increase to further prioritize enemy kills
- Attempting to teach agent to finish off weak enemies before moving on
- Timestamp: 18:35 9/1/2026

#### Destroy Spawner (decreased)
**Previous:** `200.0`  
**Current:** `180.0`

**Rationale:**
- Reduced from `200.0` to `180.0` to balance with enemy kill rewards
- Reduces over-prioritization of spawners
- Encourages agent to clear enemies before rushing next spawner
- Timestamp: 18:35 9/1/2026

#### Take Damage (more negative)
**Previous:** `-40.0`  
**Current:** `-45.0`

**Rationale:**
- Increased penalty from `-40.0` to `-45.0`
- Further discourages risky behavior and health loss
- Aims to make agent prioritize killing enemies to avoid damage
- Timestamp: 13:20 10/1/2026

---

## Complete PPO Hyperparameters

### Core Training Settings
- **Learning Rate:** `linear_schedule(1e-4)` ⭐ **CHANGED** - was constant `1e-4`
- **N Steps:** `4096` (per environment, 16,384 total with 4 parallel envs)
- **Batch Size:** `256`
- **N Epochs:** `10`
- **Gamma:** `0.99`
- **GAE Lambda:** `0.95`
- **Clip Range:** `0.2`
- **Clip Range VF:** `None`
- **VF Coef:** `0.5`
- **Max Grad Norm:** `0.5`
- **Target KL:** `None`

### Exploration Settings
- **Entropy Coefficient:** `0.05` ⭐ **CHANGED** - was `0.025`
- **Use SDE:** `False`
- **SDE Sample Freq:** `-1`

### Network Architecture
- **Policy Network:** `[256, 256]`
- **Value Network:** `[256, 256]`
- **Vectorized Environments:** `4` parallel

---

## Complete Reward Function

### Positive Rewards (Objectives)

| Reward Type | Value | Notes |
|-------------|-------|-------|
| **Destroy Enemy** | `85.0` | ⭐ Increased from `80.0` |
| **Destroy Spawner** | `180.0` | ⭐ Decreased from `200.0` |
| **Phase Complete** | `500.0` | Unchanged |
| **Hit Enemy** | `10.0` | Unchanged |
| **Accuracy Bonus** | `0.2` | Shooting when aimed at target (within 20°) |

### Negative Rewards (Penalties)

| Penalty Type | Value | Notes |
|--------------|-------|-------|
| **Take Damage** | `-45.0` | ⭐ Increased from `-40.0` (more scary) |
| **Death** | `-200.0` | Unchanged - maintains anti-suicidal stance |
| **Shot Fired** | `-0.05` | Unchanged |
| **Existence Penalty** | `-0.015` | Per step, discourages passivity |
| **Potential Further** | `-0.12` | Moving away from spawners |
| **Orientation Further** | `-0.12` | Rotating away from spawners |

### Shaping Rewards

| Reward Type | Value | Purpose |
|-------------|-------|---------|
| **Potential Closer** | `0.1` | Approaching spawners |
| **Orientation Closer** | `0.1` | Rotating towards spawners |
| **Survival Bonus** | `0.0` | Disabled to prevent hiding/farming |

---

## Entropy Coefficient Evolution Timeline

| Date/Time | Value | Reason |
|-----------|-------|--------|
| Initial | `0.01` | → Changed to `0.05` to escape corner hiding |
| 9:48 9/1 | `0.1` | Explore opposite spawners |
| 10:18 9/1 | `0.025` | Reduced back down |
| 17:12 9/1 | `0.08` | Agent failed to explore behind |
| After | `0.025` | Agent tried "not to hit enemies" with 0.08 |
| 13:20 10/1 | `0.05` | ⭐ **CURRENT** - Agent stuck with opposite spawners |

---

## Next Steps & Recommendations

Based on the current performance and limitations:

### High Priority
1. **Improve enemy dodging behavior**
   - Consider adding reward for maintaining distance from enemies
   - Add penalty for being too close to enemies without shooting
   - Potential new shaping reward: `approach_enemy_penalty`

2. **Fix spawner over-prioritization**
   - Continue adjusting `destroy_enemy` vs `destroy_spawner` balance
   - Consider adding "cleanup bonus" for killing all remaining enemies before next spawner
   - Possible new reward: `enemy_count_penalty` (penalty for having many active enemies)

3. **Learning rate monitoring**
   - Monitor TensorBoard `train/learning_rate` to verify decay
   - Experiment with different schedules (exponential, cosine annealing)
   - Consider using `learning_rate=3e-4` as starting point instead of `1e-4`

### Medium Priority
4. **Middle positioning behavior**
   - Evaluate if staying in middle is optimal or too passive
   - May need rewards for strategic positioning based on spawner locations
   - Consider adding "safe zone" rewards near center with diminishing returns

5. **Entropy coefficient fine-tuning**
   - Current `0.05` seems to work well
   - Monitor exploration vs exploitation balance
   - May need adjustment as training progresses

---

## Training Environment
- **Environment Type:** `rotation`
- **Algorithm:** `PPO`
- **Parallel Environments:** `4`
- **Device:** `auto`
- **TensorBoard Logs:** `./logs/rotation_ppo/`

---

## Change Summary

### Code Changes
1. ✅ Added linear learning rate schedule function
2. ✅ Changed `learning_rate` parameter to use scheduler instead of constant
3. ✅ Fixed `--lr` argument type from `int` to `float`
4. ✅ Updated `--lr` help text to mention scheduling

### Hyperparameter Changes
1. ✅ `learning_rate`: Constant `1e-4` → Linear schedule from `1e-4` to `0`
2. ✅ `ent_coef`: `0.025` → `0.05`

### Reward Function Changes
1. ✅ `destroy_enemy`: `80.0` → `85.0`
2. ✅ `destroy_spawner`: `200.0` → `180.0`
3. ✅ `take_damage`: `-40.0` → `-45.0`

---

## Version History

- **v1.1** (2026-01-09 03:47): Suicidal PPO - Aggressive behavior, constant learning rate
- **v1.2** (2026-01-10 13:38): Suicidal PPO - Reduced but still present (20%), learning rate scheduling, reaches Phase 4 consistently (70%) ⭐ **CURRENT**
