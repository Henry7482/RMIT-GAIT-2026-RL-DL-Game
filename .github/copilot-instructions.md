# copilot-instuctions.md

This file provides guidance to Copilot when working with code in this repository.

## Project Overview

Deep Reinforcement Learning arena game for RMIT GAIT 2026 Assessment Part II. Agents learn to destroy spawners while surviving enemies in a Pygame-based environment using Stable Baselines3 with Gymnasium.

## Commands

All commands run from `Part-II/` directory:

```bash
# Install dependencies
pip install -r requirements.txt

# Train agent (rotation control - Asteroids-style)
python train.py --env rotation --algo ppo --timesteps 5000000 --lr 1e-4

# Train agent (directional control - WASD-style)
python train.py --env directional --algo ppo --timesteps 5000000 --lr 1e-4

# Evaluate trained model
python evaluate.py --model models/rotation_ppo_test.zip --env rotation --episodes 5

# Play manually (rotation controls, or add --directional)
python play_human.py

# View training metrics
tensorboard --logdir logs/
```

## Architecture

### Environment Hierarchy
```
envs/base_env.py      → BaseArenaEnv (abstract Gymnasium environment)
  ├── envs/rotation_env.py    → RotationArenaEnv (5 actions: nothing, thrust, rotate_left, rotate_right, shoot)
  └── envs/directional_env.py → DirectionalArenaEnv (6 actions: nothing, up, down, left, right, shoot)
```

### Game Engine
- `game/arena.py` - Main Arena class managing game state, physics, collisions, reward calculation
- `game/entities.py` - Player, Enemy, Spawner, Projectile, Particle classes
- `game/constants.py` - All configuration: screen size, entity parameters, reward values, colors

### Observation Vector (14-dimensional)
Normalized feature vector including: player position/velocity/angle/health, nearest enemy distance/angle, nearest spawner distance/angle, phase, entity counts, can_shoot flag.

### Key Configuration Locations
- **Rewards**: `game/constants.py` REWARDS dict (destroy_enemy: +50, destroy_spawner: +200, phase_complete: +500, take_damage: -10)
- **Training hyperparameters**: `train.py` (learning rate schedule, entropy coefficient, network architecture)
- **Game mechanics**: `game/constants.py` (player/enemy/spawner stats, phase progression)

## Development Context

The reward structure has undergone significant tuning to prevent "suicidal mode" (agent rushing into enemies). Key solutions:
- High spawner destruction reward (200) to encourage objective focus
- Damage penalty (-10) to discourage reckless behavior
- Entropy coefficient (0.065) to prevent corner-hiding
- Learning rate schedule decaying to 2e-5 at 3M timesteps

Model variants in `models/` directory track this evolution (suicidal-* vs non-suicidal-* naming).

## Model Checkpointing System

### Directory Structure

Models are organized in versioned checkpoint folders within `models/`:

```
models/
├── {behavior}-{version}_{date}_{time}/
│   ├── {behavior}-{version}_{date}_{time}.zip    # The trained model
│   ├── README.md                                   # Comprehensive checkpoint documentation
│   └── [screenshots]                               # Optional performance screenshots
├── rotation_ppo_test.zip                          # Latest test model (overwritten each run)
└── .gitkeep
```

**Naming Convention:**
- Behavior: `suicidal` (20%+ aggressive deaths) or `non-suicidal` (0% aggressive deaths)
- Version: Major.Minor (e.g., `1.0`, `1.1`, `1.2`)
- Date: `YYYYMMDD` format (e.g., `20260111`)
- Time: `HHMM` 24-hour format (e.g., `1612`)

**Examples:**
- `non-suicidal-1.1_20260111_1612/` - Non-suicidal version 1.1, saved Jan 11 2026 at 4:12 PM
- `suicidal-1.2_20260110_1338/` - Suicidal version 1.2, saved Jan 10 2026 at 1:38 PM

### Creating a Checkpoint

When saving a new checkpoint (after significant training or hyperparameter changes):

1. **Save the model with timestamp:**
   ```bash
   # After training completes, organize the checkpoint
   mkdir -p models/non-suicidal-1.2_$(date +%Y%m%d)_$(date +%H%M)
   mv models/rotation_ppo_test.zip models/non-suicidal-1.2_$(date +%Y%m%d)_$(date +%H%M)/
   ```

2. **Create comprehensive README.md** in the checkpoint folder with:
   - **Status Summary** - Performance overview, achievements, limitations
   - **Key Changes** - What changed from previous version with rationale and timestamps
   - **Complete Training Configuration** - All PPO hyperparameters in table format
   - **Complete Reward Function** - All rewards/penalties with values and trigger conditions
   - **Game Environment Configuration** - Episode settings, player/enemy/spawner parameters
   - **Observation & Action Spaces** - Full 16-dimensional observation breakdown, 5 discrete actions
   - **Gradient Computation Details** - Loss function, GAE, gradient clipping formulas
   - **Evolution Timeline** - Track how each parameter changed over time with dates/reasons
   - **Goals for Next Version** - What to try next, experiments to run
   - **Performance Summary** - Metrics comparison table vs previous version
   - **Version History** - Chronological list of all versions with key characteristics

3. **Optional: Add screenshots** - Capture TensorBoard metrics or gameplay to document performance

### README Template Structure

See existing checkpoints for comprehensive examples:
- `models/non-suicidal-1.1_20260111_1612/README.md` - Latest non-suicidal (OPTIMAL learning rate schedule)
- `models/non-suicidal-1.0_20260110_1553/README.md` - First non-suicidal breakthrough
- `models/suicidal-1.2_20260110_1338/README.md` - Suicidal with learning rate scheduling

**Key sections to include:**
```markdown
# {Behavior} PPO v{Version} - Checkpoint

**Timestamp:** Month DD, YYYY - HH:MM
**Model File:** `{filename}.zip`
**Previous Version:** `{previous-folder-name}`

## Status Summary
[Performance overview, achievements, current limitations]

## Key Changes from v{Previous}
[What changed, why it changed, when it changed]

## Complete Training Configuration
[Full hyperparameter table]

## Complete Reward Function
[All rewards and penalties with values]

## Evolution Timeline
[Track parameter changes over time]

## Goals for Next Model
[What to try next]

## Version History
[All versions chronologically]
```

### Checkpoint Best Practices

1. **When to create checkpoints:**
   - After discovering non-suicidal behavior (major breakthrough)
   - When changing key hyperparameters (learning rate, entropy coefficient)
   - After modifying reward structure
   - Before/after extended training runs (>3M timesteps)
   - When discovering optimal configurations (e.g., learning rate schedule)

2. **Version numbering:**
   - Increment major version (1.0 → 2.0) for behavior changes or major reward restructuring
   - Increment minor version (1.0 → 1.1) for hyperparameter tuning or training extensions
   - Keep separate version tracks for `suicidal` vs `non-suicidal` behaviors

3. **Documentation discipline:**
   - Always timestamp changes with reason in "Evolution Timeline"
   - Compare metrics with previous version in "Performance Summary"
   - Document failed experiments too (e.g., "ent_coef=0.08 made agent avoid enemies")
   - Include TensorBoard run IDs (e.g., `PPO_45`) for metric tracking

4. **Key discoveries to document:**
   - Learning rate schedule: Linear decay to 2e-5 at 3M, then constant (enables 10M+ training)
   - Entropy coefficient sweet spot: 0.065 (balances exploration without chaos)
   - Reward balance: destroy_enemy=85, destroy_spawner=180, take_damage=-45, death=-200

### Training Code Integration

The current `train.py` saves to `models/rotation_ppo_test.zip` which gets overwritten each run. To create a checkpoint:

```python
# At end of train.py, save with version info
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
checkpoint_name = f"non-suicidal-1.2_{timestamp}"
checkpoint_dir = f"./models/{checkpoint_name}"
os.makedirs(checkpoint_dir, exist_ok=True)
model.save(f"{checkpoint_dir}/{checkpoint_name}")
print(f"\n✓ Checkpoint saved to {checkpoint_dir}/")
```

## Reference Documentation

- `Part-II/ALGORITHM_REFERENCE.md` - Full technical specifications, observation vector details, reward breakdown
- `Part-II/assignement.txt` - Assessment requirements (20 marks rubric)
