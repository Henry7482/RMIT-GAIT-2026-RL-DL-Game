# GAIT-2026-RL-DL-Game

Deep Reinforcement Learning project for RMIT GAIT 2026 Assessment.

## Installation

```bash
# Part-I (Q-learning/SARSA GridWorld)
pip install pygame numpy matplotlib

# Part-II (Deep RL Arena)
cd Part-II
pip install -r requirements.txt
```

## Part-I: GridWorld Q-Learning/SARSA

### Run Game
```bash
cd Part-I
python game.py
```

Controls: `S` start, `T` pause, `R` reset, `0-6` switch levels, arrow keys navigate

### Plot Results
```bash
cd Part-I
python plot_levels01.py      # Level 0 vs Level 1
python plot_level_4_5.py     # Level 4 vs Level 5 (monsters)
python plot_graph.py         # General plots
```

## Part-II: Deep RL Arena

### Train Models

**Rotation PPO (Asteroids-style controls):**
```bash
cd Part-II
python train.py --env rotation --algo ppo --timesteps 5000000 --lr 1e-4
```

**Directional DQN (WASD-style controls):**
```bash
cd Part-II
python train.py --env directional --algo dqn --timesteps 5000000 --lr 1e-4
```

### Evaluate Models

**Best Directional DQN Model (Pathfinder v1.3):**
```bash
cd Part-II
python evaluate.py --model models/pathfinder-1.3/model.zip --env directional --episodes 5
```

**Best Rotation PPO Model (Sharpshooter v1.0):**
```bash
cd Part-II
python evaluate.py --model models/sharpshooter-1.0_20260112_2224/sharpshooter-1.0_20260112_2224.zip --env rotation --episodes 5
```

**Other Non-Suicidal Rotation PPO Models:**
```bash
cd Part-II

# Version 1.3
python evaluate.py --model models/non-suicidal-1.3_20260112_1645/non-suicidal-1.3_20260112_1645.zip --env rotation --episodes 5
```

**Note:** Suicidal and Non-suicidal from 1.0 to 1.2 models cannot be evaluated due to observation space changes in the current codebase.

**Evaluate with random actions (baseline):**
```bash
cd Part-II
python evaluate.py --env rotation --episodes 5 --random
python evaluate.py --env directional --episodes 5 --random
```

### Play Manually

**Rotation controls (default):**
```bash
cd Part-II
python play_human.py
```

**Directional controls:**
```bash
cd Part-II
python play_human.py --directional
```

Controls: `W/UP` thrust/move, `A/D/LEFT/RIGHT` rotate/move, `SPACE` shoot, `R` reset, `TAB` switch mode

### TensorBoard

**View Rotation PPO training metrics:**
```bash
cd Part-II
tensorboard --logdir logs/rotation_ppo
```

**View Directional DQN training metrics:**
```bash
cd Part-II
tensorboard --logdir logs/directional_dqn
```

**View all training metrics:**
```bash
cd Part-II
tensorboard --logdir logs/
```

Open http://localhost:6006 to view training metrics.

## Project Structure

```
Part-I/               # GridWorld Q-learning/SARSA
  game.py             # Main game + training loop
  levels.py           # Level definitions
  logs/               # Training logs (CSV)

Part-II/              # Deep RL Arena
  train.py            # PPO training script
  train_dqn.py        # DQN training script
  evaluate.py         # Model evaluation script
  play_human.py       # Human playable version
  envs/               # Gymnasium environments
  game/               # Game engine + entities
  models/             # Trained models (.zip)
  logs/               # TensorBoard logs
```
