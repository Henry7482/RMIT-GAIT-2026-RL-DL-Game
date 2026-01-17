# Games & AI Assignment 3 (GAIT)

GridWorld reinforcement learning demo with Q-learning and SARSA, multiple levels, and optional intrinsic reward.

# Files

- `Part-I/game.py` main pygame app: environment, training loop, controls, logging to CSV.
- `Part-I/levels.py` level layouts and metadata (algorithm, monsters, intrinsic reward).
- `Part-I/config.py` default hyperparameters plus per-level overrides via `config_level{n}.json`.
- `Part-I/plot_levels01.py` plots Level 0 vs Level 1 learning curves from CSV logs.
- `Part-I/plot_monsters.py` plots Level 4 vs Level 5 learning curves from CSV logs.
- `Part-I/plot_level6.py` plots Level 6 intrinsic reward on vs off from CSV logs.
- `Part-I/logs/*.csv` training logs written by `game.py`.
- `**/__pycache__/` Python bytecode caches (auto-generated).

# How to run

1) Run the main demo:
```bash
python Part-I/game.py
```

2) Plot example results (optional):
```bash
python Part-I/plot_levels01.py
python Part-I/plot_monsters.py
python Part-I/plot_level6.py
```

# Controls (in the game window)

- `S` start/resume, `T` pause, `R` reset Q-table, `V` toggle speed.
- `0-6` jump to a level, arrow keys cycle levels.
- `I` toggle intrinsic reward on Level 6.
