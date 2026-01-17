import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple


def read_log(path: Path | str) -> Tuple[List[int], List[float], List[int]]:
    episodes, env_returns, steps = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            env_returns.append(float(row["env_return"]))
            steps.append(int(row.get("steps", 0)))
    return episodes, env_returns, steps


def rolling_mean(values: List[float], window: int) -> List[float]:
    if window <= 1 or not values:
        return values[:]
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        denom = min(i + 1, window)
        out.append(acc / denom)
    return out


def main():
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Plot Level 0 (Q) vs Level 1 (SARSA) with hazards")
    parser.add_argument("--level0", default=base_dir / "logs/level0_q.csv", help="CSV for Level 0 (Q-learning)")
    parser.add_argument("--level1", default=base_dir / "logs/level1_sarsa.csv", help="CSV for Level 1 (SARSA)")
    parser.add_argument("--window", type=int, default=20, help="Rolling mean window (episodes)")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write("matplotlib is required for plotting. Install via `python -m pip install matplotlib`.\n")
        sys.exit(1)

    l0_ep, l0_ret, l0_steps = read_log(args.level0)
    l1_ep, l1_ret, l1_steps = read_log(args.level1)

    n = min(len(l0_ep), len(l1_ep))
    l0_ep, l0_ret, l0_steps = l0_ep[:n], l0_ret[:n], l0_steps[:n]
    l1_ep, l1_ret, l1_steps = l1_ep[:n], l1_ret[:n], l1_steps[:n]

    l0_ret_s = rolling_mean(l0_ret, args.window)
    l1_ret_s = rolling_mean(l1_ret, args.window)
    l0_steps_s = rolling_mean(l0_steps, args.window) if any(l0_steps) else None
    l1_steps_s = rolling_mean(l1_steps, args.window) if any(l1_steps) else None

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(l0_ep, l0_ret_s, label="Level 0 (Q-learning, no hazards)")
    axes[0].plot(l1_ep, l1_ret_s, label="Level 1 (SARSA, hazards)")
    axes[0].set_ylabel("Env return")
    axes[0].set_title("Level 0 vs Level 1: SARSA conservativeness around hazards")
    axes[0].legend()

    if l0_steps_s is not None and l1_steps_s is not None:
        axes[1].plot(l0_ep, l0_steps_s, label="Level 0 steps")
        axes[1].plot(l1_ep, l1_steps_s, label="Level 1 steps")
        axes[1].set_ylabel("Steps")
        axes[1].legend()
    axes[1].set_xlabel("Episode")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
