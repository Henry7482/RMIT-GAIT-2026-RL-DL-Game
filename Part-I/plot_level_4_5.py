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
    parser = argparse.ArgumentParser(description="Plot monster levels (4: Q-learning, 5: SARSA)")
    parser.add_argument("--level4", default=base_dir / "logs/level4_q.csv", help="CSV for Level 4 (Q-learning)")
    parser.add_argument("--level5", default=base_dir / "logs/level5_sarsa.csv", help="CSV for Level 5 (SARSA)")
    parser.add_argument("--window", type=int, default=20, help="Rolling mean window (episodes)")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write("matplotlib is required for plotting. Install via `python -m pip install matplotlib`.\n")
        sys.exit(1)

    l4_ep, l4_ret, l4_steps = read_log(args.level4)
    l5_ep, l5_ret, l5_steps = read_log(args.level5)

    # Align lengths if they differ
    n = min(len(l4_ep), len(l5_ep))
    l4_ep, l4_ret, l4_steps = l4_ep[:n], l4_ret[:n], l4_steps[:n]
    l5_ep, l5_ret, l5_steps = l5_ep[:n], l5_ret[:n], l5_steps[:n]

    l4_ret_s = rolling_mean(l4_ret, args.window)
    l5_ret_s = rolling_mean(l5_ret, args.window)
    l4_steps_s = rolling_mean(l4_steps, args.window) if any(l4_steps) else None
    l5_steps_s = rolling_mean(l5_steps, args.window) if any(l5_steps) else None

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(l4_ep, l4_ret_s, label="Level 4 (Q-learning)")
    axes[0].plot(l5_ep, l5_ret_s, label="Level 5 (SARSA)")
    axes[0].set_ylabel("Env return")
    axes[0].set_title("Monster levels: learning with stochastic transitions")
    axes[0].legend()

    if l4_steps_s is not None and l5_steps_s is not None:
        axes[1].plot(l4_ep, l4_steps_s, label="Level 4 steps")
        axes[1].plot(l5_ep, l5_steps_s, label="Level 5 steps")
        axes[1].set_ylabel("Steps")
        axes[1].legend()
    axes[1].set_xlabel("Episode")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
