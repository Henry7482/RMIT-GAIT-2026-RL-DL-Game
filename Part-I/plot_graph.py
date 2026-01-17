import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple


def read_log(path: Path | str) -> Tuple[List[int], List[float], List[float], List[int]]:
    episodes, env_returns, total_returns, steps = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            env_returns.append(float(row["env_return"]))
            total_returns.append(float(row["total_return"]))
            steps.append(int(row.get("steps", 0)))
    return episodes, env_returns, total_returns, steps
    

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
    parser = argparse.ArgumentParser(description="Plot Level 6 intrinsic vs no-intrinsic returns")
    base_dir = Path(__file__).parent
    parser.add_argument("--on", default=base_dir / "logs/level6_q_withReward.csv", help="CSV for intrinsic ON run")
    parser.add_argument("--off", default=base_dir / "logs/level6_q_noReward.csv", help="CSV for intrinsic OFF run")
    parser.add_argument("--window", type=int, default=20, help="Rolling mean window (episodes)")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write("matplotlib is required for plotting. Install via `python -m pip install matplotlib`.\n")
        sys.exit(1)

    on_ep, on_env, on_total, on_steps = read_log(args.on)
    off_ep, off_env, off_total, off_steps = read_log(args.off)

    # Trim to shortest length to align episodes if logs differ
    n = min(len(on_ep), len(off_ep))
    on_ep, on_env, on_total, on_steps = on_ep[:n], on_env[:n], on_total[:n], on_steps[:n]
    off_ep, off_env, off_steps = off_ep[:n], off_env[:n], off_steps[:n]

    on_env_s = rolling_mean(on_env, args.window)
    off_env_s = rolling_mean(off_env, args.window)
    on_tot_s = rolling_mean(on_total, args.window)
    on_steps_s = rolling_mean(on_steps, args.window) if any(on_steps) else None
    off_steps_s = rolling_mean(off_steps, args.window) if any(off_steps) else None

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(on_ep, on_env_s, label="Intrinsic ON (env, smoothed)")
    axes[0].plot(off_ep, off_env_s, label="Intrinsic OFF (env, smoothed)")
    axes[0].plot(on_ep, on_tot_s, "--", label="Intrinsic ON (total, smoothed)")
    axes[0].set_ylabel("Return")
    axes[0].set_title("Level 6: Intrinsic Reward vs No Intrinsic")
    axes[0].legend()

    if on_steps_s is not None and off_steps_s is not None:
        axes[1].plot(on_ep, on_steps_s, label="Intrinsic ON (steps, smoothed)")
        axes[1].plot(off_ep, off_steps_s, label="Intrinsic OFF (steps, smoothed)")
        axes[1].set_ylabel("Steps")
        axes[1].legend()
    axes[1].set_xlabel("Episode")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
