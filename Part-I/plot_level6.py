import argparse
import csv
import sys


def read_log(path):
    episodes, env_returns, total_returns = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            env_returns.append(float(row["env_return"]))
            total_returns.append(float(row["total_return"]))
    return episodes, env_returns, total_returns


def main():
    parser = argparse.ArgumentParser(description="Plot Level 6 intrinsic vs no-intrinsic returns")
    parser.add_argument("--on", default="logs/level6_q_withReward_part1.csv", help="CSV for intrinsic ON run")
    parser.add_argument("--off", default="logs/level6_q_noReward_part1.csv", help="CSV for intrinsic OFF run")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write("matplotlib is required for plotting. Install via `python -m pip install matplotlib`.\n")
        sys.exit(1)

    on_ep, on_env, on_total = read_log(args.on)
    off_ep, off_env, off_total = read_log(args.off)

    # Trim to shortest length to align episodes if logs differ
    n = min(len(on_ep), len(off_ep))
    on_ep, on_env, on_total = on_ep[:n], on_env[:n], on_total[:n]
    off_ep, off_env = off_ep[:n], off_env[:n]

    plt.plot(on_ep, on_env, label="Intrinsic ON (env return)")
    plt.plot(off_ep, off_env, label="Intrinsic OFF (env return)")
    plt.plot(on_ep, on_total, "--", label="Intrinsic ON (total return)")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Level 6: Intrinsic Reward vs No Intrinsic")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
