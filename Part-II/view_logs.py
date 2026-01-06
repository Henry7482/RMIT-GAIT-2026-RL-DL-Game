#!/usr/bin/env python3
"""
Simple script to visualize training logs without TensorBoard.
Reads TensorFlow event files and plots training metrics.

Usage:
    python view_logs.py
    python view_logs.py --logdir logs/rotation_dqn_20260105_214256
"""

import argparse
import os
import glob
from pathlib import Path

try:
    from tensorboard.backend.event_processing import event_accumulator
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install: pip install matplotlib tensorboard")
    exit(1)


def find_latest_log_dir(base_dir='logs'):
    """Find the most recent log directory."""
    log_dirs = glob.glob(f"{base_dir}/*/")
    if not log_dirs:
        return None
    return max(log_dirs, key=os.path.getmtime)


def load_tensorboard_logs(log_dir):
    """Load all scalar data from TensorBoard logs."""
    # Find the event file
    event_files = glob.glob(f"{log_dir}/**/events.out.tfevents.*", recursive=True)
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None
    
    event_file = event_files[0]
    print(f"Loading: {event_file}")
    
    # Load the event file
    ea = event_accumulator.EventAccumulator(
        str(Path(event_file).parent),
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        }
    )
    ea.Reload()
    
    return ea


def plot_training_metrics(ea, save_path=None):
    """Plot key training metrics."""
    # Get available scalar tags
    tags = ea.Tags()['scalars']
    print(f"\nAvailable metrics: {tags}")
    
    # Define metrics to plot (including evaluation metrics)
    metrics_to_plot = {
        'rollout/ep_rew_mean': 'Training Reward',
        'eval/mean_reward': 'Evaluation Reward',
        'rollout/ep_len_mean': 'Training Episode Length',
        'eval/mean_ep_length': 'Evaluation Episode Length',
        'train/loss': 'Training Loss',
        'rollout/exploration_rate': 'Exploration Rate',
    }
    
    # Filter to only available metrics
    available_metrics = {k: v for k, v in metrics_to_plot.items() if k in tags}
    
    if not available_metrics:
        print("No standard metrics found in logs.")
        return
    
    # Create subplots in 3x2 grid layout
    n_metrics = len(available_metrics)
    n_rows = 3
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing
    
    # Plot each metric
    for idx, (tag, title) in enumerate(available_metrics.items()):
        if idx >= len(axes):
            break
        data = ea.Scalars(tag)
        steps = [d.step for d in data]
        values = [d.value for d in data]
        
        axes[idx].plot(steps, values, linewidth=2, color='tab:blue', alpha=0.6)
        axes[idx].set_xlabel('Training Steps')
        axes[idx].set_ylabel(title)
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        
        # Add smoothed line if enough data points
        if len(values) > 10:
            window = min(len(values) // 10, 50)
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
            axes[idx].plot(smooth_steps, smoothed, 'r-', linewidth=2, alpha=0.9, label='Smoothed')
            axes[idx].legend(loc='best', fontsize=8)
    
    # Hide any unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def print_summary(ea):
    """Print a text summary of training progress."""
    tags = ea.Tags()['scalars']
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    # Episode reward
    if 'rollout/ep_rew_mean' in tags:
        rewards = ea.Scalars('rollout/ep_rew_mean')
        if rewards:
            values = [r.value for r in rewards]
            print(f"\nEpisode Reward:")
            print(f"  Initial: {values[0]:.2f}")
            print(f"  Final:   {values[-1]:.2f}")
            print(f"  Best:    {max(values):.2f}")
            print(f"  Change:  {values[-1] - values[0]:+.2f}")
    
    # Episode length
    if 'rollout/ep_len_mean' in tags:
        lengths = ea.Scalars('rollout/ep_len_mean')
        if lengths:
            values = [l.value for l in lengths]
            print(f"\nEpisode Length:")
            print(f"  Initial: {values[0]:.1f}")
            print(f"  Final:   {values[-1]:.1f}")
            print(f"  Best:    {max(values):.1f}")
    
    # Training loss
    if 'train/loss' in tags:
        losses = ea.Scalars('train/loss')
        if losses:
            values = [l.value for l in losses]
            print(f"\nTraining Loss:")
            print(f"  Initial: {values[0]:.4f}")
            print(f"  Final:   {values[-1]:.4f}")
            print(f"  Min:     {min(values):.4f}")
    
    # Evaluation vs Training comparison (important for RL!)
    if 'rollout/ep_rew_mean' in tags and 'eval/mean_reward' in tags:
        train_rewards = ea.Scalars('rollout/ep_rew_mean')
        eval_rewards = ea.Scalars('eval/mean_reward')
        if train_rewards and eval_rewards:
            train_final = train_rewards[-1].value
            eval_final = eval_rewards[-1].value
            gap = train_final - eval_final
            print(f"\nTraining vs Evaluation Gap:")
            print(f"  Training reward (final):   {train_final:.2f}")
            print(f"  Evaluation reward (final): {eval_final:.2f}")
            print(f"  Gap: {gap:+.2f}")
            if abs(gap) < 10:
                print(f"  ✅ Policy is robust (small gap)")
            elif gap > 0:
                print(f"  ⚠️  Policy relies on exploration to succeed")
            else:
                print(f"  ✅ Policy performs better without exploration")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Visualize training logs')
    parser.add_argument('--logdir', type=str, default=None,
                       help='Path to log directory (default: most recent)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save plot to file instead of showing')
    parser.add_argument('--no-plot', action='store_true',
                       help='Only print summary, no plots')
    
    args = parser.parse_args()
    
    # Find log directory
    if args.logdir:
        log_dir = args.logdir
    else:
        log_dir = find_latest_log_dir()
        if not log_dir:
            print("No log directories found in 'logs/'")
            return
    
    print(f"Using log directory: {log_dir}")
    
    # Load logs
    ea = load_tensorboard_logs(log_dir)
    if ea is None:
        return
    
    # Print summary
    print_summary(ea)
    
    # Plot if requested
    if not args.no_plot:
        plot_training_metrics(ea, save_path=args.save)


if __name__ == '__main__':
    main()
