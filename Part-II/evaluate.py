#!/usr/bin/env python3
"""
Evaluation script for trained Deep RL agents.
Runs the trained model with visual rendering.

Usage:
    python evaluate.py --model models/rotation_ppo_final.zip --env rotation
    python evaluate.py --model models/directional_dqn_final.zip --env directional
    python evaluate.py --random --env rotation  # Test with random actions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time

# Import environments to register them
import envs

from envs.rotation_env import RotationArenaEnv
from envs.directional_env import DirectionalArenaEnv

from stable_baselines3 import PPO, DQN


def create_env(env_type: str, render_mode: str = 'human'):
    """Create the appropriate environment."""
    if env_type == 'rotation':
        return RotationArenaEnv(render_mode=render_mode)
    elif env_type == 'directional':
        return DirectionalArenaEnv(render_mode=render_mode)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def load_model(model_path: str):
    """Load a trained model from disk."""

    # Determine algorithm from filename or try both
    if 'ppo' in model_path.lower():
        return PPO.load(model_path)
    elif 'dqn' in model_path.lower():
        return DQN.load(model_path)
    else:
        # Try PPO first, then DQN
        try:
            return PPO.load(model_path)
        except:
            return DQN.load(model_path)


def evaluate(args):
    """Run evaluation episodes with visualization."""

    print("=" * 60)
    print("Deep RL Arena - Evaluation")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    if args.random:
        print("Mode: Random actions")
    else:
        print(f"Model: {args.model}")
    print("=" * 60)

    # Create environment with rendering
    env = create_env(args.env, render_mode='human')

    # Load model or use random
    model = None
    if not args.random:
        try:
            model = load_model(args.model)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Falling back to random actions...")
            args.random = True

    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = False

        print(f"\nEpisode {episode + 1}/{args.episodes}")

        while not done:
            # Get action
            if args.random:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            # Render is handled by environment

            # Optional delay for visibility
            if args.delay > 0:
                time.sleep(args.delay / 1000.0)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        print(f"  Steps: {steps}, Reward: {total_reward:.2f}")
        print(f"  Phase: {info['phase']}, Score: {info['score']}")

        # Pause between episodes
        if episode < args.episodes - 1:
            print("  Press any key in game window to continue...")
            time.sleep(1)

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Average reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"Average length: {sum(episode_lengths) / len(episode_lengths):.1f}")
    print(f"Best reward: {max(episode_rewards):.2f}")
    print(f"Worst reward: {min(episode_rewards):.2f}")
    print("=" * 60)

    env.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Deep RL agents')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--env', type=str, default='rotation',
                       choices=['rotation', 'directional'],
                       help='Environment type')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of evaluation episodes')
    parser.add_argument('--random', action='store_true',
                       help='Use random actions instead of model')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic actions')
    parser.add_argument('--delay', type=int, default=0,
                       help='Delay between steps in ms (for visibility)')

    args = parser.parse_args()

    if not args.random and args.model is None:
        print("Warning: No model specified, using random actions")
        args.random = True

    evaluate(args)


if __name__ == '__main__':
    main()
