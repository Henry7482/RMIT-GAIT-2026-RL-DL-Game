#!/usr/bin/env python3
"""
Training script for Deep RL agents.

This is a placeholder with the structure for training.
You need to implement the actual RL training logic using Stable Baselines3.

Usage:
    python train.py --env rotation --algo ppo --timesteps 100000
    python train.py --env directional --algo dqn --timesteps 500000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from datetime import datetime

# Import environments to register them
import envs

# TODO: Uncomment these imports when implementing RL training
# from stable_baselines3 import PPO, DQN
# from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
# from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from envs.rotation_env import RotationArenaEnv
from envs.directional_env import DirectionalArenaEnv


def create_env(env_type: str, render_mode: str = None):
    """Create the appropriate environment."""
    if env_type == 'rotation':
        return RotationArenaEnv(render_mode=render_mode)
    elif env_type == 'directional':
        return DirectionalArenaEnv(render_mode=render_mode)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def train(args):
    """
    Main training function.

    TODO: Implement the following:
    1. Create vectorized environment with DummyVecEnv
    2. Choose algorithm (PPO or DQN) based on args.algo
    3. Configure neural network (at least one hidden layer)
    4. Set up TensorBoard logging
    5. Configure callbacks (evaluation, checkpoints)
    6. Train the model
    7. Save the final model
    """

    print("=" * 60)
    print("Deep RL Arena - Training")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algo}")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    # Create environment
    env = create_env(args.env)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    env.close()

    # =========================================================================
    # TODO: IMPLEMENT YOUR RL TRAINING LOGIC HERE
    # =========================================================================
    #
    # Example structure (uncomment and modify):
    #
    # # Create vectorized environment
    # def make_env():
    #     return create_env(args.env)
    # vec_env = DummyVecEnv([make_env])
    # vec_env = VecMonitor(vec_env)
    #
    # # Set up logging directory
    # log_dir = f"./logs/{args.env}_{args.algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    #
    # # Create model
    # if args.algo == 'ppo':
    #     model = PPO(
    #         "MlpPolicy",
    #         vec_env,
    #         learning_rate=args.lr,
    #         n_steps=2048,
    #         batch_size=64,
    #         n_epochs=10,
    #         gamma=0.99,
    #         policy_kwargs=dict(net_arch=[256, 256]),  # Two hidden layers
    #         tensorboard_log=log_dir,
    #         verbose=1,
    #     )
    # elif args.algo == 'dqn':
    #     model = DQN(
    #         "MlpPolicy",
    #         vec_env,
    #         learning_rate=args.lr,
    #         buffer_size=100000,
    #         learning_starts=10000,
    #         batch_size=64,
    #         gamma=0.99,
    #         exploration_fraction=0.2,
    #         exploration_final_eps=0.05,
    #         policy_kwargs=dict(net_arch=[256, 256]),
    #         tensorboard_log=log_dir,
    #         verbose=1,
    #     )
    #
    # # Set up callbacks
    # eval_env = create_env(args.env)
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=f"./models/{args.env}_{args.algo}_best",
    #     log_path=log_dir,
    #     eval_freq=10000,
    #     deterministic=True,
    #     render=False,
    # )
    #
    # checkpoint_callback = CheckpointCallback(
    #     save_freq=50000,
    #     save_path=f"./models/{args.env}_{args.algo}_checkpoints",
    #     name_prefix=args.algo,
    # )
    #
    # # Train
    # model.learn(
    #     total_timesteps=args.timesteps,
    #     callback=[eval_callback, checkpoint_callback],
    #     progress_bar=True,
    # )
    #
    # # Save final model
    # model_path = f"./models/{args.env}_{args.algo}_final"
    # model.save(model_path)
    # print(f"Model saved to {model_path}")
    #
    # vec_env.close()
    # eval_env.close()
    #
    # =========================================================================

    print("\n" + "=" * 60)
    print("PLACEHOLDER: Training logic not yet implemented")
    print("See the TODO comments in this file for implementation guide")
    print("=" * 60)

    # Quick test: run a few random steps
    print("\nTesting environment with random actions...")
    env = create_env(args.env)
    obs, info = env.reset()
    total_reward = 0

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Random agent test: {step + 1} steps, reward: {total_reward:.2f}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Train Deep RL agents')
    parser.add_argument('--env', type=str, default='rotation',
                       choices=['rotation', 'directional'],
                       help='Environment type (rotation or directional)')
    parser.add_argument('--algo', type=str, default='ppo',
                       choices=['ppo', 'dqn'],
                       help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
