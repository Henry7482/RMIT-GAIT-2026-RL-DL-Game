#!/usr/bin/env python3
"""
Training script for Deep RL agents.

Usage:
    python train.py --env rotation --algo ppo --timesteps 5000000
    python train.py --env directional --algo dqn --timesteps 5000000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from datetime import datetime

# Import environments to register them
import envs

from envs.rotation_env import RotationArenaEnv
from envs.directional_env import DirectionalArenaEnv


def create_env(env_type: str, render_mode: str = None):
    if env_type == 'rotation':
        return RotationArenaEnv(render_mode=render_mode)
    elif env_type == 'directional':
        return DirectionalArenaEnv(render_mode=render_mode)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def train(args):
    print("=" * 60)
    print("Deep RL Arena - Training")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algo}")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    env = create_env(args.env)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    env.close()

    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.env_util import make_vec_env

    print("\nCreating vectorized environment (4 parallel envs)...")

    vec_env = make_vec_env(
        lambda: create_env(args.env),
        n_envs=4
    )
    

    def linear_schedule(initial_value: float):
        """Linear learning rate schedule from initial_value to 2e-5."""
        def func(progress_remaining: float) -> float:
            min_lr = 2e-5
            return min_lr + progress_remaining * (initial_value - min_lr)
        return func
    
    
    def linear_schedule_3m(initial_value: float, total_timesteps: int):
        """Decays linearly to 2e-5 at 3M timesteps, then stays constant."""
        decay_timesteps = 3_000_000
        min_lr = 2e-5

        def func(progress_remaining: float) -> float:
            current_timestep = total_timesteps * (1 - progress_remaining)
            if current_timestep >= decay_timesteps:
                return min_lr
            else:
                decay_progress = current_timestep / decay_timesteps
                return initial_value - decay_progress * (initial_value - min_lr)
        return func


    def step_schedule_2m(total_timesteps: int):
        """1e-4 for first 2M timesteps, then 3e-5 afterward."""
        switch_timesteps = 2_000_000
        high_lr = 1e-4
        low_lr = 3e-5

        def func(progress_remaining: float) -> float:
            current_timestep = total_timesteps * (1 - progress_remaining)
            return low_lr if current_timestep >= switch_timesteps else high_lr
        return func

    if args.algo == 'ppo':
        print("Creating PPO model with hyperparameters...")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=linear_schedule_3m(args.lr, args.timesteps),  # Decay to 2e-5 at 3M, then constant
            n_steps=4096,              # Number of steps to run for each environment per update (4096 * 4 = 16384 steps per update!)
            batch_size=256,            # Minibatch size for each gradient update (increased for parallel data)
            n_epochs=10,               # Number of epoch when optimizing the surrogate loss
            gamma=0.99,                # Discount factor
            gae_lambda=0.95,           # Factor for trade-off of bias vs variance for GAE
            clip_range=0.2,            # Clipping parameter for PPO
            clip_range_vf=None,        # Clipping parameter for value function (None = no clipping)
            ent_coef=0.10,             # Entropy coefficient for exploration
            vf_coef=0.5,               # Value function coefficient for loss calculation
            max_grad_norm=0.5,         # Maximum value for gradient clipping
            use_sde=False,             # Whether to use State Dependent Exploration
            sde_sample_freq=-1,        # Sample a new noise matrix every n steps (-1 = only at rollout start)
            target_kl=None,            # Target KL divergence threshold (None = no limit)
            tensorboard_log=f"./logs/{args.env}_{args.algo}",
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Policy and value network architecture
            ),
            verbose=1,
            seed=None,                 # Random seed
            device="auto"              # Device: 'cpu', 'cuda', or 'auto'
        )

        print(f"\nTraining PPO for {args.timesteps} timesteps...")
        model.learn(total_timesteps=args.timesteps)

        # Save model
        os.makedirs("./models", exist_ok=True)
        model.save(f"./models/{args.env}_{args.algo}_test")
        print(f"\n✓ Model saved to ./models/{args.env}_{args.algo}_test.zip")

        vec_env.close()

    elif args.algo == 'dqn':
        # DQN: Uses single environment with replay buffer (off-policy)
        print("\nCreating single environment for DQN (off-policy)...")
        dqn_env = create_env(args.env)

        print("Creating DQN model with hyperparameters...")
        model = DQN(
            policy="MlpPolicy",
            env=dqn_env,
            learning_rate=linear_schedule_3m(args.lr, args.timesteps),
            buffer_size=100_000,                      # Replay buffer capacity
            learning_starts=10_000,                   # Steps before learning begins
            batch_size=128,                           # Minibatch size for gradient updates
            tau=0.005,                                # Soft update coefficient for target network
            gamma=0.99,                               # Discount factor (match PPO)
            train_freq=(4, "step"),                   # Train every 4 environment steps
            gradient_steps=1,                         # Gradient steps per training call
            target_update_interval=1000,              # Steps between target network updates
            exploration_fraction=0.3,                 # Fraction of training for epsilon decay
            exploration_initial_eps=1.0,             # Starting epsilon (full exploration)
            exploration_final_eps=0.05,              # Final epsilon (5% random actions)
            max_grad_norm=10.0,                       # Gradient clipping (DQN default)
            tensorboard_log=f"./logs/{args.env}_{args.algo}",
            policy_kwargs=dict(
                net_arch=[256, 256]                   # Match PPO network architecture
            ),
            verbose=1,
            seed=None,
            device="auto"
        )

        print(f"\nTraining DQN for {args.timesteps} timesteps...")
        print(f"  Buffer size: 100,000")
        print(f"  Epsilon: 1.0 -> 0.05 over 30% of training")
        print(f"  Learning starts after: 10,000 steps")

        # Setup checkpoint callback to save model every 100k steps
        from stable_baselines3.common.callbacks import CheckpointCallback
        
        os.makedirs("./models", exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=300000,
            save_path="./models/auto_checkpoints",
            name_prefix="directional_dqn_best",
            save_replay_buffer=False,
            save_vecnormalize=False,
            verbose=1
        )
        
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

        # Save model
        os.makedirs("./models", exist_ok=True)
        model.save(f"./models/{args.env}_{args.algo}_test")
        print(f"\n✓ Model saved to ./models/{args.env}_{args.algo}_test.zip")

        dqn_env.close()


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
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Initial learning rate (will be scheduled)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
