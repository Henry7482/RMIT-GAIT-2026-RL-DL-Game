#!/usr/bin/env python3
"""
Training script for Deep Q-Network (DQN) agents.

Usage:
    python train_dqn.py --env rotation --timesteps 100000
    python train_dqn.py --env directional --timesteps 500000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from datetime import datetime

# Import environments to register them
import envs

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from envs.rotation_env import RotationArenaEnv
from envs.directional_env import DirectionalArenaEnv


class GameMetricsCallback(BaseCallback):
    """
    Custom callback to log game-specific metrics (score, phase) to TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_scores = []
        self.episode_phases = []
        
    def _on_step(self) -> bool:
        # Check if episode ended (done signal)
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        
        for i, (info, done) in enumerate(zip(infos, dones)):
            if done:
                # Log score and phase at end of episode
                if 'score' in info:
                    self.episode_scores.append(info['score'])
                if 'phase' in info:
                    self.episode_phases.append(info['phase'])
        
        return True
    
    def _on_rollout_end(self) -> None:
        # Log average metrics when we have data
        if self.episode_scores:
            self.logger.record('game/score_mean', sum(self.episode_scores) / len(self.episode_scores))
            self.episode_scores = []
        if self.episode_phases:
            self.logger.record('game/phase_mean', sum(self.episode_phases) / len(self.episode_phases))
            self.episode_phases = []


def create_env(env_type: str, render_mode: str = None, curriculum: bool = False, total_timesteps: int = None):
    """Create the appropriate environment.
    
    Args:
        env_type: 'rotation' or 'directional'
        render_mode: Optional render mode
        curriculum: If True, wrap with curriculum learning
        total_timesteps: Required if curriculum=True
    """
    if env_type == 'rotation':
        env = RotationArenaEnv(render_mode=render_mode)
    elif env_type == 'directional':
        env = DirectionalArenaEnv(render_mode=render_mode)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    if curriculum:
        from envs.curriculum_wrapper import CurriculumWrapper
        if total_timesteps is None:
            raise ValueError("total_timesteps required when curriculum=True")
        env = CurriculumWrapper(env, total_timesteps)
    
    return env


def train(args):
    """
    Main training function for DQN.
    """
    print("=" * 60)
    print("Deep Q-Network (DQN) Training")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Learning rate: {args.lr}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Create directories
    models_dir = "./models"
    logs_dir = "./logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Create vectorized environment (with optional curriculum)
    curriculum_enabled = getattr(args, 'curriculum', False)
    if curriculum_enabled:
        print(f"[Curriculum] Enabled - difficulty will scale with progress")
    
    def make_env():
        return create_env(
            args.env, 
            curriculum=curriculum_enabled, 
            total_timesteps=args.timesteps
        )
    
    vec_env = DummyVecEnv([make_env])
    vec_env = VecMonitor(vec_env)

    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")

    # Set up logging directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"{logs_dir}/{args.env}_dqn_{timestamp}"

    # Create DQN model
    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs=dict(net_arch=[512, 256, 128]),  # Three hidden layers 
        tensorboard_log=log_dir,
        verbose=1,
    )

    print(f"\nModel architecture: MLP with hidden layers [256, 256, 128]")
    print(f"TensorBoard logs: {log_dir}")

    # Set up callbacks
    eval_env = create_env(args.env)
    
    # Create model save paths
    best_model_path = f"{models_dir}/{args.env}_dqn_best"
    checkpoint_path = f"{models_dir}/{args.env}_dqn_checkpoints"
    final_model_path = f"{models_dir}/{args.env}_dqn_final"
    
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoint_path,
        name_prefix="dqn",
    )

    print(f"\nStarting training for {args.timesteps} timesteps...")
    print(f"Best model will be saved to: {best_model_path}")
    print(f"Checkpoints will be saved to: {checkpoint_path}")
    print(f"Final model will be saved to: {final_model_path}")
    print("=" * 60)

    # Create custom game metrics callback
    metrics_callback = GameMetricsCallback()

    # Train the model
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback, metrics_callback],
        progress_bar=True,
    )

    # Save final model
    model.save(final_model_path)
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {best_model_path}")
    print(f"{'=' * 60}")

    # Cleanup
    vec_env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent')
    
    # Environment arguments
    parser.add_argument('--env', type=str, default='rotation',
                       choices=['rotation', 'directional'],
                       help='Environment type (rotation or directional)')
    
    # Training arguments
    parser.add_argument('--timesteps', type=int, default=10000000,
                       help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=5e-6,
                       help='Learning rate')
    
    # DQN-specific arguments
    parser.add_argument('--buffer_size', type=int, default=3000000,
                       help='Size of the replay buffer')
    parser.add_argument('--learning_starts', type=int, default=1000,
                       help='Number of steps before learning starts')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--tau', type=float, default=1.0,
                       help='Soft update coefficient for target network')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--train_freq', type=int, default=4,
                       help='Update the model every train_freq steps')
    parser.add_argument('--gradient_steps', type=int, default=1,
                       help='Number of gradient steps after each rollout')
    parser.add_argument('--target_update_interval', type=int, default=500,
                       help='Update target network every N steps')
    
    # Exploration arguments
    parser.add_argument('--exploration_fraction', type=float, default=0.5,
                       help='Fraction of training for epsilon exploration decay')
    parser.add_argument('--exploration_initial_eps', type=float, default=1.0,
                       help='Initial exploration epsilon')
    parser.add_argument('--exploration_final_eps', type=float, default=0.1,
                       help='Final exploration epsilon')
    
    # Callback arguments
    parser.add_argument('--eval_freq', type=int, default=10000,
                       help='Evaluation frequency (timesteps)')
    parser.add_argument('--n_eval_episodes', type=int, default=5,
                       help='Number of episodes for evaluation')
    parser.add_argument('--checkpoint_freq', type=int, default=25000,
                       help='Checkpoint save frequency (timesteps)')
    
    # Curriculum learning
    parser.add_argument('--curriculum', action='store_true',
                       help='Enable curriculum learning (easy -> hard)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
