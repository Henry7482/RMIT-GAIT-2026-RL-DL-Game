"""
Curriculum Learning Wrapper for Arena Environment.

Uses monkey-patching to modify game constants based on training progress.
Automatically adjusts game difficulty:
- Stage 1 (0-30%): Easy - 1 spawner, 2 max enemies, 80% enemy speed
- Stage 2 (30-60%): Medium - 2 spawners, 3 max enemies, normal speed  
- Stage 3 (60-100%): Hard - Full difficulty
"""

import gymnasium as gym
from game import constants


class CurriculumWrapper(gym.Wrapper):
    """
    Wrapper that implements curriculum learning by monkey-patching
    game constants based on training progress (timesteps).
    """
    
    def __init__(self, env, total_timesteps: int):
        super().__init__(env)
        self.total_timesteps = total_timesteps
        self.current_timestep = 0
        
        # Save original values for restoration
        self._original_initial_spawners = constants.INITIAL_SPAWNERS
        self._original_spawner_max_enemies = constants.SPAWNER_MAX_ENEMIES
        self._original_enemy_speed = constants.ENEMY_SPEED
        self._original_max_phase = constants.MAX_PHASE
        
        # Difficulty stages (progress thresholds)
        self.stages = [
            (0.0, 0.3, "easy"),    # 0-30%: Easy
            (0.3, 0.6, "medium"),  # 30-60%: Medium
            (0.6, 1.0, "hard"),    # 60-100%: Hard
        ]
        
        # Difficulty configurations
        self.difficulty_config = {
            "easy": {
                "initial_spawners": 1,
                "spawner_max_enemies": 2,
                "enemy_speed": 2.0,  # 80% of base 2.5
                "max_phase": 1,      # Only phase 1
            },
            "medium": {
                "initial_spawners": 2,
                "spawner_max_enemies": 3,
                "enemy_speed": 2.5,  # Normal
                "max_phase": 2,      # Up to phase 2
            },
            "hard": {
                "initial_spawners": self._original_initial_spawners,
                "spawner_max_enemies": self._original_spawner_max_enemies,
                "enemy_speed": self._original_enemy_speed,
                "max_phase": self._original_max_phase,  # Full game
            },
        }
        
        self.current_stage = "easy"
        self._stage_announced = set()
    
    def get_progress(self) -> float:
        """Get training progress as fraction [0, 1]."""
        return min(self.current_timestep / max(self.total_timesteps, 1), 1.0)
    
    def get_current_stage(self) -> str:
        """Get current difficulty stage based on progress."""
        progress = self.get_progress()
        for start, end, stage_name in self.stages:
            if start <= progress < end:
                return stage_name
        return "hard"
    
    def apply_difficulty(self, stage: str):
        """Apply difficulty settings by patching constants."""
        config = self.difficulty_config[stage]
        
        # Monkey-patch the constants module
        constants.INITIAL_SPAWNERS = config["initial_spawners"]
        constants.SPAWNER_MAX_ENEMIES = config["spawner_max_enemies"]
        constants.ENEMY_SPEED = config["enemy_speed"]
        constants.MAX_PHASE = config["max_phase"]
    
    def reset(self, **kwargs):
        """Reset environment and apply current difficulty."""
        # Update stage based on progress
        new_stage = self.get_current_stage()
        
        # Announce stage changes
        if new_stage != self.current_stage:
            print(f"[Curriculum] Stage change: {self.current_stage} -> {new_stage} "
                  f"(progress: {self.get_progress()*100:.1f}%)")
            self.current_stage = new_stage
        elif new_stage not in self._stage_announced:
            print(f"[Curriculum] Starting with stage: {new_stage}")
            self._stage_announced.add(new_stage)
        
        # Apply difficulty settings BEFORE reset
        self.apply_difficulty(self.current_stage)
        
        # Now reset the environment (Arena will use updated constants)
        obs, info = self.env.reset(**kwargs)
        
        # Add curriculum info to info dict
        info['curriculum_stage'] = self.current_stage
        info['curriculum_progress'] = self.get_progress()
        
        return obs, info
    
    def step(self, action):
        """Step environment and track progress."""
        self.current_timestep += 1
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add curriculum info to info dict
        info['curriculum_stage'] = self.current_stage
        info['curriculum_progress'] = self.get_progress()
        
        return obs, reward, terminated, truncated, info
    
    def set_timestep(self, timestep: int):
        """Set current timestep (useful for resuming training)."""
        self.current_timestep = timestep
        self.current_stage = self.get_current_stage()
    
    def restore_defaults(self):
        """Restore original constant values."""
        constants.INITIAL_SPAWNERS = self._original_initial_spawners
        constants.SPAWNER_MAX_ENEMIES = self._original_spawner_max_enemies
        constants.ENEMY_SPEED = self._original_enemy_speed
        constants.MAX_PHASE = self._original_max_phase
    
    def close(self):
        """Close environment and restore defaults."""
        self.restore_defaults()
        super().close()
    
    def __del__(self):
        """Ensure defaults are restored on garbage collection."""
        try:
            self.restore_defaults()
        except:
            pass  # Ignore errors during cleanup


def make_curriculum_env(env_type: str, total_timesteps: int, render_mode: str = None):
    """
    Factory function to create a curriculum-wrapped environment.
    
    Args:
        env_type: 'rotation' or 'directional'
        total_timesteps: Total planned training timesteps
        render_mode: Optional render mode
        
    Returns:
        CurriculumWrapper wrapping the base environment
    """
    from envs.rotation_env import RotationArenaEnv
    from envs.directional_env import DirectionalArenaEnv
    
    if env_type == 'rotation':
        env = RotationArenaEnv(render_mode=render_mode)
    else:
        env = DirectionalArenaEnv(render_mode=render_mode)
    
    return CurriculumWrapper(env, total_timesteps)
