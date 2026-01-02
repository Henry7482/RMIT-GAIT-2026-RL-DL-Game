"""
Base Gym environment for the Deep RL Arena.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Tuple, Dict, Any

from game.arena import Arena
from game.constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, OBSERVATION_SIZE,
)


class BaseArenaEnv(gym.Env):
    """
    Base Gymnasium environment for the arena game.
    Subclasses implement specific control schemes.
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': FPS,
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.render_mode = render_mode
        self.arena = Arena()

        # Observation space: fixed-size vector
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32
        )

        # Action space will be defined by subclasses
        self.action_space = None

        # Pygame setup
        self.screen = None
        self.clock = None
        self._pygame_initialized = False

    def _init_pygame(self):
        """Initialize pygame for rendering."""
        if not self._pygame_initialized:
            pygame.init()
            pygame.display.set_caption("Deep RL Arena")
            if self.render_mode == 'human':
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            else:
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self._pygame_initialized = True

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        Returns: (observation, info)
        """
        super().reset(seed=seed)

        self.arena.reset()
        observation = np.array(self.arena.get_observation(), dtype=np.float32)
        info = self.arena.get_info()

        if self.render_mode == 'human':
            self._init_pygame()
            self.render()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        Returns: (observation, reward, terminated, truncated, info)
        """
        # Apply action (implemented by subclass)
        self._apply_action(action)

        # Update arena
        reward = self.arena.update()

        # Get observation
        observation = np.array(self.arena.get_observation(), dtype=np.float32)

        # Check termination
        terminated = self.arena.done and not self.arena.player.alive
        truncated = self.arena.done and self.arena.player.alive

        # Get info
        info = self.arena.get_info()

        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action: int) -> None:
        """
        Apply the given action to the player.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclass must implement _apply_action")

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current state.
        Returns RGB array if render_mode is 'rgb_array'.
        """
        self._init_pygame()

        # Handle pygame events (to prevent freezing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        # Render arena
        self.arena.render(self.screen)

        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(FPS)
            return None
        elif self.render_mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )

    def close(self) -> None:
        """Clean up resources."""
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False
            self.screen = None
            self.clock = None
