"""
Rotation-based control environment (Control Style 1).

Actions:
    0: No action
    1: Thrust forward
    2: Rotate left
    3: Rotate right
    4: Shoot
"""

from gymnasium import spaces
from .base_env import BaseArenaEnv


class RotationArenaEnv(BaseArenaEnv):
    """
    Arena environment with rotation-based controls.

    The player rotates left/right and thrusts forward, like Asteroids.
    """

    def __init__(self, **kwargs):
        super().__init__(env_type='rotation', **kwargs)

        # 5 discrete actions: nothing, thrust, rotate_left, rotate_right, shoot
        self.action_space = spaces.Discrete(5)

        # Action mapping
        self.action_names = {
            0: 'nothing',
            1: 'thrust',
            2: 'rotate_left',
            3: 'rotate_right',
            4: 'shoot',
        }

    def _apply_action(self, action: int) -> None:
        """Apply rotation-based action to the player."""
        player = self.arena.player
        
        # Track current action for stuck rotation logic
        self.arena.current_action = action

        if action == 0:
            # No action - just drift
            pass
        elif action == 1:
            # Thrust forward
            player.apply_thrust()
            self.arena.events['thrust_action'] = 1  # Track for stuck detection
        elif action == 2:
            # Rotate left (counter-clockwise)
            player.rotate_left()
        elif action == 3:
            # Rotate right (clockwise)
            player.rotate_right()
        elif action == 4:
            # Shoot
            self.arena.player_shoot()
