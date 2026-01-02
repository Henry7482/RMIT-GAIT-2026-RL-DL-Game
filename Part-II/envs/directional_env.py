"""
Directional movement environment (Control Style 2).

Actions:
    0: No action
    1: Move up
    2: Move down
    3: Move left
    4: Move right
    5: Shoot
"""

from gymnasium import spaces
from .base_env import BaseArenaEnv


class DirectionalArenaEnv(BaseArenaEnv):
    """
    Arena environment with directional controls.

    The player moves in 4 cardinal directions directly.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 6 discrete actions: nothing, up, down, left, right, shoot
        self.action_space = spaces.Discrete(6)

        # Action mapping
        self.action_names = {
            0: 'nothing',
            1: 'up',
            2: 'down',
            3: 'left',
            4: 'right',
            5: 'shoot',
        }

    def _apply_action(self, action: int) -> None:
        """Apply directional action to the player."""
        player = self.arena.player

        if action == 0:
            # No action - stop moving
            player.move_direction(0, 0)
        elif action == 1:
            # Move up
            player.move_direction(0, -1)
        elif action == 2:
            # Move down
            player.move_direction(0, 1)
        elif action == 3:
            # Move left
            player.move_direction(-1, 0)
        elif action == 4:
            # Move right
            player.move_direction(1, 0)
        elif action == 5:
            # Shoot (don't change movement)
            self.arena.player_shoot()
