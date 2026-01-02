"""
Gymnasium environments for the Deep RL Arena.
"""

from gymnasium.envs.registration import register

from .base_env import BaseArenaEnv
from .rotation_env import RotationArenaEnv
from .directional_env import DirectionalArenaEnv

# Register environments with Gymnasium
register(
    id='ArenaRotation-v0',
    entry_point='envs.rotation_env:RotationArenaEnv',
)

register(
    id='ArenaDirectional-v0',
    entry_point='envs.directional_env:DirectionalArenaEnv',
)

__all__ = [
    'BaseArenaEnv',
    'RotationArenaEnv',
    'DirectionalArenaEnv',
]
