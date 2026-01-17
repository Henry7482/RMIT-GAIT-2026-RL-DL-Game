"""
Game constants and configuration for the Deep RL Arena.
"""

# Screen settings
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60

# Colors - Dark space theme with neon accents
COLORS = {
    'background': (10, 10, 25),
    'player': (0, 255, 200),
    'player_glow': (0, 200, 150),
    'enemy': (255, 50, 80),
    'enemy_glow': (200, 30, 60),
    'spawner': (255, 150, 0),
    'spawner_glow': (200, 100, 0),
    'projectile_player': (0, 255, 255),
    'projectile_enemy': (255, 100, 100),
    'health_bar_bg': (40, 40, 60),
    'health_bar_player': (0, 255, 100),
    'health_bar_enemy': (255, 80, 80),
    'health_bar_spawner': (255, 180, 0),
    'ui_text': (200, 200, 220),
    'ui_accent': (100, 200, 255),
    'phase_indicator': (255, 220, 100),
    'particle': (255, 255, 200),
}

# Player settings
PLAYER_RADIUS = 20
PLAYER_MAX_HEALTH = 100
PLAYER_SPEED = 5.0  # For directional movement
PLAYER_THRUST = 0.3  # For rotation movement
PLAYER_MAX_VELOCITY = 6.0
PLAYER_ROTATION_SPEED = 5.0  # Degrees per frame
                            # 13:38 12/01/2026 - Temporarily changed to 5.0 for testing
                            # 13:50 12/01/2026 - Reverted to 2.0 (original game mechanics) after confirming penalties were bottleneck
PLAYER_FRICTION = 0.98
PLAYER_SHOOT_COOLDOWN = 10  # Frames between shots
PLAYER_INVINCIBILITY_FRAMES = 30  # Frames of invincibility after hit

# Projectile settings
PROJECTILE_SPEED = 10.0
PROJECTILE_RADIUS = 5
PROJECTILE_LIFETIME = 120  # Frames before disappearing

# Enemy settings
ENEMY_RADIUS = 15
ENEMY_HEALTH = 30
ENEMY_SPEED = 2.5
ENEMY_DAMAGE = 10

# Spawner settings
SPAWNER_RADIUS = 42  # Increased from 35 to match visual glow and improve hitbox
SPAWNER_HEALTH = 100
SPAWNER_SPAWN_INTERVAL = 180  # Frames between spawning enemies
SPAWNER_MAX_ENEMIES = 5  # Max enemies a spawner can have active

# Phase settings
INITIAL_SPAWNERS = 2
SPAWNERS_PER_PHASE = 1  # Additional spawners per phase
MAX_PHASE = 5
PHASE_ENEMY_HEALTH_MULT = 1.2  # Enemy health multiplier per phase
PHASE_ENEMY_SPEED_MULT = 1.1  # Enemy speed multiplier per phase

# Episode settings
MAX_STEPS = 5000  # Maximum steps per episode
STEP_PENALTY = 0.0  # Penalty per step (to encourage faster completion)

# ============================================================================
# REWARD VALUES - SEPARATED BY ALGORITHM/ENVIRONMENT
# ============================================================================

# PPO with Rotation control (Asteroids-style)
REWARD_PPO_ROTATIONAL = {
    # Aim: Reduce suicidal even more, and focus little more on kill enemies to prevent health losing
    'destroy_enemy': 170.0,       # Increased from 50
                                # Decrease to 50 to balance with destroy spawner
                                # 16:46 9/1/2026 - Increase to 60 to encourage killing enemies first
                                # 17:12 9/1/2026 - Increase to 80 to make killing enemies even more valueable
                                # 18:35 9/1/2026 - Increase to 85 to make killing enemies even more valueable
                                # 16:12 10/1/2025 - Increase to 90 as well as increase take damage to -50 to make killing enemies even more valueable
                                # 17:09 10/1/2026 - Increase to 100 to encourage kill remaining enemies even after kill spawners
                                # 00:32 11/1/2026 - Increase to 120 to encourage kill remaining enemies even after kill spawners
                                # 21:22 11/1/2026 - Increase even more to 150
                                # 16:06 12/1/2026 - Increase to 170 (close to a spawner)
    'destroy_spawner': 180.0,    # Increased from 100
                                # 18:35 9/1/2026 - Decrease a bit from 200 to 180 to encourage focus on enemies

    'phase_complete': 500.0,     # Increased from 200
                                # 16:12 10/1/2025 - Decrease drastically to 400 to hope for no longer just focus on spawners
                                # Increase right back to 500 (cause 400 potentially not enough for it to aim at the spawners at all)
    'clean_phase_bonus': 300,    # Bonus for no enemies alive when phase completes (syncs enemy clearing with phase reward)
    'hit_enemy': 30.0,           # Increased from 5
                                # 00:32 11/1/2026 - Increase from 10 to 15 to encourse finishing enemies before moving on to next spawners
                                # 21:22 11/1/2026 - Increase more (from 15 to 30) for former timesteps of random behaviour to learn that enemies are worthy too
    'hit_spawner': 0.0,
    'take_damage': -80.0,        # Increase from -10 (more scary)
                                # 16:46 9/1/2026 - Decrease to -35: Encourage to kill enemies before
                                # 17:12 9/1/2026 - Increased from -35 to -50 - make damage more scary
                                # Decrease right back to -40 (-50 is too agressive)
                                # 13:20 10/1/2026 - Increase from -40 to -45 to see if theres possibility to prioritize kill the enemies
                                # 16:12 10/1/2025 - Increase to 90 as well as increase take damage to -50 to make killing enemies even more valueable
                                # Switch right back to -45 (keep only the increase of kill enemy)
                                # 17:09 10/1/2026 - Increase to -46
                                # 22:51 10/1/2025 - Increase to -50 again with 4m timestep to see if it can kill remaining enemies before switching to next spawners
                                # 11/1/2026 - Set to -50 to discourage hide-after-hit behavior
                                # 15:00 12/1/2026 - Increase to -75 to see if the agent sees enemies as losing health, rather than just obstacle to shooting spawners
                                # 15:45 12/1/2026 - Increase to -100 to see if the agent sees enemies as losing health, rather than just obstacle to shooting spawners
                                # -100 is too high
    'death': -200.0,             # Increase from -50 (MUCH more scary)
                                # 17:12 9/1/2026: increase from -200 to -300 to even more agressively about losing health
                                # Decrease back to -200 right away (The agent got stuck in the corner with this)
    'shot_fired': -0.05,         # Changed from -0.1 to -0.05 - Reduced penalty to allow more shooting
    'existence_penalty': -0.015,  # Time pressure penalty per step to encourage faster completion
                                 # 12/1/2026 - Combined survival_bonus (-0.05) + existence_penalty (-0.015) = -0.065
                                 # 11:37 12/01/2025 - -0.065 is too high 
    'potential_closer': 0.1,     # Changed from 0.01 to 0.1 - Increased 10x to make navigation signal stronger
                                 # 11/1/2026 - Reduced from 0.1 to 0.05 (too aggressive, agent lost direction)
                                 # 11/1/2026 - Adjusted to 0.075 (25% reduction) to balance guidance vs freedom
                                 # 11/1/2026 - Restored to 0.1 to strengthen movement signal - agent should break or learn when to move/stay
    'potential_further': 0.0,    # Changed from -0.01 to -0.12 - Increased penalty to strongly discourage retreating
                                 # 11/1/2026 - Reduced from -0.12 to -0.06 (too aggressive)
                                 # 11/1/2026 - Adjusted to -0.09 (25% reduction) for balance
                                 # 11/1/2026 - Restored to -0.12 to strengthen movement signal - agent should break or learn when to move/stay
                                 # 13:50 12/01/2026 - REMOVED PENALTY: Set to 0.0. Fast rotation test (5°/frame) showed penalties were bottleneck, not mechanics.
                                 #                    Let positive gradients (+0.1 closer) + natural rewards + high entropy guide navigation without artificial constraints.
    'orientation_closer': 0.1,   # New - Reward for rotating towards nearest spawner (matches potential_closer)
                                 # 11/1/2026 - Adjusted from 0.1 to 0.075 to match potential_closer
                                 # 11/1/2026 - Restored to 0.1 to strengthen movement signal - agent should break or learn when to move/stay
    'orientation_further': 0.0,  # New - Penalty for rotating away from nearest spawner (matches potential_further)
                                 # 11/1/2026 - Adjusted from -0.12 to -0.09 to match potential_further
                                 # 17:09 10/1/2026 - Slightly reduce orientation to encourage random spin (prevent hide in corner))
                                 # (Also make symmetric since the stucking issue)
                                 # 21:22 11/1/2026 - Restored to -0.12 to strengthen movement signal - agent should break or learn when to move/stay
                                 # 13:50 12/01/2026 - REMOVED PENALTY: Set to 0.0. Fast rotation test showed penalties constrained exploration.
                                 #                    Let agent discover optimal rotation patterns via positive rewards + entropy, not penalties.


    'accuracy_bonus': 0.2,       # New - Reward for shooting when aimed at target (within accuracy_threshold)
    'accuracy_threshold': 30.0,  # New - Angle threshold in degrees for accuracy bonus
                                 # 11:02 12/1/2026 - Increased from 20.0 to 30.0 - agent was aiming but missing threshold by small margins
    
    # Rotation-specific stuck handling (12/1/2026)
    'stuck_thrust_penalty': -1.0,       # Penalty for thrusting when velocity < 1.0 (stuck against wall)
    'stuck_velocity_threshold': 1.0,    # Velocity threshold to consider agent "stuck"
    'escape_velocity_scale': 0.5,       # Multiplier for velocity increase when escaping stuck state
    'stuck_multiplier': 10.0,           # Amplify orientation rewards when stuck to encourage escape
    'stuck_wrong_rotation_penalty': -5.0,  # Penalty for rotating away from spawner when stuck
}

# DQN with Directional control (WASD-style)
REWARD_DQN_DIRECTIONAL = {
    # Core objectives (same as rotation)
    'destroy_enemy': 170.0,
    'destroy_spawner': 180.0, 
    'phase_complete': 500.0,
    'clean_phase_bonus': 300.0,
    'hit_enemy': 30.0,
    'hit_spawner': 5.0,
    'take_damage': -80.0,
    'death': -200.0,
    'shot_fired': -0.05,
    'existence_penalty': -0.015,

    # Navigation rewards (same as rotation)
    'potential_closer': 0.1,
    'potential_further': 0.0,
    'orientation_closer': 0.1,     # Will auto-fire based on movement direction
    'orientation_further': 0.0,

    # Accuracy (same as rotation)
    'accuracy_bonus':  0.2,
    'accuracy_threshold': 30.0,

    # Directional-specific stuck handling
    'stuck_velocity_threshold': 1.0,              # Velocity threshold to consider agent "stuck"
    'stuck_movement_penalty': -1.0,               # Penalty for moving when stuck (pushing wall)
    'stuck_direction_change_reward': 2.0,         # Reward for trying different direction when stuck
    'stuck_repeat_direction_penalty': -0.5,       # Penalty for repeating same stuck direction
}

# Default rewards (points to PPO rotational for backwards compatibility)
REWARDS = REWARD_PPO_ROTATIONAL

# Observation space size
# Player: x, y, vx, vy, velocity_magnitude, wall_proximity, angle, health (8)
# Nearest enemy: distance, cos(angle), sin(angle) (3)
# Nearest spawner: distance, cos(angle), sin(angle) (3)
# Game state: phase, enemies_count, spawners_count, can_shoot (4)
OBSERVATION_SIZE = 18

# Action spaces
ROTATION_ACTIONS = {
    0: 'nothing',
    1: 'thrust',
    2: 'rotate_left',
    3: 'rotate_right',
    4: 'shoot',
}

DIRECTIONAL_ACTIONS = {
    0: 'nothing',
    1: 'up',
    2: 'down',
    3: 'left',
    4: 'right',
    5: 'shoot',
}

# Particle effects
PARTICLE_LIFETIME = 30
PARTICLE_COUNT_EXPLOSION = 15
PARTICLE_COUNT_HIT = 5

IS_RANDOM = False
