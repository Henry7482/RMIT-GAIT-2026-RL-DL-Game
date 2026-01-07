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
PLAYER_ROTATION_SPEED = 5.0  # Degrees per frame (increased for better tracking)
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
SPAWNER_RADIUS = 35
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

# Reward values
REWARDS = {
    # Core rewards (sparse, clear signal)
    'destroy_enemy': 50.0,
    'destroy_spawner': 200.0,
    'phase_complete': 400.0,
    'hit_enemy': 10.0,
    'hit_spawner': 25.0,
    # Penalties
    'take_damage': -40.0,
    'death': -100.0,
    'survival_bonus': -0.5,
    # Simple aiming feedback (within 20 degrees of target)
    'aimed_at_target': 0.05,
    'aim_threshold': 20.0,
    # Movement incentives
    'stationary_speed_threshold': 0.5,
    'stationary_penalty': -0.03,
}

# Observation space size (16 values)
# Player: x, y, vx, vy, angle, health (6)
# Nearest enemy: distance, angle, health_ratio (3)
# Nearest spawner: distance, angle, health_ratio (3)
# Game state: phase, enemies_count, spawners_count, can_shoot (4)
OBSERVATION_SIZE = 16

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
