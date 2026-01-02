"""
Game entities: Player, Enemy, Spawner, Projectile, Particle.
"""

import math
import random
import pygame
from .constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, COLORS,
    PLAYER_RADIUS, PLAYER_MAX_HEALTH, PLAYER_SPEED, PLAYER_THRUST,
    PLAYER_MAX_VELOCITY, PLAYER_ROTATION_SPEED, PLAYER_FRICTION,
    PLAYER_SHOOT_COOLDOWN, PLAYER_INVINCIBILITY_FRAMES,
    PROJECTILE_SPEED, PROJECTILE_RADIUS, PROJECTILE_LIFETIME,
    ENEMY_RADIUS, ENEMY_HEALTH, ENEMY_SPEED, ENEMY_DAMAGE,
    SPAWNER_RADIUS, SPAWNER_HEALTH, SPAWNER_SPAWN_INTERVAL, SPAWNER_MAX_ENEMIES,
    PARTICLE_LIFETIME,
)


class Particle:
    """Visual particle effect for explosions and hits."""

    def __init__(self, x: float, y: float, color: tuple):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-3, 3)
        self.lifetime = PARTICLE_LIFETIME
        self.max_lifetime = PARTICLE_LIFETIME
        self.radius = random.uniform(2, 5)

    def update(self) -> bool:
        """Update particle. Returns True if still alive."""
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.95
        self.vy *= 0.95
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, screen: pygame.Surface):
        """Draw particle with fade effect."""
        alpha = self.lifetime / self.max_lifetime
        radius = int(self.radius * alpha)
        if radius > 0:
            color = tuple(int(c * alpha) for c in self.color)
            pygame.draw.circle(screen, color, (int(self.x), int(self.y)), radius)


class Player:
    """Player ship entity."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = -90  # Pointing up initially
        self.health = PLAYER_MAX_HEALTH
        self.max_health = PLAYER_MAX_HEALTH
        self.radius = PLAYER_RADIUS
        self.shoot_cooldown = 0
        self.invincibility = 0
        self.alive = True

    def reset(self, x: float, y: float):
        """Reset player to initial state."""
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = -90
        self.health = PLAYER_MAX_HEALTH
        self.shoot_cooldown = 0
        self.invincibility = 0
        self.alive = True

    def apply_thrust(self):
        """Apply forward thrust based on current angle."""
        rad = math.radians(self.angle)
        self.vx += PLAYER_THRUST * math.cos(rad)
        self.vy += PLAYER_THRUST * math.sin(rad)
        # Limit velocity
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > PLAYER_MAX_VELOCITY:
            self.vx = (self.vx / speed) * PLAYER_MAX_VELOCITY
            self.vy = (self.vy / speed) * PLAYER_MAX_VELOCITY

    def rotate_left(self):
        """Rotate counter-clockwise."""
        self.angle -= PLAYER_ROTATION_SPEED

    def rotate_right(self):
        """Rotate clockwise."""
        self.angle += PLAYER_ROTATION_SPEED

    def move_direction(self, dx: float, dy: float):
        """Move in a specific direction (for directional control)."""
        self.vx = dx * PLAYER_SPEED
        self.vy = dy * PLAYER_SPEED
        # Update angle to face movement direction
        if dx != 0 or dy != 0:
            self.angle = math.degrees(math.atan2(dy, dx))

    def update(self):
        """Update player position and state."""
        # Apply friction (for rotation mode)
        self.vx *= PLAYER_FRICTION
        self.vy *= PLAYER_FRICTION

        # Update position
        self.x += self.vx
        self.y += self.vy

        # Keep within bounds
        self.x = max(self.radius, min(SCREEN_WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(SCREEN_HEIGHT - self.radius, self.y))

        # Update cooldowns
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.invincibility > 0:
            self.invincibility -= 1

    def can_shoot(self) -> bool:
        """Check if player can shoot."""
        return self.shoot_cooldown == 0

    def shoot(self) -> 'Projectile':
        """Create a projectile. Returns None if on cooldown."""
        if not self.can_shoot():
            return None
        self.shoot_cooldown = PLAYER_SHOOT_COOLDOWN
        rad = math.radians(self.angle)
        # Spawn projectile slightly in front of player
        spawn_x = self.x + self.radius * 1.5 * math.cos(rad)
        spawn_y = self.y + self.radius * 1.5 * math.sin(rad)
        return Projectile(spawn_x, spawn_y, self.angle, owner='player')

    def take_damage(self, amount: int) -> bool:
        """Take damage. Returns True if damage was applied."""
        if self.invincibility > 0:
            return False
        self.health -= amount
        self.invincibility = PLAYER_INVINCIBILITY_FRAMES
        if self.health <= 0:
            self.health = 0
            self.alive = False
        return True

    def draw(self, screen: pygame.Surface):
        """Draw player ship with glow effect."""
        # Draw glow
        glow_radius = self.radius + 5
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        for i in range(3):
            alpha = 50 - i * 15
            r = glow_radius - i * 2
            color = (*COLORS['player_glow'], alpha)
            pygame.draw.circle(glow_surface, color, (glow_radius, glow_radius), r)
        screen.blit(glow_surface, (int(self.x - glow_radius), int(self.y - glow_radius)))

        # Flash when invincible
        if self.invincibility > 0 and self.invincibility % 6 < 3:
            color = (255, 255, 255)
        else:
            color = COLORS['player']

        # Draw triangular ship
        rad = math.radians(self.angle)
        points = []
        # Front point
        points.append((
            self.x + self.radius * math.cos(rad),
            self.y + self.radius * math.sin(rad)
        ))
        # Back left
        points.append((
            self.x + self.radius * 0.7 * math.cos(rad + 2.5),
            self.y + self.radius * 0.7 * math.sin(rad + 2.5)
        ))
        # Back center (indent)
        points.append((
            self.x + self.radius * 0.3 * math.cos(rad + math.pi),
            self.y + self.radius * 0.3 * math.sin(rad + math.pi)
        ))
        # Back right
        points.append((
            self.x + self.radius * 0.7 * math.cos(rad - 2.5),
            self.y + self.radius * 0.7 * math.sin(rad - 2.5)
        ))

        pygame.draw.polygon(screen, color, points)
        pygame.draw.polygon(screen, (255, 255, 255), points, 2)


class Enemy:
    """Enemy entity that chases the player."""

    def __init__(self, x: float, y: float, health: int = None, speed: float = None):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.health = health if health else ENEMY_HEALTH
        self.max_health = self.health
        self.speed = speed if speed else ENEMY_SPEED
        self.radius = ENEMY_RADIUS
        self.damage = ENEMY_DAMAGE
        self.alive = True

    def update(self, player: Player):
        """Update enemy - move toward player."""
        if not self.alive:
            return

        # Calculate direction to player
        dx = player.x - self.x
        dy = player.y - self.y
        dist = math.sqrt(dx**2 + dy**2)

        if dist > 0:
            self.vx = (dx / dist) * self.speed
            self.vy = (dy / dist) * self.speed

        self.x += self.vx
        self.y += self.vy

        # Keep within bounds
        self.x = max(self.radius, min(SCREEN_WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(SCREEN_HEIGHT - self.radius, self.y))

    def take_damage(self, amount: int):
        """Take damage from projectile."""
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.alive = False

    def draw(self, screen: pygame.Surface):
        """Draw enemy with glow effect."""
        if not self.alive:
            return

        # Draw glow
        glow_radius = self.radius + 4
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        for i in range(3):
            alpha = 40 - i * 12
            r = glow_radius - i * 2
            color = (*COLORS['enemy_glow'], alpha)
            pygame.draw.circle(glow_surface, color, (glow_radius, glow_radius), r)
        screen.blit(glow_surface, (int(self.x - glow_radius), int(self.y - glow_radius)))

        # Draw enemy (diamond shape)
        points = [
            (self.x, self.y - self.radius),
            (self.x + self.radius, self.y),
            (self.x, self.y + self.radius),
            (self.x - self.radius, self.y),
        ]
        pygame.draw.polygon(screen, COLORS['enemy'], points)
        pygame.draw.polygon(screen, (255, 150, 150), points, 2)

        # Draw health bar
        self._draw_health_bar(screen)

    def _draw_health_bar(self, screen: pygame.Surface):
        """Draw health bar above enemy."""
        bar_width = self.radius * 2
        bar_height = 4
        bar_x = self.x - bar_width / 2
        bar_y = self.y - self.radius - 10

        # Background
        pygame.draw.rect(screen, COLORS['health_bar_bg'],
                        (bar_x, bar_y, bar_width, bar_height))
        # Health
        health_width = (self.health / self.max_health) * bar_width
        pygame.draw.rect(screen, COLORS['health_bar_enemy'],
                        (bar_x, bar_y, health_width, bar_height))


class Spawner:
    """Enemy spawner that periodically creates enemies."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.health = SPAWNER_HEALTH
        self.max_health = SPAWNER_HEALTH
        self.radius = SPAWNER_RADIUS
        self.spawn_timer = SPAWNER_SPAWN_INTERVAL
        self.spawn_interval = SPAWNER_SPAWN_INTERVAL
        self.active_enemies = 0
        self.max_enemies = SPAWNER_MAX_ENEMIES
        self.alive = True
        self.rotation = 0  # For visual effect

    def update(self) -> bool:
        """Update spawner. Returns True if should spawn an enemy."""
        if not self.alive:
            return False

        self.rotation += 1  # Visual rotation

        self.spawn_timer -= 1
        if self.spawn_timer <= 0 and self.active_enemies < self.max_enemies:
            self.spawn_timer = self.spawn_interval
            return True
        return False

    def spawn_enemy(self, phase: int) -> Enemy:
        """Create an enemy at spawner location."""
        from .constants import PHASE_ENEMY_HEALTH_MULT, PHASE_ENEMY_SPEED_MULT

        # Spawn at edge of spawner
        angle = random.uniform(0, 2 * math.pi)
        spawn_x = self.x + self.radius * math.cos(angle)
        spawn_y = self.y + self.radius * math.sin(angle)

        # Scale stats by phase
        health = int(ENEMY_HEALTH * (PHASE_ENEMY_HEALTH_MULT ** (phase - 1)))
        speed = ENEMY_SPEED * (PHASE_ENEMY_SPEED_MULT ** (phase - 1))

        self.active_enemies += 1
        return Enemy(spawn_x, spawn_y, health, speed)

    def enemy_destroyed(self):
        """Called when an enemy from this spawner is destroyed."""
        self.active_enemies = max(0, self.active_enemies - 1)

    def take_damage(self, amount: int):
        """Take damage from projectile."""
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.alive = False

    def draw(self, screen: pygame.Surface):
        """Draw spawner with rotating glow effect."""
        if not self.alive:
            return

        # Draw pulsing glow
        pulse = abs(math.sin(self.rotation * 0.05)) * 0.5 + 0.5
        glow_radius = int(self.radius + 8 + pulse * 5)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        for i in range(4):
            alpha = int((50 - i * 12) * pulse)
            r = glow_radius - i * 3
            color = (*COLORS['spawner_glow'], alpha)
            pygame.draw.circle(glow_surface, color, (glow_radius, glow_radius), r)
        screen.blit(glow_surface, (int(self.x - glow_radius), int(self.y - glow_radius)))

        # Draw hexagonal spawner
        points = []
        for i in range(6):
            angle = math.radians(self.rotation + i * 60)
            points.append((
                self.x + self.radius * math.cos(angle),
                self.y + self.radius * math.sin(angle)
            ))
        pygame.draw.polygon(screen, COLORS['spawner'], points)
        pygame.draw.polygon(screen, (255, 220, 150), points, 3)

        # Draw inner detail
        inner_points = []
        for i in range(6):
            angle = math.radians(-self.rotation * 0.5 + i * 60 + 30)
            inner_points.append((
                self.x + self.radius * 0.5 * math.cos(angle),
                self.y + self.radius * 0.5 * math.sin(angle)
            ))
        pygame.draw.polygon(screen, COLORS['spawner_glow'], inner_points)

        # Draw health bar
        self._draw_health_bar(screen)

    def _draw_health_bar(self, screen: pygame.Surface):
        """Draw health bar above spawner."""
        bar_width = self.radius * 2
        bar_height = 6
        bar_x = self.x - bar_width / 2
        bar_y = self.y - self.radius - 15

        # Background
        pygame.draw.rect(screen, COLORS['health_bar_bg'],
                        (bar_x, bar_y, bar_width, bar_height))
        # Health
        health_width = (self.health / self.max_health) * bar_width
        pygame.draw.rect(screen, COLORS['health_bar_spawner'],
                        (bar_x, bar_y, health_width, bar_height))


class Projectile:
    """Projectile fired by player or enemy."""

    def __init__(self, x: float, y: float, angle: float, owner: str = 'player'):
        self.x = x
        self.y = y
        rad = math.radians(angle)
        self.vx = PROJECTILE_SPEED * math.cos(rad)
        self.vy = PROJECTILE_SPEED * math.sin(rad)
        self.radius = PROJECTILE_RADIUS
        self.owner = owner  # 'player' or 'enemy'
        self.lifetime = PROJECTILE_LIFETIME
        self.alive = True
        self.damage = 10 if owner == 'player' else 5

    def update(self) -> bool:
        """Update projectile. Returns True if still alive."""
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1

        # Check bounds
        if (self.x < 0 or self.x > SCREEN_WIDTH or
            self.y < 0 or self.y > SCREEN_HEIGHT):
            self.alive = False

        if self.lifetime <= 0:
            self.alive = False

        return self.alive

    def draw(self, screen: pygame.Surface):
        """Draw projectile with glow."""
        if not self.alive:
            return

        color = COLORS['projectile_player'] if self.owner == 'player' else COLORS['projectile_enemy']

        # Draw glow
        glow_radius = self.radius + 4
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        for i in range(3):
            alpha = 60 - i * 20
            r = glow_radius - i
            glow_color = (*color, alpha)
            pygame.draw.circle(glow_surface, glow_color, (glow_radius, glow_radius), r)
        screen.blit(glow_surface, (int(self.x - glow_radius), int(self.y - glow_radius)))

        # Draw core
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), self.radius - 2)
