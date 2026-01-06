"""
Arena - Main game logic and state management.
"""

import math
import random
import pygame
from typing import List, Tuple, Dict, Any

from .constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, COLORS,
    INITIAL_SPAWNERS, SPAWNERS_PER_PHASE, MAX_PHASE,
    MAX_STEPS, REWARDS, OBSERVATION_SIZE,
    PARTICLE_COUNT_EXPLOSION, PARTICLE_COUNT_HIT,
)
from .entities import Player, Enemy, Spawner, Projectile, Particle


class Arena:
    """
    Main arena class that manages all game state.
    """

    def __init__(self):
        self.player: Player = None
        self.enemies: List[Enemy] = []
        self.spawners: List[Spawner] = []
        self.projectiles: List[Projectile] = []
        self.particles: List[Particle] = []

        self.phase = 1
        self.step_count = 0
        self.score = 0
        self.done = False

        # Track events for reward calculation
        self.events: Dict[str, int] = {}

        # Initialize
        self.reset()

    def reset(self) -> None:
        """Reset the arena to initial state."""
        # Create player at center
        self.player = Player(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

        # Clear all entities
        self.enemies.clear()
        self.spawners.clear()
        self.projectiles.clear()
        self.particles.clear()

        # Reset state
        self.phase = 1
        self.step_count = 0
        self.score = 0
        self.done = False
        self.events = {}
        
        # Reset tracking for potential-based shaping
        self._prev_target_angle_diff = None
        self._prev_enemy_dist = None

        # Create initial spawners
        self._spawn_spawners(INITIAL_SPAWNERS)

    def _spawn_spawners(self, count: int) -> None:
        """Spawn new spawners at random edge positions."""
        margin = 80  # Distance from edges

        for _ in range(count):
            # Choose a random edge and position
            edge = random.choice(['top', 'bottom', 'left', 'right'])

            if edge == 'top':
                x = random.uniform(margin, SCREEN_WIDTH - margin)
                y = margin
            elif edge == 'bottom':
                x = random.uniform(margin, SCREEN_WIDTH - margin)
                y = SCREEN_HEIGHT - margin
            elif edge == 'left':
                x = margin
                y = random.uniform(margin, SCREEN_HEIGHT - margin)
            else:  # right
                x = SCREEN_WIDTH - margin
                y = random.uniform(margin, SCREEN_HEIGHT - margin)

            # Ensure not too close to player
            dist_to_player = math.sqrt(
                (x - self.player.x)**2 + (y - self.player.y)**2
            )
            if dist_to_player < 200:
                # Retry with opposite edge
                if edge == 'top':
                    y = SCREEN_HEIGHT - margin
                elif edge == 'bottom':
                    y = margin
                elif edge == 'left':
                    x = SCREEN_WIDTH - margin
                else:
                    x = margin

            self.spawners.append(Spawner(x, y))

    def _advance_phase(self) -> None:
        """Advance to the next phase."""
        self.phase += 1
        self.events['phase_complete'] = self.events.get('phase_complete', 0) + 1

        if self.phase <= MAX_PHASE:
            # Spawn more spawners for next phase
            new_spawners = INITIAL_SPAWNERS + (self.phase - 1) * SPAWNERS_PER_PHASE
            self._spawn_spawners(new_spawners)

    def update(self) -> float:
        """
        Update game state for one step.
        Returns the reward for this step.
        """
        if self.done:
            return 0.0

        self.step_count += 1
        self.events = {}  # Reset events
        reward = 0.0

        # Update player
        self.player.update()

        # Update enemies
        for enemy in self.enemies:
            enemy.update(self.player)

        # Update spawners and spawn enemies
        for spawner in self.spawners:
            if spawner.alive and spawner.update():
                new_enemy = spawner.spawn_enemy(self.phase)
                self.enemies.append(new_enemy)

        # Update projectiles
        for projectile in self.projectiles[:]:
            if not projectile.update():
                self.projectiles.remove(projectile)

        # Update particles
        for particle in self.particles[:]:
            if not particle.update():
                self.particles.remove(particle)
        # Check collisions (includes hit rewards)
        reward += self._check_collisions()

        # Check for player death
        if not self.player.alive:
            reward += REWARDS['death']
            self.events['death'] = 1
            self.done = True
            return reward

        # =====================================================================
        # AIMING REWARD SHAPING (Loose - Agent Chooses Target)
        # Rewards improving aim toward WHICHEVER target agent is best aimed at
        # This lets agent learn to prioritize one target at a time
        # =====================================================================
        
        living_spawners = [s for s in self.spawners if s.alive]
        living_enemies = [e for e in self.enemies if e.alive]
        
        # Helper to calculate angle difference to a target
        def get_angle_diff(target):
            dx = target.x - self.player.x
            dy = target.y - self.player.y
            target_angle = math.degrees(math.atan2(dy, dx))
            player_angle = self.player.angle % 360
            if player_angle > 180:
                player_angle -= 360
            diff = abs(target_angle - player_angle)
            if diff > 180:
                diff = 360 - diff
            return diff
        
        # Find the target the agent is BEST aimed at (smallest angle difference)
        # This lets the agent choose which target to focus on
        all_targets = living_spawners + living_enemies
        if all_targets:
            best_target = min(all_targets, key=get_angle_diff)
            angle_diff = get_angle_diff(best_target)
            
            # Potential-based: reward improvement in aim toward chosen target
            if self._prev_target_angle_diff is not None:
                angle_improvement = self._prev_target_angle_diff - angle_diff
                reward += angle_improvement * REWARDS['aim_improvement']
            
            self._prev_target_angle_diff = angle_diff

        # =====================================================================
        # ENEMY AVOIDANCE SHAPING (Potential-Based)
        # Rewards moving AWAY from nearest enemy, penalizes moving TOWARD
        # No hardcoded threshold - agent learns optimal distance
        # =====================================================================
        
        if living_enemies:
            nearest_enemy = min(living_enemies, key=lambda e:
                math.sqrt((e.x - self.player.x)**2 + (e.y - self.player.y)**2))
            curr_enemy_dist = math.sqrt(
                (nearest_enemy.x - self.player.x)**2 + 
                (nearest_enemy.y - self.player.y)**2
            )
            
            if self._prev_enemy_dist is not None:
                # Positive = moved away, Negative = moved closer
                distance_change = curr_enemy_dist - self._prev_enemy_dist
                reward += distance_change * REWARDS['enemy_avoidance']
            
            self._prev_enemy_dist = curr_enemy_dist
        else:
            self._prev_enemy_dist = None

        # Check for phase completion
        active_spawners = [s for s in self.spawners if s.alive]
        if len(active_spawners) == 0:
            reward += REWARDS['phase_complete']
            if self.phase >= MAX_PHASE:
                # Game won!
                self.done = True
            else:
                self._advance_phase()

        # Check max steps
        if self.step_count >= MAX_STEPS:
            self.done = True

        # Survival bonus (optional - currently disabled)
        reward += REWARDS['survival_bonus']

        return reward

    def _check_collisions(self) -> float:
        """Check all collisions and return reward."""
        reward = 0.0

        # Player projectiles vs enemies
        for projectile in self.projectiles[:]:
            if projectile.owner != 'player' or not projectile.alive:
                continue

            for enemy in self.enemies:
                if not enemy.alive:
                    continue

                dist = math.sqrt(
                    (projectile.x - enemy.x)**2 +
                    (projectile.y - enemy.y)**2
                )

                if dist < projectile.radius + enemy.radius:
                    projectile.alive = False
                    enemy.take_damage(projectile.damage)
                    self._spawn_particles(enemy.x, enemy.y, COLORS['enemy'], PARTICLE_COUNT_HIT)
                    
                    # Reward for hitting enemy (shaping)
                    reward += REWARDS['hit_enemy']
                    self.events['hit_enemy'] = self.events.get('hit_enemy', 0) + 1

                    if not enemy.alive:
                        reward += REWARDS['destroy_enemy']
                        self.events['enemy_destroyed'] = self.events.get('enemy_destroyed', 0) + 1
                        self.score += 10
                        self._spawn_particles(enemy.x, enemy.y, COLORS['enemy'], PARTICLE_COUNT_EXPLOSION)
                        # Notify spawner
                        for spawner in self.spawners:
                            if spawner.alive:
                                spawner.enemy_destroyed()
                    break

        # Player projectiles vs spawners
        for projectile in self.projectiles[:]:
            if projectile.owner != 'player' or not projectile.alive:
                continue

            for spawner in self.spawners:
                if not spawner.alive:
                    continue

                dist = math.sqrt(
                    (projectile.x - spawner.x)**2 +
                    (projectile.y - spawner.y)**2
                )

                if dist < projectile.radius + spawner.radius:
                    projectile.alive = False
                    spawner.take_damage(projectile.damage)
                    self._spawn_particles(spawner.x, spawner.y, COLORS['spawner'], PARTICLE_COUNT_HIT)
                    
                    # Reward for hitting spawner (shaping)
                    reward += REWARDS['hit_spawner']
                    self.events['hit_spawner'] = self.events.get('hit_spawner', 0) + 1

                    if not spawner.alive:
                        reward += REWARDS['destroy_spawner']
                        self.events['spawner_destroyed'] = self.events.get('spawner_destroyed', 0) + 1
                        self.score += 50
                        self._spawn_particles(spawner.x, spawner.y, COLORS['spawner'], PARTICLE_COUNT_EXPLOSION * 2)
                    break

        # Enemies vs player
        for enemy in self.enemies:
            if not enemy.alive or not self.player.alive:
                continue

            dist = math.sqrt(
                (enemy.x - self.player.x)**2 +
                (enemy.y - self.player.y)**2
            )

            if dist < enemy.radius + self.player.radius:
                if self.player.take_damage(enemy.damage):
                    reward += REWARDS['take_damage']
                    self.events['damage_taken'] = self.events.get('damage_taken', 0) + enemy.damage
                    self._spawn_particles(self.player.x, self.player.y, COLORS['player'], PARTICLE_COUNT_HIT)

                # Destroy enemy on contact
                enemy.alive = False
                self._spawn_particles(enemy.x, enemy.y, COLORS['enemy'], PARTICLE_COUNT_EXPLOSION)
                for spawner in self.spawners:
                    if spawner.alive:
                        spawner.enemy_destroyed()

        # Remove dead entities
        self.enemies = [e for e in self.enemies if e.alive]
        self.projectiles = [p for p in self.projectiles if p.alive]

        return reward

    def _spawn_particles(self, x: float, y: float, color: tuple, count: int) -> None:
        """Spawn particle effects at a location."""
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def player_shoot(self) -> bool:
        """Make player shoot. Returns True if successful."""
        projectile = self.player.shoot()
        if projectile:
            self.projectiles.append(projectile)
            return True
        return False

    def get_observation(self) -> List[float]:
        """
        Get the observation vector for RL agent.
        Returns a fixed-size vector of floats.
        
        Structure (14 values):
        - Player state: 6 values (x, y, vx, vy, angle, health)
        - Nearest enemy: 2 values (distance, relative angle)
        - Nearest spawner: 2 values (distance, relative angle)
        - Game state: 4 values (phase, enemy_count, spawner_count, can_shoot)
        """
        obs = []

        # Normalize helpers
        def norm_x(x): return x / SCREEN_WIDTH
        def norm_y(y): return y / SCREEN_HEIGHT
        max_dist = max(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        def get_relative_angle(target_x, target_y):
            """Get angle to target relative to player facing direction."""
            dx = target_x - self.player.x
            dy = target_y - self.player.y
            target_angle = math.degrees(math.atan2(dy, dx))
            # Normalize player angle to [-180, 180]
            player_angle = self.player.angle % 360
            if player_angle > 180:
                player_angle -= 360
            rel_angle = target_angle - player_angle
            # Wrap to [-180, 180]
            while rel_angle > 180:
                rel_angle -= 360
            while rel_angle < -180:
                rel_angle += 360
            return rel_angle / 180.0  # Normalize to [-1, 1]

        # Player state (6 values)
        obs.append(norm_x(self.player.x))
        obs.append(norm_y(self.player.y))
        obs.append(self.player.vx / 10.0)
        obs.append(self.player.vy / 10.0)
        obs.append(self.player.angle / 360.0)
        obs.append(self.player.health / self.player.max_health)

        # Nearest enemy (2 values: distance, relative angle)
        living_enemies = [e for e in self.enemies if e.alive]
        if living_enemies:
            nearest = min(living_enemies, key=lambda e:
                math.sqrt((e.x - self.player.x)**2 + (e.y - self.player.y)**2))
            dist = math.sqrt((nearest.x - self.player.x)**2 + (nearest.y - self.player.y)**2)
            obs.append(min(dist / max_dist, 1.0))
            obs.append(get_relative_angle(nearest.x, nearest.y))
        else:
            obs.append(1.0)  # Max distance
            obs.append(0.0)  # Neutral angle

        # Nearest spawner (2 values: distance, relative angle)
        living_spawners = [s for s in self.spawners if s.alive]
        if living_spawners:
            nearest = min(living_spawners, key=lambda s:
                math.sqrt((s.x - self.player.x)**2 + (s.y - self.player.y)**2))
            dist = math.sqrt((nearest.x - self.player.x)**2 + (nearest.y - self.player.y)**2)
            obs.append(min(dist / max_dist, 1.0))
            obs.append(get_relative_angle(nearest.x, nearest.y))
        else:
            obs.append(1.0)  # Max distance
            obs.append(0.0)  # Neutral angle

        # Game state (4 values)
        obs.append(self.phase / MAX_PHASE)
        obs.append(min(len(living_enemies) / 20.0, 1.0))
        obs.append(min(len(living_spawners) / 10.0, 1.0))
        obs.append(1.0 if self.player.can_shoot() else 0.0)

        return obs

    def render(self, screen: pygame.Surface) -> None:
        """Render the arena to the screen."""
        # Clear screen with background color
        screen.fill(COLORS['background'])

        # Draw starfield background
        self._draw_starfield(screen)

        # Draw grid lines (subtle)
        self._draw_grid(screen)

        # Draw spawners
        for spawner in self.spawners:
            spawner.draw(screen)

        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(screen)

        # Draw projectiles
        for projectile in self.projectiles:
            projectile.draw(screen)

        # Draw particles
        for particle in self.particles:
            particle.draw(screen)

        # Draw player
        self.player.draw(screen)

        # Draw UI
        self._draw_ui(screen)

    def _draw_starfield(self, screen: pygame.Surface) -> None:
        """Draw a simple starfield background."""
        random.seed(42)  # Consistent stars
        for _ in range(100):
            x = random.randint(0, SCREEN_WIDTH)
            y = random.randint(0, SCREEN_HEIGHT)
            brightness = random.randint(50, 150)
            size = random.choice([1, 1, 1, 2])
            color = (brightness, brightness, brightness + 20)
            if size == 1:
                screen.set_at((x, y), color)
            else:
                pygame.draw.circle(screen, color, (x, y), size)
        random.seed()  # Reset seed

    def _draw_grid(self, screen: pygame.Surface) -> None:
        """Draw subtle grid lines."""
        grid_color = (30, 30, 50)
        grid_spacing = 50

        for x in range(0, SCREEN_WIDTH, grid_spacing):
            pygame.draw.line(screen, grid_color, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, grid_spacing):
            pygame.draw.line(screen, grid_color, (0, y), (SCREEN_WIDTH, y))

    def _draw_ui(self, screen: pygame.Surface) -> None:
        """Draw the HUD/UI elements."""
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 28)

        # Player health bar (top left)
        bar_x, bar_y = 20, 20
        bar_width, bar_height = 200, 25

        # Background
        pygame.draw.rect(screen, COLORS['health_bar_bg'],
                        (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        # Health fill
        health_width = (self.player.health / self.player.max_health) * bar_width
        if health_width > 0:
            # Gradient effect
            health_color = COLORS['health_bar_player']
            pygame.draw.rect(screen, health_color,
                            (bar_x, bar_y, health_width, bar_height), border_radius=5)
        # Border
        pygame.draw.rect(screen, COLORS['ui_accent'],
                        (bar_x, bar_y, bar_width, bar_height), 2, border_radius=5)
        # Label
        health_text = small_font.render(f"HP: {self.player.health}/{self.player.max_health}",
                                       True, COLORS['ui_text'])
        screen.blit(health_text, (bar_x + 5, bar_y + 3))

        # Phase indicator (top center)
        phase_text = font.render(f"PHASE {self.phase}", True, COLORS['phase_indicator'])
        phase_rect = phase_text.get_rect(center=(SCREEN_WIDTH / 2, 30))
        screen.blit(phase_text, phase_rect)

        # Score (top right)
        score_text = font.render(f"Score: {self.score}", True, COLORS['ui_text'])
        score_rect = score_text.get_rect(topright=(SCREEN_WIDTH - 20, 20))
        screen.blit(score_text, score_rect)

        # Enemy/Spawner counts (below score)
        living_enemies = len([e for e in self.enemies if e.alive])
        living_spawners = len([s for s in self.spawners if s.alive])

        count_text = small_font.render(
            f"Enemies: {living_enemies}  Spawners: {living_spawners}",
            True, COLORS['ui_text']
        )
        count_rect = count_text.get_rect(topright=(SCREEN_WIDTH - 20, 55))
        screen.blit(count_text, count_rect)

        # Step counter (bottom left)
        step_text = small_font.render(f"Step: {self.step_count}/{MAX_STEPS}",
                                     True, (100, 100, 120))
        screen.blit(step_text, (20, SCREEN_HEIGHT - 30))

        # Game over message
        if self.done:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))

            if self.player.alive:
                if self.phase > MAX_PHASE:
                    msg = "VICTORY!"
                    color = (100, 255, 100)
                else:
                    msg = "TIME UP"
                    color = COLORS['ui_accent']
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)

            game_over_font = pygame.font.Font(None, 72)
            game_over_text = game_over_font.render(msg, True, color)
            game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            screen.blit(game_over_text, game_over_rect)

            final_score = font.render(f"Final Score: {self.score}", True, COLORS['ui_text'])
            final_rect = final_score.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 50))
            screen.blit(final_score, final_rect)

    def get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state."""
        return {
            'phase': self.phase,
            'score': self.score,
            'step_count': self.step_count,
            'player_health': self.player.health,
            'enemies_alive': len([e for e in self.enemies if e.alive]),
            'spawners_alive': len([s for s in self.spawners if s.alive]),
            'events': self.events.copy(),
        }
