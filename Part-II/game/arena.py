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
    MAX_STEPS, OBSERVATION_SIZE,
    REWARD_PPO_ROTATIONAL, REWARD_DQN_DIRECTIONAL,
    PARTICLE_COUNT_EXPLOSION, PARTICLE_COUNT_HIT,
    ENEMY_HEALTH, PHASE_ENEMY_HEALTH_MULT,
    PLAYER_MAX_VELOCITY,
    IS_RANDOM
)
from .entities import Player, Enemy, Spawner, Projectile, Particle


class Arena:
    """
    Main arena class that manages all game state.
    """

    def __init__(self, env_type: str = 'rotation'):
        self.env_type = env_type  # 'rotation' or 'directional'

        # Select appropriate reward dictionary based on environment type
        if env_type == 'rotation':
            self.rewards = REWARD_PPO_ROTATIONAL
        elif env_type == 'directional':
            self.rewards = REWARD_DQN_DIRECTIONAL
        else:
            raise ValueError(f"Unknown env_type: {env_type}")

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

        # Track previous distance to nearest spawner for potential-based reward
        self.previous_spawner_distance = None

        # Track previous angle to nearest spawner for orientation reward
        self.previous_spawner_angle = None

        # Track previous velocity for stuck detection and escape rewards
        self.previous_velocity = 0.0

        # Track last rotation action for stuck rotation commitment (0=none, 2=left, 3=right)
        self.last_rotation_action = 0

        # Track last movement action for directional stuck handling (0=none, 1=up, 2=down, 3=left, 4=right)
        self.last_movement_action = 0

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
        self.previous_spawner_distance = None
        self.previous_spawner_angle = None
        self.previous_velocity = 0.0
        self.last_rotation_action = 0
        self.last_movement_action = 0

        # Create initial spawners
        rng = random if IS_RANDOM else random.Random(self.phase)
        self._spawn_spawners(INITIAL_SPAWNERS, rng)

    def _spawn_spawners(self, count: int, rng) -> None:
        """Spawn new spawners at random edge positions."""
        margin = 80  # Distance from edges

        for _ in range(count):
            # Choose a random edge and position
            edge = rng.choice(['top', 'bottom', 'left', 'right'])

            if edge == 'top':
                x = rng.uniform(margin, SCREEN_WIDTH - margin)
                y = margin
            elif edge == 'bottom':
                x = rng.uniform(margin, SCREEN_WIDTH - margin)
                y = SCREEN_HEIGHT - margin
            elif edge == 'left':
                x = margin
                y = rng.uniform(margin, SCREEN_HEIGHT - margin)
            else:  # right
                x = SCREEN_WIDTH - margin
                y = rng.uniform(margin, SCREEN_HEIGHT - margin)

            # Ensure not too close to player

            # dist_to_player = math.sqrt(
            #     (x - self.player.x)**2 + (y - self.player.y)**2
            # )
            # if dist_to_player < 200:
            #     # Retry with opposite edge
            #     if edge == 'top':
            #         y = SCREEN_HEIGHT - margin
            #     elif edge == 'bottom':
            #         y = margin
            #     elif edge == 'left':
            #         x = SCREEN_WIDTH - margin
            #     else:
            #         x = margin

            self.spawners.append(Spawner(x, y))

    def _advance_phase(self) -> None:
        """Advance to the next phase."""
        self.phase += 1
        self.events['phase_complete'] = self.events.get('phase_complete', 0) + 1

        if self.phase <= MAX_PHASE:
            # Spawn more spawners for next phase
            new_spawners = INITIAL_SPAWNERS + (self.phase - 1) * SPAWNERS_PER_PHASE
            rng = random if IS_RANDOM else random.Random(self.phase)
            self._spawn_spawners(new_spawners, rng)

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

        # REWARD SHAPING: Discourage shooting
        if self.events.get('shot_fired', 0) > 0:
            reward += self.rewards['shot_fired']
            
            # ACCURACY BONUS: Reward shooting when aimed at target
            living_spawners = [s for s in self.spawners if s.alive]
            if living_spawners:
                nearest_spawner = min(living_spawners, key=lambda s:
                    math.sqrt((s.x - self.player.x)**2 + (s.y - self.player.y)**2))
                dx = nearest_spawner.x - self.player.x
                dy = nearest_spawner.y - self.player.y
                angle_to_spawner = math.degrees(math.atan2(dy, dx))
                relative_angle = angle_to_spawner - self.player.angle
                # Normalize to [-180, 180]
                while relative_angle > 180:
                    relative_angle -= 360
                while relative_angle < -180:
                    relative_angle += 360
                
                # Grant accuracy bonus if aimed at target
                if abs(relative_angle) < self.rewards['accuracy_threshold']:
                    reward += self.rewards['accuracy_bonus']

        # REWARD SHAPING: Simple potential-based reward for approaching spawners
        living_spawners = [s for s in self.spawners if s.alive]
        if len(living_spawners) > 0:
            nearest_spawner = min(living_spawners, key=lambda s:
                math.sqrt((s.x - self.player.x)**2 + (s.y - self.player.y)**2))
            current_dist = math.sqrt(
                (nearest_spawner.x - self.player.x)**2 + 
                (nearest_spawner.y - self.player.y)**2
            )
            
            # Calculate angle to spawner for orientation reward
            dx = nearest_spawner.x - self.player.x
            dy = nearest_spawner.y - self.player.y
            angle_to_spawner = math.degrees(math.atan2(dy, dx))
            relative_angle = angle_to_spawner - self.player.angle
            # Normalize to [-180, 180]
            while relative_angle > 180:
                relative_angle -= 360
            while relative_angle < -180:
                relative_angle += 360
            
            current_angle_diff = abs(relative_angle)
            
            # Potential-based reward: reward getting closer, punish getting further
            if self.previous_spawner_distance is not None:
                if current_dist < self.previous_spawner_distance:
                    reward += self.rewards['potential_closer']
                else:
                    reward += self.rewards['potential_further']
            
            self.previous_spawner_distance = current_dist
            
            # === ORIENTATION REWARDS ===
            # Normal: Guide rotation toward spawner
            # Stuck: Override - reward ANY rotation, penalize direction changes
            if self.previous_spawner_angle is not None:
                orientation_diff = current_angle_diff - self.previous_spawner_angle
                
                # Check if stuck (low velocity)
                velocity_magnitude = math.sqrt(self.player.vx**2 + self.player.vy**2)
                is_stuck = velocity_magnitude < self.rewards['stuck_velocity_threshold']
                
                if not is_stuck:
                    # NORMAL: Guide toward spawner
                    if orientation_diff < 0:  # Rotating toward spawner
                        reward += self.rewards['orientation_closer']  # 0.1
                    elif orientation_diff > 0:  # Rotating away from spawner
                        reward += self.rewards['orientation_further']  # 0.0
                else:
                    # STUCK: Override orientation logic based on env_type
                    current_action = getattr(self, 'current_action', 0)

                    if self.env_type == 'rotation':
                        # Rotation mode: Reward ANY rotation, penalize direction changes
                        if current_action in [2, 3]:  # Any rotation action
                            reward += 5.0  # Reward rotating when stuck

                            # Penalize switching rotation direction
                            if self.last_rotation_action != 0 and current_action != self.last_rotation_action:
                                reward += self.rewards['stuck_wrong_rotation_penalty']  # -5.0

                            self.last_rotation_action = current_action

                    elif self.env_type == 'directional':
                        # Directional mode: Reward direction changes, penalize repeating
                        if current_action in [1, 2, 3, 4]:  # Any movement action
                            reward += self.rewards['stuck_movement_penalty']  # -1.0 (still stuck)

                            # Reward trying different direction
                            if self.last_movement_action != 0 and current_action != self.last_movement_action:
                                reward += self.rewards['stuck_direction_change_reward']  # +2.0
                            else:
                                reward += self.rewards['stuck_repeat_direction_penalty']  # -0.5

                            self.last_movement_action = current_action
            
            self.previous_spawner_angle = current_angle_diff

        # Check collisions
        reward += self._check_collisions()

        # Check for player death
        if not self.player.alive:
            reward += self.rewards['death']
            self.events['death'] = 1
            self.done = True
            return reward

        # Check for phase completion
        active_spawners = [s for s in self.spawners if s.alive]
        if len(active_spawners) == 0:
            reward += self.rewards['phase_complete']
            
            # Bonus for clearing all enemies before phase ends
            living_enemies = [e for e in self.enemies if e.alive]
            if len(living_enemies) == 0:
                reward += self.rewards['clean_phase_bonus']
            
            if self.phase >= MAX_PHASE:
                # Game won!
                self.done = True
            else:
                self._advance_phase()

        # Check max steps
        if self.step_count >= MAX_STEPS:
            self.done = True

        # Time pressure penalty to encourage faster completion
        reward += self.rewards['existence_penalty']

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
                    
                    # REWARD FOR HITTING (even if doesn't kill)
                    reward += self.rewards['hit_enemy']
                    
                    enemy.take_damage(projectile.damage)
                    self._spawn_particles(enemy.x, enemy.y, COLORS['enemy'], PARTICLE_COUNT_HIT)

                    if not enemy.alive:
                        reward += self.rewards['destroy_enemy']
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
                    reward += self.rewards['hit_spawner']

                    if not spawner.alive:
                        reward += self.rewards['destroy_spawner']
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
                    reward += self.rewards['take_damage']
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

        # Track Shooting Events
        if projectile:
            self.projectiles.append(projectile)
            self.events['shot_fired'] = self.events.get('shot_fired', 0) + 1
            return True
        return False

    def get_observation(self) -> List[float]:
        """
        Get the observation vector for RL agent.
        Returns a fixed-size vector of floats.
        """
        obs = []

        # Normalize helper
        def norm_x(x): return x / SCREEN_WIDTH
        def norm_y(y): return y / SCREEN_HEIGHT

        # Player state (7 values)
        obs.append(norm_x(self.player.x))
        obs.append(norm_y(self.player.y))
        obs.append(self.player.vx / 10.0)  # Normalize velocity
        obs.append(self.player.vy / 10.0)
        
        # Velocity magnitude - helps agent detect being stuck against walls
        # RATIONALE: Agent gets stuck sprinting into walls (thrust action but velocity ~0).
        # Without this, agent can't distinguish "moving toward spawner" vs "stuck against wall."
        # With velocity magnitude, agent learns: "if velocity low + thrusting + spawner far → stuck, rotate away"
        velocity_magnitude = math.sqrt(self.player.vx**2 + self.player.vy**2)
        obs.append(velocity_magnitude / PLAYER_MAX_VELOCITY)

        # Wall proximity - distance to nearest wall (normalized)
        # RATIONALE: Agent needs to see walls to avoid hitting them. Without this,
        # agent only learns "I'm stuck" after already stuck, which is too late.
        # With wall distance, agent learns: "wall close + thrusting toward wall = will get stuck = bad"
        wall_dist = min(
            self.player.x,  # Distance to left wall
            SCREEN_WIDTH - self.player.x,  # Distance to right wall
            self.player.y,  # Distance to top wall
            SCREEN_HEIGHT - self.player.y  # Distance to bottom wall
        )
        obs.append(wall_dist / max(SCREEN_WIDTH, SCREEN_HEIGHT))

        obs.append(self.player.angle / 360.0)  # Normalize angle
        obs.append(self.player.health / self.player.max_health)

        # Nearest enemy (3 values: distance, cos(angle), sin(angle))
        nearest_enemy_dist = 1.0
        nearest_enemy_cos = 0.0  # Default forward
        nearest_enemy_sin = 0.0
        living_enemies = [e for e in self.enemies if e.alive]

        if living_enemies:
            nearest = min(living_enemies, key=lambda e:
                math.sqrt((e.x - self.player.x)**2 + (e.y - self.player.y)**2))
            dx = nearest.x - self.player.x
            dy = nearest.y - self.player.y
            dist = math.sqrt(dx**2 + dy**2)
            nearest_enemy_dist = min(dist / max(SCREEN_WIDTH, SCREEN_HEIGHT), 1.0)
            # Calculate relative angle and convert to sin/cos components
            angle_to_enemy = math.degrees(math.atan2(dy, dx))
            relative_angle = angle_to_enemy - self.player.angle
            rel_angle_rad = math.radians(relative_angle)
            nearest_enemy_cos = math.cos(rel_angle_rad)
            nearest_enemy_sin = math.sin(rel_angle_rad)

        obs.append(nearest_enemy_dist)
        obs.append(nearest_enemy_cos)
        obs.append(nearest_enemy_sin)
        # obs.append(nearest_enemy_health)  # COMMENTED OUT - removed health tracking

        # # Second nearest enemy (4 values: distance, cos(angle), sin(angle), health)
        # # COMMENTED OUT - simplified observation space to 16D
        # # RATIONALE: Agent needs to track multiple threats to make informed decisions about
        # # which enemy to engage next. Without this, agent only sees closest enemy and may
        # # ignore a second enemy that's nearly as close, leading to suboptimal target switching.
        # # This enables learning: "finish current target if second enemy is far, but switch
        # # targets if multiple enemies are clustered together."
        # second_enemy_dist = 1.0
        # second_enemy_cos = 0.0
        # second_enemy_sin = 0.0
        # second_enemy_health = 1.0  # NEW

        # if len(living_enemies) >= 2:
        #     # Sort enemies by distance to get second nearest
        #     sorted_enemies = sorted(living_enemies,
        #                            key=lambda e: math.sqrt((e.x - self.player.x)**2 + (e.y - self.player.y)**2))
        #     second = sorted_enemies[1]
        #     dx = second.x - self.player.x
        #     dy = second.y - self.player.y
        #     dist = math.sqrt(dx**2 + dy**2)
        #     second_enemy_dist = min(dist / max(SCREEN_WIDTH, SCREEN_HEIGHT), 1.0)
        #     # Calculate relative angle and convert to sin/cos components
        #     angle_to_enemy = math.degrees(math.atan2(dy, dx))
        #     relative_angle = angle_to_enemy - self.player.angle
        #     rel_angle_rad = math.radians(relative_angle)
        #     second_enemy_cos = math.cos(rel_angle_rad)
        #     second_enemy_sin = math.sin(rel_angle_rad)

        #     # NEW: Health info
        #     # Use int() to match the actual enemy creation in spawn_enemy()
        #     max_enemy_health = int(ENEMY_HEALTH * (PHASE_ENEMY_HEALTH_MULT ** (MAX_PHASE - 1)))
        #     second_enemy_health = second.health / max_enemy_health

        # obs.append(second_enemy_dist)
        # obs.append(second_enemy_cos)
        # obs.append(second_enemy_sin)
        # obs.append(second_enemy_health)  # NEW

        # Nearest spawner (3 values: distance, cos(angle), sin(angle))
        nearest_spawner_dist = 1.0
        nearest_spawner_cos = 0.0  # Default forward
        nearest_spawner_sin = 0.0
        living_spawners = [s for s in self.spawners if s.alive]

        if living_spawners:
            nearest = min(living_spawners, key=lambda s:
                math.sqrt((s.x - self.player.x)**2 + (s.y - self.player.y)**2))
            dx = nearest.x - self.player.x
            dy = nearest.y - self.player.y
            dist = math.sqrt(dx**2 + dy**2)
            nearest_spawner_dist = min(dist / max(SCREEN_WIDTH, SCREEN_HEIGHT), 1.0)
            # Calculate relative angle and convert to sin/cos components
            angle_to_spawner = math.degrees(math.atan2(dy, dx))
            relative_angle = angle_to_spawner - self.player.angle
            rel_angle_rad = math.radians(relative_angle)
            nearest_spawner_cos = math.cos(rel_angle_rad)
            nearest_spawner_sin = math.sin(rel_angle_rad)

        obs.append(nearest_spawner_dist)
        obs.append(nearest_spawner_cos)
        obs.append(nearest_spawner_sin)

        # # Enemies near target spawner (1 value: normalized count)
        # # COMMENTED OUT - simplified observation space to 16D
        # # RATIONALE: Agent needs to understand spatial clustering of threats around objectives.
        # # Without this, agent sees "spawner at distance X" but doesn't know if path is clear
        # # or heavily defended. This enables learning: "if spawner has 3+ enemies nearby,
        # # clear the defenders first before engaging spawner" vs "if spawner is exposed, rush it directly."
        # # This is strategic area control information.
        # enemies_near_spawner_count = 0.0

        # if living_spawners:
        #     # Count enemies within 300 units of the nearest spawner
        #     enemies_near_target = sum(1 for e in living_enemies
        #                              if math.sqrt((e.x - nearest.x)**2 + (e.y - nearest.y)**2) < 300)
        #     enemies_near_spawner_count = min(enemies_near_target / 10.0, 1.0)

        # obs.append(enemies_near_spawner_count)

        # Game state (4 values)
        obs.append(self.phase / MAX_PHASE)
        obs.append(min(len(living_enemies) / 20.0, 1.0))  # Normalized enemy count
        obs.append(min(len(living_spawners) / 10.0, 1.0))  # Normalized spawner count
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
