#!/usr/bin/env python3
"""
Human playable version of the arena game.
Use this to test game mechanics manually.

Controls:
    Rotation Mode (default):
        W or UP:     Thrust forward
        A or LEFT:   Rotate left
        D or RIGHT:  Rotate right
        SPACE:       Shoot

    Directional Mode (--directional):
        W or UP:     Move up
        S or DOWN:   Move down
        A or LEFT:   Move left
        D or RIGHT:  Move right
        SPACE:       Shoot

    General:
        R:           Reset game
        ESC or Q:    Quit
        TAB:         Toggle control mode
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pygame

from game.arena import Arena
from game.constants import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, COLORS


def main():
    parser = argparse.ArgumentParser(description='Play the arena game manually')
    parser.add_argument('--directional', '-d', action='store_true',
                       help='Use directional controls instead of rotation')
    args = parser.parse_args()

    # Initialize pygame
    pygame.init()
    pygame.display.set_caption("Deep RL Arena - Human Play")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # Create arena
    arena = Arena()

    # Control mode
    use_directional = args.directional

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    arena.reset()
                elif event.key == pygame.K_TAB:
                    use_directional = not use_directional
                    print(f"Switched to {'Directional' if use_directional else 'Rotation'} mode")

        # Handle continuous key presses
        if not arena.done:
            keys = pygame.key.get_pressed()

            if use_directional:
                # Directional controls
                dx, dy = 0, 0
                if keys[pygame.K_w] or keys[pygame.K_UP]:
                    dy = -1
                if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                    dy = 1
                if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                    dx = -1
                if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                    dx = 1

                arena.player.move_direction(dx, dy)

                if keys[pygame.K_SPACE]:
                    arena.player_shoot()
            else:
                # Rotation controls
                if keys[pygame.K_w] or keys[pygame.K_UP]:
                    arena.player.apply_thrust()
                if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                    arena.player.rotate_left()
                if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                    arena.player.rotate_right()
                if keys[pygame.K_SPACE]:
                    arena.player_shoot()

        # Update game
        if not arena.done:
            arena.update()

        # Render
        arena.render(screen)

        # Draw control mode indicator
        mode_text = f"Mode: {'Directional' if use_directional else 'Rotation'} (TAB to switch)"
        mode_surface = font.render(mode_text, True, COLORS['ui_accent'])
        screen.blit(mode_surface, (20, SCREEN_HEIGHT - 55))

        # Draw controls help
        if use_directional:
            controls = "WASD/Arrows: Move | SPACE: Shoot | R: Reset | ESC: Quit"
        else:
            controls = "W/UP: Thrust | A/D: Rotate | SPACE: Shoot | R: Reset | ESC: Quit"
        controls_surface = font.render(controls, True, (100, 100, 120))
        screen.blit(controls_surface, (SCREEN_WIDTH // 2 - controls_surface.get_width() // 2,
                                       SCREEN_HEIGHT - 25))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == '__main__':
    main()
