import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pygame

from config import DEFAULT_CFG, load_config
from levels import ALGO_NAMES, LEVELS, LEVEL_ORDER, LevelSpec, WIDTH_TILES, HEIGHT_TILES

CFG0 = load_config(0)
random.seed(int(CFG0["seed"]))

# -----------------------------
# Pygame window
# -----------------------------
TILE_SIZE = int(CFG0["tileSize"])
WIDTH, HEIGHT = WIDTH_TILES * TILE_SIZE, HEIGHT_TILES * TILE_SIZE
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)
pygame.display.set_caption("GridWorld Reinforcement Learning")

# Colors
COL_BG = (25, 28, 34)
COL_GRID = (45, 50, 58)
COL_AGENT = (74, 222, 128)
COL_APPLE = (252, 92, 101)
COL_TEXT = (240, 240, 240)
COL_ROCK = (90, 96, 112)
COL_FIRE = (255, 94, 0)
COL_KEY = (255, 214, 10)
COL_CHEST = (233, 196, 106)
COL_CHEST_OPEN = (173, 232, 153)
COL_MONSTER = (151, 117, 250)

# Actions: 0 up, 1 right, 2 down, 3 left
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
A_UP, A_RIGHT, A_DOWN, A_LEFT = 0, 1, 2, 3
ALL_ACTIONS = [A_UP, A_RIGHT, A_DOWN, A_LEFT]


def log_episode(level_id: int, algo: str, ep: int, env_return: float, total_return: float, steps: int, out_dir: Path | None = None) -> None:
    """Append episode returns to a CSV file for plotting."""
    base_dir = Path(__file__).parent
    log_dir = out_dir if out_dir is not None else base_dir / "logs"
    Path(log_dir).mkdir(exist_ok=True)
    path = log_dir / f"level{level_id}_{algo}.csv"
    if not path.exists():
        path.write_text("episode,env_return,total_return,steps\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{ep},{env_return},{total_return},{steps}\n")


# -----------------------------
# Environment
# -----------------------------
@dataclass
class StepResult:
    next_state: Tuple
    reward: float
    done: bool
    info: dict


class GridWorld:
    def __init__(self, layout: List[str], monster_move_prob: float = 0.0):
        self.layout = layout
        self.w, self.h = len(layout[0]), len(layout)
        self.monster_move_prob = monster_move_prob
        # object sets
        self.rocks, self.fires = set(), set()
        self.apples, self.apple_index = [], {}
        self.keys, self.key_index = [], {}
        self.chests, self.chest_index = [], {}
        self.monster_start = []
        self.start = (0, 0)
        for y, row in enumerate(layout):
            for x, ch in enumerate(row):
                p = (x, y)
                if ch == "A":
                    self.apple_index[p] = len(self.apples)
                    self.apples.append(p)
                elif ch == "S":
                    self.start = p
                elif ch == "R":
                    self.rocks.add(p)
                elif ch == "F":
                    self.fires.add(p)
                elif ch == "K":
                    self.key_index[p] = len(self.keys)
                    self.keys.append(p)
                elif ch == "C":
                    self.chest_index[p] = len(self.chests)
                    self.chests.append(p)
                elif ch == "M":
                    self.monster_start.append(p)
        self.reset()

    def reset(self) -> Tuple:
        self.agent = self.start
        self.keys_in_hand = 0
        self.monsters = list(self.monster_start)
        self.apple_mask = sum(1 << i for i in range(len(self.apples)))
        self.key_mask = sum(1 << i for i in range(len(self.keys)))
        self.chest_mask = sum(1 << i for i in range(len(self.chests)))
        self.step_count = 0
        self.alive = True
        return self.encode_state()

    def encode_state(self) -> Tuple:
        return (
            self.agent[0],
            self.agent[1],
            self.apple_mask,
            self.chest_mask,
            self.key_mask,
            self.keys_in_hand,
            tuple(sorted(self.monsters)),
        )

    # movement helpers
    def in_bounds(self, p): return 0 <= p[0] < self.w and 0 <= p[1] < self.h
    def blocked(self, p): return p in self.rocks
    def cell_contains_monster(self, p): return p in self.monsters

    def try_move(self, p, a):
        dx, dy = ACTIONS[a]
        np = (p[0] + dx, p[1] + dy)
        if not self.in_bounds(np): return p
        if self.blocked(np): return p
        return np

    def remaining_apples(self) -> int:
        return bin(self.apple_mask).count("1")

    def remaining_chests(self) -> int:
        return bin(self.chest_mask).count("1")

    def all_rewards_collected(self) -> bool:
        return self.apple_mask == 0 and self.chest_mask == 0

    def collect_items(self) -> float:
        reward = 0.0
        if self.agent in self.apple_index:
            idx = self.apple_index[self.agent]
            if (self.apple_mask >> idx) & 1:
                self.apple_mask &= ~(1 << idx)
                reward += 1.0
        if self.agent in self.key_index:
            idx = self.key_index[self.agent]
            if (self.key_mask >> idx) & 1:
                self.key_mask &= ~(1 << idx)
                self.keys_in_hand += 1
        if self.agent in self.chest_index:
            idx = self.chest_index[self.agent]
            if (self.chest_mask >> idx) & 1 and self.keys_in_hand > 0:
                self.chest_mask &= ~(1 << idx)
                self.keys_in_hand -= 1
                reward += 2.0
        return reward

    def move_monsters(self):
        new_positions = []
        for p in self.monsters:
            np = p
            if random.random() < self.monster_move_prob:
                candidates = []
                for a in ALL_ACTIONS:
                    cand = (p[0] + ACTIONS[a][0], p[1] + ACTIONS[a][1])
                    if not self.in_bounds(cand): continue
                    if self.blocked(cand): continue
                    candidates.append(cand)
                if candidates:
                    np = random.choice(candidates)
            new_positions.append(np)
        self.monsters = new_positions

    def step(self, action: int) -> StepResult:
        self.step_count += 1
        reward = 0.0
        self.agent = self.try_move(self.agent, action)
        if self.agent in self.fires or self.cell_contains_monster(self.agent):
            self.alive = False
            return StepResult(self.encode_state(), reward, True, {"event": "death"})

        reward += self.collect_items()

        if self.monster_move_prob > 0.0 and self.monsters:
            self.move_monsters()
            if self.cell_contains_monster(self.agent):
                self.alive = False
                return StepResult(self.encode_state(), reward, True, {"event": "death"})

        done = self.all_rewards_collected()
        return StepResult(self.encode_state(), reward, done, {})


# -----------------------------
# Q-table and learning helpers
# -----------------------------
class QTable:
    def __init__(self):
        self.q: Dict[Tuple, float] = {}

    def get(self, s, a): return self.q.get((s, a), 0.0)
    def set(self, s, a, v): self.q[(s, a)] = v
    def best_value(self, s): return max(self.get(s, a) for a in ALL_ACTIONS)

    def best_actions(self, s):
        vals = [self.get(s, a) for a in ALL_ACTIONS]
        m = max(vals)
        return [a for a, v in zip(ALL_ACTIONS, vals) if v == m]


def linear_epsilon(ep, start, end, decay_ep):
    if decay_ep <= 0:
        return end
    t = min(ep / decay_ep, 1.0)
    return start + t * (end - start)


def epsilon_greedy(qtab: QTable, s, eps):
    if random.random() < eps:
        return random.choice(ALL_ACTIONS)
    best = qtab.best_actions(s)
    return random.choice(best)


def q_learning_update(qtab: QTable, s, a, r, sp, alpha, gamma):
    current = qtab.get(s, a)
    target = r + gamma * qtab.best_value(sp)
    qtab.set(s, a, current + alpha * (target - current))


def sarsa_update(qtab: QTable, s, a, r, sp, ap, alpha, gamma):
    current = qtab.get(s, a)
    target = r + gamma * qtab.get(sp, ap)
    qtab.set(s, a, current + alpha * (target - current))


# -----------------------------
# Drawing
# -----------------------------
def draw_grid(env: GridWorld, spec: LevelSpec, episode, step, epsilon, env_reward, total_reward, speed_label, intrinsic_label, paused: bool, started: bool):
    screen.fill(COL_BG)
    for x in range(env.w):
        for y in range(env.h):
            pygame.draw.rect(
                screen,
                COL_GRID,
                pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                1,
            )

    for p in env.rocks:
        pygame.draw.rect(screen, COL_ROCK, pygame.Rect(p[0] * TILE_SIZE, p[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE))
    for p in env.fires:
        pygame.draw.rect(screen, COL_FIRE, pygame.Rect(p[0] * TILE_SIZE + 6, p[1] * TILE_SIZE + 6, TILE_SIZE - 12, TILE_SIZE - 12), border_radius=6)

    for p, idx in env.apple_index.items():
        if (env.apple_mask >> idx) & 1:
            cx, cy = p[0] * TILE_SIZE + TILE_SIZE // 2, p[1] * TILE_SIZE + TILE_SIZE // 2
            pygame.draw.circle(screen, COL_APPLE, (cx, cy), TILE_SIZE // 3)

    for p, idx in env.key_index.items():
        if (env.key_mask >> idx) & 1:
            cx, cy = p[0] * TILE_SIZE + TILE_SIZE // 2, p[1] * TILE_SIZE + TILE_SIZE // 2
            pygame.draw.circle(screen, COL_KEY, (cx, cy), TILE_SIZE // 4)

    for p, idx in env.chest_index.items():
        color = COL_CHEST if (env.chest_mask >> idx) & 1 else COL_CHEST_OPEN
        pygame.draw.rect(screen, color, pygame.Rect(p[0] * TILE_SIZE + 6, p[1] * TILE_SIZE + 6, TILE_SIZE - 12, TILE_SIZE - 12), border_radius=4)

    for p in env.monsters:
        pygame.draw.rect(screen, COL_MONSTER, pygame.Rect(p[0] * TILE_SIZE + 4, p[1] * TILE_SIZE + 4, TILE_SIZE - 8, TILE_SIZE - 8), border_radius=4)

    ax, ay = env.agent
    pygame.draw.rect(
        screen,
        COL_AGENT,
        (ax * TILE_SIZE + 8, ay * TILE_SIZE + 8, TILE_SIZE - 16, TILE_SIZE - 16),
        border_radius=6,
    )

    status_line = "Controls: V toggle speed, R reset, S start, T stop"
    if paused and not started:
        status_line = "Press S to start"
    elif paused:
        status_line = "Paused (press S to resume)"
    hud = [
        f"{spec.name} ({ALGO_NAMES[spec.algo]})",
        f"Ep {episode + 1}  step {step}  eps {epsilon:.3f}",
        f"Apples {env.remaining_apples()} | Chests {env.remaining_chests()} | Keys in hand {env.keys_in_hand}",
        f"Return env {env_reward:.2f} / total {total_reward:.2f}",
        f"Speed:{speed_label} | Intrinsic:{intrinsic_label}  I intrinsic (level 6)",
        status_line,
        "Number keys 0-6 switch level, arrows to cycle",
    ]
    for i, t in enumerate(hud):
        screen.blit(font.render(t, True, COL_TEXT), (10, 8 + i * 20))
    pygame.display.flip()


# -----------------------------
# Training loop and controls
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GridWorld RL (Q-learning, SARSA, monsters, intrinsic reward)")
    parser.add_argument("--level", type=int, default=0, help="Start at level id (0-6)")
    return parser


def init_episode(env: GridWorld, spec: LevelSpec, cfg: Dict[str, Any], qtab: QTable, ep: int) -> Tuple[Tuple, Optional[int], Dict[Tuple, int], float, float, int, float]:
    s = env.reset()
    visit_counts: Dict[Tuple, int] = defaultdict(int)
    visit_counts[s] += 1
    eps = linear_epsilon(ep, float(cfg["epsilonStart"]), float(cfg["epsilonEnd"]), int(cfg["epsilonDecayEpisodes"]))
    a = epsilon_greedy(qtab, s, eps) if spec.algo == "sarsa" else None
    return s, a, visit_counts, 0.0, 0.0, 0, eps


def status_labels(visualize: bool, intrinsic_enabled: bool, spec: LevelSpec) -> Tuple[str, str]:
    speed_label = "visual" if visualize else "fast"
    intrinsic_label = "on" if intrinsic_enabled and spec.use_intrinsic else "off"
    return speed_label, intrinsic_label


def intrinsic_reward(cfg: Dict[str, Any], visit_counts: Dict[Tuple, int], state: Tuple) -> float:
    scale = float(cfg.get("intrinsicScale", DEFAULT_CFG["intrinsicScale"]))
    return scale / max(1, visit_counts[state])


def run_training(start_level: int = 0):
    current_level = start_level if start_level in LEVELS else 0
    spec = LEVELS[current_level]
    cfg = load_config(current_level)
    alpha = float(cfg["alpha"])
    gamma = float(cfg["gamma"])
    visualize, running = True, True
    paused = True
    started = False
    env = GridWorld(spec.layout, monster_move_prob=spec.monster_move_prob)
    qtab = QTable()
    ep = 0
    pending_level = None
    intrinsic_enabled = spec.use_intrinsic

    while running:
        s, a, visit_counts, ep_reward_env, ep_reward_total, steps, eps = init_episode(env, spec, cfg, qtab, ep)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_v:
                        visualize = not visualize
                    elif event.key == pygame.K_r:
                        qtab = QTable()
                        s, a, visit_counts, ep_reward_env, ep_reward_total, steps, eps = init_episode(env, spec, cfg, qtab, ep)
                    elif event.key == pygame.K_s:
                        paused = False
                        started = True
                    elif event.key == pygame.K_t:
                        paused = True
                    elif event.key == pygame.K_i and current_level == 6:
                        intrinsic_enabled = not intrinsic_enabled
                    elif pygame.K_0 <= event.key <= pygame.K_6:
                        pending_level = event.key - pygame.K_0
                    elif event.key == pygame.K_RIGHT:
                        idx = LEVEL_ORDER.index(current_level)
                        pending_level = LEVEL_ORDER[(idx + 1) % len(LEVEL_ORDER)]
                    elif event.key == pygame.K_LEFT:
                        idx = LEVEL_ORDER.index(current_level)
                        pending_level = LEVEL_ORDER[(idx - 1) % len(LEVEL_ORDER)]
            if not running:
                break

            if pending_level is not None and pending_level in LEVELS:
                current_level = pending_level
                spec = LEVELS[current_level]
                cfg = load_config(current_level)
                alpha = float(cfg["alpha"])
                gamma = float(cfg["gamma"])
                env = GridWorld(spec.layout, monster_move_prob=spec.monster_move_prob)
                qtab = QTable()
                ep = 0
                pending_level = None
                intrinsic_enabled = spec.use_intrinsic
                s, a, visit_counts, ep_reward_env, ep_reward_total, steps, eps = init_episode(env, spec, cfg, qtab, ep)
                continue
            pending_level = None

            if paused:
                speed_label, intrinsic_label = status_labels(visualize, intrinsic_enabled, spec)
                draw_grid(env, spec, ep, steps, eps, ep_reward_env, ep_reward_total, speed_label, intrinsic_label, paused=True, started=started)
                clock.tick(int(cfg["fpsVisual"]))
                continue

            if spec.algo == "q":
                a = epsilon_greedy(qtab, s, eps)
                res = env.step(a)
                visit_counts[res.next_state] += 1
                intrinsic = intrinsic_reward(cfg, visit_counts, res.next_state) if spec.use_intrinsic and intrinsic_enabled else 0.0
                reward_for_update = res.reward + intrinsic
                q_learning_update(qtab, s, a, reward_for_update, res.next_state, alpha, gamma)
                s = res.next_state
                ep_reward_env += res.reward
                ep_reward_total += reward_for_update
            else:
                res = env.step(a)
                visit_counts[res.next_state] += 1
                intrinsic = intrinsic_reward(cfg, visit_counts, res.next_state) if spec.use_intrinsic and intrinsic_enabled else 0.0
                ap = epsilon_greedy(qtab, res.next_state, eps)
                reward_for_update = res.reward + intrinsic
                sarsa_update(qtab, s, a, reward_for_update, res.next_state, ap, alpha, gamma)
                s, a = res.next_state, ap
                ep_reward_env += res.reward
                ep_reward_total += reward_for_update

            steps += 1

            speed_label, intrinsic_label = status_labels(visualize, intrinsic_enabled, spec)
            if visualize:
                draw_grid(env, spec, ep, steps, eps, ep_reward_env, ep_reward_total, speed_label, intrinsic_label, paused=False, started=started)
                clock.tick(int(cfg["fpsVisual"]))
            else:
                if steps % 5 == 0:
                    draw_grid(env, spec, ep, steps, eps, ep_reward_env, ep_reward_total, speed_label, intrinsic_label, paused=False, started=started)
                clock.tick(int(cfg["fpsFast"]))

            if res.done or steps >= int(cfg["maxStepsPerEpisode"]):
                draw_grid(env, spec, ep, steps, eps, ep_reward_env, ep_reward_total, speed_label, intrinsic_label, paused=False, started=started)
                break

        log_episode(current_level, spec.algo, ep, ep_reward_env, ep_reward_total, steps)
        ep += 1
        if ep >= int(cfg["episodes"]):
            ep = 0  # keep running but restart episode counter

    pygame.quit()


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_training(start_level=args.level)
