from dataclasses import dataclass
from typing import Dict, List

WIDTH_TILES, HEIGHT_TILES = 12, 12


@dataclass
class LevelSpec:
    name: str
    layout: List[str]
    algo: str  # "q" or "sarsa"
    monster_move_prob: float = 0.0
    use_intrinsic: bool = False


def normalized_layout(layout: List[str]) -> List[str]:
    rows = [row.ljust(WIDTH_TILES)[:WIDTH_TILES] for row in layout[:HEIGHT_TILES]]
    while len(rows) < HEIGHT_TILES:
        rows.append(" " * WIDTH_TILES)
    return rows


def center_layout(rows: List[str]) -> List[str]:
    blanks = max(0, HEIGHT_TILES - len(rows))
    top = blanks // 2
    bottom = blanks - top
    return ([" " * WIDTH_TILES] * top) + rows + ([" " * WIDTH_TILES] * bottom)


def make_levels() -> Dict[int, LevelSpec]:
    level0 = normalized_layout(
        center_layout(
            [
                "S           ",
                "            ",
                "        A   ",
                "        A   ",
                "        A   ",
                "        A   ",
                "        A   ",
                "        A   ",
            ]
        )
    )
    level1 = normalized_layout(
        center_layout(
            [
                "S   F   A   ",
                " RRRR R R   ",
                "     R F A  ",
                " RRRR R R   ",
                "     R F A  ",
                " RRRR R R   ",
                "            ",
                "            ",
            ]
        )
    )
    level2 = normalized_layout(
        center_layout(
            [
                "   S   A    ",
                "  RRRR RRR  ",
                "  K A   C   ",
                "  RRRR R     ",
                "   A   R    ",
                "      R     ",
                "   A        ",
                "            ",
            ]
        )
    )
    level3 = normalized_layout(
        center_layout(
            [
                "   S F A     ",
                "  RRRRFRRR  ",
                "  K   F C    ",
                "  RRRR R     ",
                " A   F   A   ",
                " RRRR RRR   ",
                "     A      ",
                "            ",
            ]
        )
    )
    level4 = normalized_layout(
        center_layout(
            [
                "   S M A     ",
                "  RRRR RRR  ",
                "    A   F   ",
                "  RRRR R     ",
                "    A  M    ",
                "  RRRR RRR  ",
                "     A      ",
                "            ",
            ]
        )
    )
    level5 = normalized_layout(
        center_layout(
            [
                "   S M A     ",
                "  RRRR RRR  ",
                "  K   F C    ",
                "  RRRR R     ",
                " A   F   A   ",
                " RRRR RRR   ",
                "   M   A    ",
                "            ",
            ]
        )
    )
    level6 = normalized_layout(
        center_layout(
            [
                "  S   A   A ",
                "   RRRR R   ",
                " K A   C    ",
                "   RRRR R   ",
                " K    A     ",
                "   RRRR R   ",
                "     A   C  ",
                "            ",
            ]
        )
    )
    return {
        0: LevelSpec("Level 0 - Apples (Q-learning)", level0, "q"),
        1: LevelSpec("Level 1 - Hazards (SARSA)", level1, "sarsa"),
        2: LevelSpec("Level 2 - Keys/Chest (Q-learning)", level2, "q"),
        3: LevelSpec("Level 3 - Hazards + Chest (SARSA)", level3, "sarsa"),
        4: LevelSpec("Level 4 - Monster Q-learning", level4, "q", monster_move_prob=0.4),
        5: LevelSpec("Level 5 - Monster SARSA", level5, "sarsa", monster_move_prob=0.4),
        6: LevelSpec(
            "Level 6 - Intrinsic Reward",
            level6,
            "q",
            monster_move_prob=0.0,
            use_intrinsic=True,
        ),
    }


LEVELS = make_levels()
LEVEL_ORDER = sorted(LEVELS.keys())
ALGO_NAMES = {"q": "Q-learning", "sarsa": "SARSA"}
