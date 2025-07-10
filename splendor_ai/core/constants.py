"""
Constants for the Splendor game.

This module defines all the game constants used throughout the Splendor implementation,
including gem types, card tiers, game limits, and victory conditions.
"""
from enum import Enum, auto
from typing import Dict, List, NamedTuple, Set, Tuple, Final


class GemColor(Enum):
    """Enum representing the different gem colors in Splendor."""
    WHITE = auto()
    BLUE = auto()
    GREEN = auto()
    RED = auto()
    BLACK = auto()
    GOLD = auto()  # Special wild/joker gem


# List of regular gems (excluding gold)
REGULAR_GEMS: Final[List[GemColor]] = [
    GemColor.WHITE,
    GemColor.BLUE,
    GemColor.GREEN,
    GemColor.RED,
    GemColor.BLACK,
]

# All gems including gold
ALL_GEMS: Final[List[GemColor]] = REGULAR_GEMS + [GemColor.GOLD]

# Display names for gems (for pretty printing)
GEM_DISPLAY_NAMES: Final[Dict[GemColor, str]] = {
    GemColor.WHITE: "White",
    GemColor.BLUE: "Blue",
    GemColor.GREEN: "Green",
    GemColor.RED: "Red",
    GemColor.BLACK: "Black",
    GemColor.GOLD: "Gold",
}

# Unicode symbols for gems (for terminal display)
GEM_SYMBOLS: Final[Dict[GemColor, str]] = {
    GemColor.WHITE: "⚪",
    GemColor.BLUE: "🔵",
    GemColor.GREEN: "🟢",
    GemColor.RED: "🔴",
    GemColor.BLACK: "⚫",
    GemColor.GOLD: "🟡",
}


class CardTier(Enum):
    """Enum representing the three tiers of development cards."""
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3


# Number of cards in each tier's deck
CARDS_PER_TIER: Final[Dict[CardTier, int]] = {
    CardTier.TIER_1: 40,
    CardTier.TIER_2: 30,
    CardTier.TIER_3: 20,
}

# Number of cards visible for purchase in each tier
VISIBLE_CARDS_PER_TIER: Final[int] = 4

# Card point ranges per tier
TIER_POINT_RANGES: Final[Dict[CardTier, Tuple[int, int]]] = {
    CardTier.TIER_1: (0, 1),    # Tier 1 cards have 0-1 points
    CardTier.TIER_2: (1, 3),    # Tier 2 cards have 1-3 points
    CardTier.TIER_3: (3, 5),    # Tier 3 cards have 3-5 points
}

# Maximum cost of a card in each gem color per tier
MAX_CARD_COST_PER_TIER: Final[Dict[CardTier, int]] = {
    CardTier.TIER_1: 4,
    CardTier.TIER_2: 5,
    CardTier.TIER_3: 7,
}


# Player limits
MIN_PLAYERS: Final[int] = 2
MAX_PLAYERS: Final[int] = 4
MAX_RESERVED_CARDS: Final[int] = 3
MAX_GEMS_TOTAL: Final[int] = 10  # Maximum total gems a player can hold

# Victory conditions
VICTORY_POINTS: Final[int] = 15  # Points needed to trigger end game

# Noble settings
NOBLE_POINTS: Final[int] = 3  # Each noble is worth 3 prestige points
NOBLE_REQUIREMENT_MIN: Final[int] = 3  # Minimum gems of same color required
NOBLE_REQUIREMENT_MAX: Final[int] = 4  # Maximum gems of same color required
NOBLE_REQUIRED_COLORS: Final[int] = 3  # Number of colors with requirements per noble

# Number of nobles in play based on player count
NOBLES_IN_PLAY: Final[Dict[int, int]] = {
    2: 3,
    3: 4,
    4: 5,
}

# Total number of nobles in the game
TOTAL_NOBLES: Final[int] = 10

# Gem supply based on player count
GEMS_PER_COLOR_BY_PLAYERS: Final[Dict[int, int]] = {
    2: 4,
    3: 5,
    4: 7,
}

# Gold gems are always 5 regardless of player count
GOLD_GEMS_COUNT: Final[int] = 5

# Action limits
MAX_GEMS_TAKE_SAME_COLOR: Final[int] = 2  # Can take 2 gems of same color if 4+ available
MAX_GEMS_TAKE_DIFFERENT: Final[int] = 3  # Can take up to 3 gems of different colors

# Game end conditions
FINAL_ROUND_AFTER_POINTS: Final[bool] = True  # Game ends after the round when someone reaches points
END_ON_EMPTY_DECK: Final[bool] = False  # Game doesn't end if a deck is depleted

# AI and simulation settings
DEFAULT_MCTS_ITERATIONS: Final[int] = 1000
DEFAULT_MCTS_EXPLORATION: Final[float] = 1.41  # UCB1 exploration parameter (sqrt(2))
DEFAULT_SIMULATION_GAMES: Final[int] = 100
