"""
Splendor AI - A reinforcement learning and MCTS framework for the board game Splendor.

This package provides a complete implementation of Splendor rules, along with
AI agents that can play the game using different strategies.
"""

__version__ = "0.1.0"
__author__ = "Splendor AI Team"

# Make key components available at package level
from splendor_ai.core.game import Game, GameState
from splendor_ai.core.player import Player
from splendor_ai.core.actions import Action

# Version info as a tuple for programmatic access
VERSION_INFO = tuple(map(int, __version__.split('.')))

# Default configuration
DEFAULT_CONFIG = {
    "num_players": 2,
    "victory_points": 15,
    "max_gems_per_color": 7,
    "max_gems_total": 10
}
