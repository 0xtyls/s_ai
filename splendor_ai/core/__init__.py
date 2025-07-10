"""
Splendor AI Core Package

This package contains the core game logic for Splendor, including:
- Game state representation
- Game rules and mechanics
- Player actions
- Card and noble definitions
- Constants and enums

All core components can be imported directly from this package.
"""

# Game and game state
from splendor_ai.core.game import (
    Game, GameState, GameMode, GameResult,
    create_game, simulate_random_game
)

# Player
from splendor_ai.core.player import Player

# Cards and nobles
from splendor_ai.core.cards import (
    Card, Noble,
    create_balanced_card_deck, create_standard_card_decks,
    create_noble_deck, get_historical_cards, get_historical_nobles,
    filter_affordable_cards
)

# Actions
from splendor_ai.core.actions import (
    Action, ActionType,
    TakeGemsAction, PurchaseCardAction, ReserveCardAction,
    get_all_valid_actions, create_action_from_dict,
    encode_action_for_neural_network
)

# Constants
from splendor_ai.core.constants import (
    GemColor, CardTier,
    REGULAR_GEMS, ALL_GEMS,
    VICTORY_POINTS, MAX_GEMS_TOTAL, MAX_RESERVED_CARDS
)

__all__ = [
    # Game
    'Game', 'GameState', 'GameMode', 'GameResult',
    'create_game', 'simulate_random_game',
    
    # Player
    'Player',
    
    # Cards
    'Card', 'Noble',
    'create_balanced_card_deck', 'create_standard_card_decks',
    'create_noble_deck', 'get_historical_cards', 'get_historical_nobles',
    'filter_affordable_cards',
    
    # Actions
    'Action', 'ActionType',
    'TakeGemsAction', 'PurchaseCardAction', 'ReserveCardAction',
    'get_all_valid_actions', 'create_action_from_dict',
    'encode_action_for_neural_network',
    
    # Constants
    'GemColor', 'CardTier',
    'REGULAR_GEMS', 'ALL_GEMS',
    'VICTORY_POINTS', 'MAX_GEMS_TOTAL', 'MAX_RESERVED_CARDS'
]
