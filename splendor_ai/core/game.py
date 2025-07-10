"""
Game state and flow management for Splendor.

This module defines the core game mechanics, including:
- GameState: Complete representation of a game's state
- Game: Manager for game flow and rules
- Helper functions for game setup, validation, and serialization

The game follows the official Splendor rules with configurable parameters.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
import random
import json
import copy
from enum import Enum, auto
import time
from collections import defaultdict

from splendor_ai.core.constants import (
    GemColor, CardTier, REGULAR_GEMS, ALL_GEMS, GOLD_GEMS_COUNT,
    GEMS_PER_COLOR_BY_PLAYERS, NOBLES_IN_PLAY, VISIBLE_CARDS_PER_TIER,
    MIN_PLAYERS, MAX_PLAYERS, VICTORY_POINTS, FINAL_ROUND_AFTER_POINTS
)
from splendor_ai.core.cards import (
    Card, Noble, create_standard_card_decks, create_noble_deck,
    get_historical_cards, get_historical_nobles
)
from splendor_ai.core.player import Player
from splendor_ai.core.actions import (
    Action, ActionType, TakeGemsAction, PurchaseCardAction, ReserveCardAction,
    get_all_valid_actions, create_action_from_dict
)


class GameMode(Enum):
    """Enum representing different game modes."""
    HUMAN_VS_HUMAN = auto()
    HUMAN_VS_AI = auto()
    AI_VS_AI = auto()
    SELF_PLAY = auto()  # Single AI playing against itself (for training)


class GameResult(Enum):
    """Enum representing possible game results."""
    IN_PROGRESS = auto()
    WINNER = auto()  # Game has a winner
    DRAW = auto()  # Game ended in a draw (rare but possible)


@dataclass
class GameState:
    """
    Complete representation of a Splendor game state.
    
    This class contains all information needed to represent the game at any point,
    including players, cards, gems, and game status.
    """
    # Players
    players: List[Player] = field(default_factory=list)
    current_player_idx: int = 0
    
    # Cards and nobles
    card_decks: Dict[CardTier, List[Card]] = field(default_factory=dict)
    card_tiers: Dict[CardTier, List[Card]] = field(default_factory=dict)
    nobles: List[Noble] = field(default_factory=list)
    
    # Gems
    gem_pool: Dict[GemColor, int] = field(default_factory=dict)
    
    # Game state
    turn_count: int = 0
    actions_history: List[Tuple[int, Action]] = field(default_factory=list)  # (player_id, action)
    is_final_round: bool = False
    final_round_player: Optional[int] = None  # Player who triggered final round
    game_over: bool = False
    winner: Optional[int] = None  # Winner player ID
    result: GameResult = GameResult.IN_PROGRESS
    
    # Statistics
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    stats: Dict[str, Any] = field(default_factory=lambda: defaultdict(int))
    
    def __post_init__(self):
        """Initialize default values for the game state."""
        # Initialize empty gem pool
        if not self.gem_pool:
            self.gem_pool = {color: 0 for color in ALL_GEMS}
    
    @property
    def current_player(self) -> int:
        """Get the ID of the current player."""
        return self.current_player_idx
    
    @property
    def num_players(self) -> int:
        """Get the number of players in the game."""
        return len(self.players)
    
    def get_player(self, player_id: int) -> Player:
        """
        Get a player by ID.
        
        Args:
            player_id: ID of the player to get
            
        Returns:
            Player object
        """
        return self.players[player_id]
    
    def get_visible_cards(self) -> List[Card]:
        """
        Get all visible cards on the board.
        
        Returns:
            List of visible cards
        """
        visible_cards = []
        for tier_cards in self.card_tiers.values():
            visible_cards.extend(tier_cards)
        return visible_cards
    
    def get_valid_actions(self, player_id: int) -> List[Action]:
        """
        Get all valid actions for a player.
        
        Args:
            player_id: ID of the player
            
        Returns:
            List of valid actions
        """
        return get_all_valid_actions(self, player_id)
    
    def apply_action(self, player_id: int, action: Action) -> bool:
        """
        Apply an action to the game state.
        
        Args:
            player_id: ID of the player performing the action
            action: Action to apply
            
        Returns:
            True if the action was applied successfully, False otherwise
        """
        # Validate the action
        if not action.validate(self, player_id):
            return False
        
        # Execute the action
        action.execute(self, player_id)
        
        # Record the action in history
        self.actions_history.append((player_id, action))
        
        # Update statistics
        self._update_stats(player_id, action)
        
        # Check for game end conditions
        self._check_game_end(player_id)
        
        return True
    
    def _update_stats(self, player_id: int, action: Action) -> None:
        """
        Update game statistics based on the action.
        
        Args:
            player_id: ID of the player performing the action
            action: Action performed
        """
        # Track action types
        self.stats[f"actions_{action.action_type.name.lower()}"] += 1
        self.stats[f"player_{player_id}_actions"] += 1
        
        # Track specific statistics based on action type
        if isinstance(action, TakeGemsAction):
            gems_taken = sum(action.gems.values())
            self.stats["total_gems_taken"] += gems_taken
            self.stats[f"player_{player_id}_gems_taken"] += gems_taken
            
        elif isinstance(action, PurchaseCardAction):
            self.stats["cards_purchased"] += 1
            self.stats[f"player_{player_id}_cards_purchased"] += 1
            
            # Find the card to get its tier and points
            card = None
            player = self.players[player_id]
            
            # Look through player's cards (it was just added)
            for c in player.cards:
                if c.id == action.card_id:
                    card = c
                    break
            
            if card:
                self.stats[f"tier_{card.tier.value}_cards_purchased"] += 1
                self.stats["total_points_from_cards"] += card.points
                self.stats[f"player_{player_id}_points_from_cards"] += card.points
            
        elif isinstance(action, ReserveCardAction):
            self.stats["cards_reserved"] += 1
            self.stats[f"player_{player_id}_cards_reserved"] += 1
    
    def _check_game_end(self, player_id: int) -> None:
        """
        Check if the game should end.
        
        Args:
            player_id: ID of the player who just took an action
        """
        player = self.players[player_id]
        
        # Check if the player has reached the victory point threshold
        if player.points >= VICTORY_POINTS and not self.is_final_round:
            self.is_final_round = True
            self.final_round_player = player_id
        
        # If we're in the final round and have returned to the player who triggered it,
        # or if we've completed a full round after the final round was triggered,
        # end the game
        if self.is_final_round:
            if self.final_round_player == player_id and self.turn_count > 0:
                self._end_game()
    
    def _end_game(self) -> None:
        """End the game and determine the winner."""
        self.game_over = True
        self.end_time = time.time()
        
        # Find the player with the most points
        max_points = -1
        winners = []
        
        for player in self.players:
            if player.points > max_points:
                max_points = player.points
                winners = [player.id]
            elif player.points == max_points:
                winners.append(player.id)
        
        # Set the result
        if len(winners) == 1:
            self.winner = winners[0]
            self.result = GameResult.WINNER
        else:
            # Tiebreaker: player with fewer development cards
            min_cards = float('inf')
            tiebreak_winners = []
            
            for player_id in winners:
                player = self.players[player_id]
                num_cards = len(player.cards)
                
                if num_cards < min_cards:
                    min_cards = num_cards
                    tiebreak_winners = [player_id]
                elif num_cards == min_cards:
                    tiebreak_winners.append(player_id)
            
            if len(tiebreak_winners) == 1:
                self.winner = tiebreak_winners[0]
                self.result = GameResult.WINNER
            else:
                # It's a true draw (very rare)
                self.result = GameResult.DRAW
        
        # Update final statistics
        self.stats["game_duration"] = self.end_time - self.start_time
        self.stats["total_turns"] = self.turn_count
        if self.winner is not None:
            self.stats["winner"] = self.winner
            self.stats["winner_points"] = self.players[self.winner].points
    
    def next_turn(self) -> int:
        """
        Advance to the next player's turn.
        
        Returns:
            ID of the new current player
        """
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        
        # If we've gone full circle, increment the turn counter
        if self.current_player_idx == 0:
            self.turn_count += 1
        
        return self.current_player_idx
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the game state to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the game state
        """
        # Create a dictionary of all cards and nobles for easy lookup during deserialization
        all_cards = {}
        for tier, deck in self.card_decks.items():
            for card in deck:
                all_cards[card.id] = card
        for tier, visible in self.card_tiers.items():
            for card in visible:
                all_cards[card.id] = card
        for player in self.players:
            for card in player.cards:
                all_cards[card.id] = card
            for card in player.reserved_cards:
                all_cards[card.id] = card
        
        all_nobles = {noble.id: noble for noble in self.nobles}
        for player in self.players:
            for noble in player.nobles:
                all_nobles[noble.id] = noble
        
        # Convert card decks and tiers
        card_decks_dict = {
            tier.name: [card.id for card in deck]
            for tier, deck in self.card_decks.items()
        }
        
        card_tiers_dict = {
            tier.name: [card.id for card in visible]
            for tier, visible in self.card_tiers.items()
        }
        
        # Convert gem pool
        gem_pool_dict = {
            color.name: count
            for color, count in self.gem_pool.items()
        }
        
        # Convert actions history
        actions_history_dict = [
            (player_id, action.to_dict())
            for player_id, action in self.actions_history
        ]
        
        return {
            "players": [player.to_dict() for player in self.players],
            "current_player_idx": self.current_player_idx,
            "card_decks": card_decks_dict,
            "card_tiers": card_tiers_dict,
            "nobles": [noble.id for noble in self.nobles],
            "gem_pool": gem_pool_dict,
            "turn_count": self.turn_count,
            "actions_history": actions_history_dict,
            "is_final_round": self.is_final_round,
            "final_round_player": self.final_round_player,
            "game_over": self.game_over,
            "winner": self.winner,
            "result": self.result.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "stats": dict(self.stats),
            # Include all cards and nobles for deserialization
            "_all_cards": {str(card_id): card.to_dict() for card_id, card in all_cards.items()},
            "_all_nobles": {str(noble_id): noble.to_dict() for noble_id, noble in all_nobles.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameState':
        """
        Create a game state from a dictionary representation.
        
        Args:
            data: Dictionary representation of the game state
            
        Returns:
            GameState object
        """
        # Reconstruct all cards and nobles
        all_cards = {}
        for card_id, card_data in data["_all_cards"].items():
            card_id = int(card_id)
            tier = CardTier[card_data["tier"]]
            bonus = GemColor[card_data["bonus"]]
            cost = {GemColor[color]: count for color, count in card_data["cost"].items()}
            
            all_cards[card_id] = Card(
                id=card_id,
                tier=tier,
                points=card_data["points"],
                cost=cost,
                bonus=bonus
            )
        
        all_nobles = {}
        for noble_id, noble_data in data["_all_nobles"].items():
            noble_id = int(noble_id)
            requirements = {
                GemColor[color]: count
                for color, count in noble_data["requirements"].items()
            }
            
            all_nobles[noble_id] = Noble(
                id=noble_id,
                points=noble_data["points"],
                requirements=requirements
            )
        
        # Reconstruct players
        players = []
        for player_data in data["players"]:
            player = Player.from_dict(player_data, all_cards, all_nobles)
            players.append(player)
        
        # Reconstruct card decks and tiers
        card_decks = {
            CardTier[tier]: [all_cards[card_id] for card_id in deck]
            for tier, deck in data["card_decks"].items()
        }
        
        card_tiers = {
            CardTier[tier]: [all_cards[card_id] for card_id in visible]
            for tier, visible in data["card_tiers"].items()
        }
        
        # Reconstruct nobles
        nobles = [all_nobles[noble_id] for noble_id in data["nobles"]]
        
        # Reconstruct gem pool
        gem_pool = {
            GemColor[color]: count
            for color, count in data["gem_pool"].items()
        }
        
        # Reconstruct actions history
        actions_history = [
            (player_id, create_action_from_dict(action_data))
            for player_id, action_data in data["actions_history"]
        ]
        
        # Create the game state
        game_state = cls(
            players=players,
            current_player_idx=data["current_player_idx"],
            card_decks=card_decks,
            card_tiers=card_tiers,
            nobles=nobles,
            gem_pool=gem_pool,
            turn_count=data["turn_count"],
            actions_history=actions_history,
            is_final_round=data["is_final_round"],
            final_round_player=data["final_round_player"],
            game_over=data["game_over"],
            winner=data["winner"],
            result=GameResult[data["result"]],
            start_time=data["start_time"],
            end_time=data["end_time"],
            stats=defaultdict(int, data["stats"])
        )
        
        return game_state
    
    def to_json(self) -> str:
        """
        Convert the game state to a JSON string.
        
        Returns:
            JSON string representation of the game state
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'GameState':
        """
        Create a game state from a JSON string.
        
        Args:
            json_str: JSON string representation of the game state
            
        Returns:
            GameState object
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def clone(self) -> 'GameState':
        """
        Create a deep copy of the game state.
        
        Returns:
            Copy of the game state
        """
        return copy.deepcopy(self)
    
    def get_observation(self, player_id: int) -> Dict[str, Any]:
        """
        Get an observation of the game state from a player's perspective.
        
        This includes only information that would be visible to the player
        in a real game (no hidden cards, etc.).
        
        Args:
            player_id: ID of the player
            
        Returns:
            Dictionary containing the observation
        """
        observation = {}
        
        # Player information (public)
        observation["players"] = []
        for i, player in enumerate(self.players):
            player_info = {
                "id": player.id,
                "name": player.name,
                "gems": {color.name: count for color, count in player.gems.items()},
                "bonuses": {color.name: count for color, count in player.bonuses.items()},
                "points": player.points,
                "num_cards": len(player.cards),
                "num_reserved_cards": len(player.reserved_cards),
                "nobles": [
                    {
                        "id": noble.id,
                        "points": noble.points,
                        "requirements": {color.name: count for color, count in noble.requirements.items()}
                    }
                    for noble in player.nobles
                ]
            }
            
            # Add cards (public information)
            player_info["cards"] = [
                {
                    "id": card.id,
                    "tier": card.tier.value,
                    "points": card.points,
                    "cost": {color.name: count for color, count in card.cost.items()},
                    "bonus": card.bonus.name
                }
                for card in player.cards
            ]
            
            # Add reserved cards (only visible to the player who reserved them)
            if i == player_id:
                player_info["reserved_cards"] = [
                    {
                        "id": card.id,
                        "tier": card.tier.value,
                        "points": card.points,
                        "cost": {color.name: count for color, count in card.cost.items()},
                        "bonus": card.bonus.name
                    }
                    for card in player.reserved_cards
                ]
            else:
                # For other players, only reveal the count
                player_info["reserved_cards"] = [
                    {"tier": card.tier.value}
                    for card in player.reserved_cards
                ]
            
            observation["players"].append(player_info)
        
        # Board information
        observation["current_player"] = self.current_player_idx
        observation["turn_count"] = self.turn_count
        observation["gem_pool"] = {color.name: count for color, count in self.gem_pool.items()}
        
        # Visible cards
        observation["card_tiers"] = {
            tier.name: [
                {
                    "id": card.id,
                    "tier": card.tier.value,
                    "points": card.points,
                    "cost": {color.name: count for color, count in card.cost.items()},
                    "bonus": card.bonus.name
                }
                for card in cards
            ]
            for tier, cards in self.card_tiers.items()
        }
        
        # Deck sizes (public information)
        observation["deck_sizes"] = {
            tier.name: len(deck)
            for tier, deck in self.card_decks.items()
        }
        
        # Nobles
        observation["nobles"] = [
            {
                "id": noble.id,
                "points": noble.points,
                "requirements": {color.name: count for color, count in noble.requirements.items()}
            }
            for noble in self.nobles
        ]
        
        # Game state
        observation["is_final_round"] = self.is_final_round
        observation["game_over"] = self.game_over
        observation["winner"] = self.winner
        
        # Valid actions for the current player
        if player_id == self.current_player_idx:
            observation["valid_actions"] = [
                action.to_dict()
                for action in self.get_valid_actions(player_id)
            ]
        
        return observation
    
    def get_state_features(self) -> List[float]:
        """
        Get a feature vector representing the game state for ML models.
        
        Returns:
            List of normalized features
        """
        features = []
        
        # Game progress
        features.append(self.turn_count / 30.0)  # Normalize by an assumed max of 30 turns
        features.append(1.0 if self.is_final_round else 0.0)
        
        # Gem pool (normalized by max gems)
        max_gems = 7  # Maximum gems per color
        for color in ALL_GEMS:
            features.append(self.gem_pool.get(color, 0) / max_gems)
        
        # Card counts
        for tier in CardTier:
            # Deck size (normalized by original deck size)
            max_deck_size = {CardTier.TIER_1: 40, CardTier.TIER_2: 30, CardTier.TIER_3: 20}[tier]
            features.append(len(self.card_decks.get(tier, [])) / max_deck_size)
            
            # Visible cards (normalized by max visible)
            features.append(len(self.card_tiers.get(tier, [])) / VISIBLE_CARDS_PER_TIER)
        
        # Noble count (normalized by max nobles)
        max_nobles = 10
        features.append(len(self.nobles) / max_nobles)
        
        # Current player index (one-hot encoded)
        for i in range(MAX_PLAYERS):
            features.append(1.0 if i == self.current_player_idx else 0.0)
        
        # Add features for each player
        for i in range(MAX_PLAYERS):
            if i < len(self.players):
                player_features = self.players[i].get_state_features()
                features.extend(player_features)
            else:
                # Padding for non-existent players
                features.extend([0.0] * 15)  # Assuming player features are length 15
        
        return features


class Game:
    """
    Manager for Splendor game flow and rules.
    
    This class handles game setup, turn management, and provides interfaces
    for different types of players (human, AI).
    """
    def __init__(
        self,
        num_players: int = 2,
        player_names: Optional[List[str]] = None,
        use_historical_cards: bool = True,
        victory_points: int = VICTORY_POINTS,
        random_seed: Optional[int] = None
    ):
        """
        Initialize a new Splendor game.
        
        Args:
            num_players: Number of players (2-4)
            player_names: List of player names (defaults to "Player 1", "Player 2", etc.)
            use_historical_cards: Whether to use the historical card distribution
            victory_points: Points needed to trigger the end game
            random_seed: Random seed for reproducibility
        """
        if num_players < MIN_PLAYERS or num_players > MAX_PLAYERS:
            raise ValueError(f"Number of players must be between {MIN_PLAYERS} and {MAX_PLAYERS}")
        
        self.num_players = num_players
        self.use_historical_cards = use_historical_cards
        self.victory_points = victory_points
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
        
        # Set player names
        if player_names is None:
            self.player_names = [f"Player {i+1}" for i in range(num_players)]
        else:
            if len(player_names) != num_players:
                raise ValueError("Number of player names must match number of players")
            self.player_names = player_names
        
        # Initialize game state
        self.state = self._setup_game()
        
        # Game mode and agent callbacks
        self.mode = GameMode.HUMAN_VS_HUMAN
        self.agent_callbacks = {}  # Map from player ID to agent callback function
    
    def _setup_game(self) -> GameState:
        """
        Set up a new game state.
        
        Returns:
            Initialized GameState
        """
        # Create players
        players = [
            Player(id=i, name=self.player_names[i])
            for i in range(self.num_players)
        ]
        
        # Create card decks
        if self.use_historical_cards:
            card_decks = get_historical_cards()
        else:
            card_decks = create_standard_card_decks()
        
        # Shuffle card decks
        for deck in card_decks.values():
            random.shuffle(deck)
        
        # Set up visible cards
        card_tiers = {
            tier: [deck.pop() for _ in range(VISIBLE_CARDS_PER_TIER)]
            for tier, deck in card_decks.items()
        }
        
        # Create nobles
        if self.use_historical_cards:
            all_nobles = get_historical_nobles()
        else:
            all_nobles = create_noble_deck()
        
        # Shuffle and select nobles based on player count
        random.shuffle(all_nobles)
        nobles = all_nobles[:NOBLES_IN_PLAY[self.num_players]]
        
        # Set up gem pool
        gem_pool = {
            color: GEMS_PER_COLOR_BY_PLAYERS[self.num_players]
            for color in REGULAR_GEMS
        }
        gem_pool[GemColor.GOLD] = GOLD_GEMS_COUNT
        
        # Create game state
        state = GameState(
            players=players,
            card_decks=card_decks,
            card_tiers=card_tiers,
            nobles=nobles,
            gem_pool=gem_pool
        )
        
        return state
    
    def reset(self) -> GameState:
        """
        Reset the game to a new initial state.
        
        Returns:
            New game state
        """
        self.state = self._setup_game()
        return self.state
    
    def set_mode(self, mode: GameMode) -> None:
        """
        Set the game mode.
        
        Args:
            mode: Game mode to set
        """
        self.mode = mode
    
    def register_agent(self, player_id: int, agent_callback: Callable[[GameState, int], Action]) -> None:
        """
        Register an AI agent for a player.
        
        The agent callback should take a game state and player ID and return an action.
        
        Args:
            player_id: ID of the player
            agent_callback: Function that selects an action given the game state
        """
        self.agent_callbacks[player_id] = agent_callback
    
    def step(self, action: Optional[Action] = None) -> Tuple[GameState, bool]:
        """
        Advance the game by one step.
        
        If an action is provided, it will be applied. Otherwise, if the current player
        has an agent callback registered, that will be used to select an action.
        
        Args:
            action: Optional action to apply
            
        Returns:
            Tuple of (new game state, whether the game is over)
        """
        if self.state.game_over:
            return self.state, True
        
        current_player = self.state.current_player
        
        # If no action is provided, try to get one from the agent callback
        if action is None and current_player in self.agent_callbacks:
            action = self.agent_callbacks[current_player](self.state, current_player)
        
        # If we still don't have an action, we can't proceed
        if action is None:
            raise ValueError("No action provided and no agent callback registered for current player")
        
        # Apply the action
        if not self.state.apply_action(current_player, action):
            raise ValueError("Invalid action")
        
        # Check if the game is over
        if self.state.game_over:
            return self.state, True
        
        # Advance to the next player
        self.state.next_turn()
        
        return self.state, self.state.game_over
    
    def run_game(self, max_turns: int = 100) -> GameState:
        """
        Run the game until completion or max turns.
        
        This method requires all players to have agent callbacks registered.
        
        Args:
            max_turns: Maximum number of turns to run
            
        Returns:
            Final game state
        """
        # Check that all players have agent callbacks
        for i in range(self.num_players):
            if i not in self.agent_callbacks:
                raise ValueError(f"No agent callback registered for player {i}")
        
        # Run the game
        while not self.state.game_over and self.state.turn_count < max_turns:
            self.step()
        
        return self.state
    
    def get_winner(self) -> Optional[int]:
        """
        Get the ID of the winning player, if any.
        
        Returns:
            ID of the winning player, or None if the game is not over or ended in a draw
        """
        if not self.state.game_over:
            return None
        return self.state.winner
    
    def get_result(self) -> GameResult:
        """
        Get the result of the game.
        
        Returns:
            Game result
        """
        return self.state.result
    
    def get_player_score(self, player_id: int) -> int:
        """
        Get a player's score.
        
        Args:
            player_id: ID of the player
            
        Returns:
            Player's score
        """
        return self.state.players[player_id].points
    
    def get_scores(self) -> List[int]:
        """
        Get all players' scores.
        
        Returns:
            List of scores
        """
        return [player.points for player in self.state.players]
    
    def get_game_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the game.
        
        Returns:
            Dictionary of game statistics
        """
        stats = dict(self.state.stats)
        
        # Add some additional statistics
        if self.state.game_over and self.state.end_time:
            stats["duration"] = self.state.end_time - self.state.start_time
        else:
            stats["duration"] = time.time() - self.state.start_time
        
        stats["turns"] = self.state.turn_count
        stats["players"] = self.num_players
        
        if self.state.game_over:
            stats["result"] = self.state.result.name
            if self.state.winner is not None:
                stats["winner"] = self.state.winner
                stats["winner_name"] = self.state.players[self.state.winner].name
                stats["winner_score"] = self.state.players[self.state.winner].points
        
        # Per-player statistics
        for i, player in enumerate(self.state.players):
            stats[f"player_{i}_score"] = player.points
            stats[f"player_{i}_cards"] = len(player.cards)
            stats[f"player_{i}_nobles"] = len(player.nobles)
            stats[f"player_{i}_gems"] = sum(player.gems.values())
            stats[f"player_{i}_bonuses"] = sum(player.bonuses.values())
        
        return stats
    
    def save_game(self, filename: str) -> None:
        """
        Save the current game state to a file.
        
        Args:
            filename: Name of the file to save to
        """
        with open(filename, 'w') as f:
            f.write(self.state.to_json())
    
    @classmethod
    def load_game(cls, filename: str) -> 'Game':
        """
        Load a game from a file.
        
        Args:
            filename: Name of the file to load from
            
        Returns:
            Loaded Game object
        """
        with open(filename, 'r') as f:
            json_str = f.read()
        
        state = GameState.from_json(json_str)
        
        # Create a game object
        game = cls(num_players=len(state.players))
        game.state = state
        
        return game
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the game.
        
        Returns:
            String representation
        """
        result = f"Splendor Game (Players: {self.num_players}, Turn: {self.state.turn_count})\n"
        
        # Add gem pool
        result += "Gem Pool:\n"
        for color in ALL_GEMS:
            count = self.state.gem_pool.get(color, 0)
            if count > 0:
                result += f"  {color.name}: {count}\n"
        
        # Add nobles
        result += "\nNobles:\n"
        for noble in self.state.nobles:
            result += f"  {noble}\n"
        
        # Add card tiers
        result += "\nCards:\n"
        for tier in CardTier:
            result += f"  Tier {tier.value}:\n"
            for card in self.state.card_tiers.get(tier, []):
                result += f"    {card}\n"
            result += f"    (Deck: {len(self.state.card_decks.get(tier, []))} cards)\n"
        
        # Add players
        result += "\nPlayers:\n"
        for i, player in enumerate(self.state.players):
            result += f"  {player.name} (ID: {player.id}):\n"
            result += f"    Points: {player.points}\n"
            
            # Add gems
            result += "    Gems: "
            gems_str = ", ".join(f"{count} {color.name}" for color, count in player.gems.items() if count > 0)
            result += gems_str or "None"
            result += "\n"
            
            # Add bonuses
            result += "    Bonuses: "
            bonuses_str = ", ".join(f"{count} {color.name}" for color, count in player.bonuses.items() if count > 0)
            result += bonuses_str or "None"
            result += "\n"
            
            # Add cards
            result += f"    Cards: {len(player.cards)}\n"
            
            # Add reserved cards
            result += f"    Reserved Cards: {len(player.reserved_cards)}\n"
            
            # Add nobles
            result += f"    Nobles: {len(player.nobles)}\n"
            
            # Indicate current player
            if i == self.state.current_player:
                result += "    (Current Player)\n"
        
        # Add game status
        if self.state.game_over:
            result += "\nGame Over\n"
            if self.state.result == GameResult.WINNER:
                winner = self.state.players[self.state.winner]
                result += f"Winner: {winner.name} with {winner.points} points\n"
            else:
                result += "Result: Draw\n"
        elif self.state.is_final_round:
            result += "\nFinal Round\n"
        
        return result


def create_game(
    num_players: int = 2,
    player_names: Optional[List[str]] = None,
    use_historical_cards: bool = True,
    random_seed: Optional[int] = None
) -> Game:
    """
    Create a new Splendor game.
    
    Args:
        num_players: Number of players (2-4)
        player_names: List of player names
        use_historical_cards: Whether to use the historical card distribution
        random_seed: Random seed for reproducibility
        
    Returns:
        Game object
    """
    return Game(
        num_players=num_players,
        player_names=player_names,
        use_historical_cards=use_historical_cards,
        random_seed=random_seed
    )


def simulate_random_game(
    num_players: int = 2,
    max_turns: int = 100,
    random_seed: Optional[int] = None
) -> Tuple[GameState, List[int]]:
    """
    Simulate a game with random agents.
    
    Args:
        num_players: Number of players
        max_turns: Maximum number of turns
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (final game state, scores)
    """
    game = create_game(num_players=num_players, random_seed=random_seed)
    
    # Register random agents for all players
    for i in range(num_players):
        game.register_agent(i, lambda state, player_id: random.choice(state.get_valid_actions(player_id)))
    
    # Run the game
    final_state = game.run_game(max_turns=max_turns)
    
    return final_state, game.get_scores()
