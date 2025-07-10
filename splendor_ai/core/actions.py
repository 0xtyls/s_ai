"""
Actions for the Splendor game.

This module defines all possible actions in Splendor:
- Taking gems (same color or different colors)
- Purchasing development cards (from board or reserved)
- Reserving cards (from visible cards or deck)

Each action includes validation logic and utility methods for encoding/decoding
for AI agents.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union, ClassVar, Type

from splendor_ai.core.constants import (
    GemColor, CardTier, REGULAR_GEMS, ALL_GEMS,
    MAX_GEMS_TAKE_SAME_COLOR, MAX_GEMS_TAKE_DIFFERENT,
    MAX_GEMS_TOTAL, MAX_RESERVED_CARDS
)


class ActionType(Enum):
    """Enum representing the different types of actions in Splendor."""
    TAKE_GEMS = auto()
    PURCHASE_CARD = auto()
    RESERVE_CARD = auto()


class Action(ABC):
    """
    Abstract base class for all Splendor actions.
    
    All specific action types inherit from this class and implement
    the required abstract methods.
    """
    action_type: ClassVar[ActionType]
    
    @abstractmethod
    def validate(self, game_state, player_id: int) -> bool:
        """
        Validate if the action is legal in the current game state.
        
        Args:
            game_state: Current state of the game
            player_id: ID of the player performing the action
            
        Returns:
            True if the action is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def execute(self, game_state, player_id: int) -> None:
        """
        Execute the action, modifying the game state.
        
        Args:
            game_state: Current state of the game
            player_id: ID of the player performing the action
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict:
        """
        Convert the action to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the action
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict) -> Action:
        """
        Create an action from a dictionary representation.
        
        Args:
            data: Dictionary representation of the action
            
        Returns:
            Action object
        """
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the action.
        
        Returns:
            String representation
        """
        pass


@dataclass
class TakeGemsAction(Action):
    """
    Action to take gems from the supply.
    
    Players can either take:
    - 2 gems of the same color (if 4+ are available)
    - Up to 3 gems of different colors
    """
    action_type: ClassVar[ActionType] = ActionType.TAKE_GEMS
    gems: Dict[GemColor, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the action after initialization."""
        # Ensure we're not taking gold gems
        if GemColor.GOLD in self.gems:
            raise ValueError("Cannot take gold gems directly")
        
        # Ensure we're taking a valid number of gems
        total_gems = sum(self.gems.values())
        if total_gems <= 0:
            raise ValueError("Must take at least one gem")
        
        # Check if we're taking gems of the same color
        if len(self.gems) == 1:
            color, count = next(iter(self.gems.items()))
            if count > MAX_GEMS_TAKE_SAME_COLOR:
                raise ValueError(f"Cannot take more than {MAX_GEMS_TAKE_SAME_COLOR} gems of the same color")
        
        # Check if we're taking gems of different colors
        elif len(self.gems) > MAX_GEMS_TAKE_DIFFERENT:
            raise ValueError(f"Cannot take gems of more than {MAX_GEMS_TAKE_DIFFERENT} different colors")
        
        # Check if we're taking the correct number of each color
        for color, count in self.gems.items():
            if count <= 0:
                raise ValueError(f"Must take a positive number of {color.name} gems")
            if len(self.gems) > 1 and count > 1:
                raise ValueError("When taking gems of different colors, can only take 1 of each")
    
    def validate(self, game_state, player_id: int) -> bool:
        """
        Validate if the gem-taking action is legal.
        
        Args:
            game_state: Current state of the game
            player_id: ID of the player performing the action
            
        Returns:
            True if the action is valid, False otherwise
        """
        player = game_state.players[player_id]
        
        # Check if we're taking gems of the same color
        if len(self.gems) == 1:
            color, count = next(iter(self.gems.items()))
            # Can only take 2 of the same color if there are at least 4 available
            if count == 2 and game_state.gem_pool.get(color, 0) < 4:
                return False
        
        # Check if there are enough gems in the supply
        for color, count in self.gems.items():
            if game_state.gem_pool.get(color, 0) < count:
                return False
        
        # Check if the player would exceed the gem limit
        current_gems = sum(player.gems.values())
        new_gems = sum(self.gems.values())
        if current_gems + new_gems > MAX_GEMS_TOTAL:
            return False
        
        return True
    
    def execute(self, game_state, player_id: int) -> None:
        """
        Execute the gem-taking action.
        
        Args:
            game_state: Current state of the game
            player_id: ID of the player performing the action
        """
        player = game_state.players[player_id]
        
        # Take gems from the supply
        for color, count in self.gems.items():
            game_state.gem_pool[color] -= count
            player.gems[color] = player.gems.get(color, 0) + count
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "action_type": self.action_type.name,
            "gems": {color.name: count for color, count in self.gems.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> TakeGemsAction:
        """Create from dictionary representation."""
        gems = {GemColor[color]: count for color, count in data["gems"].items()}
        return cls(gems=gems)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if len(self.gems) == 1:
            color, count = next(iter(self.gems.items()))
            return f"Take {count} {color.name} gems"
        else:
            gems_str = ", ".join(f"1 {color.name}" for color in self.gems.keys())
            return f"Take {gems_str} gems"


@dataclass
class PurchaseCardAction(Action):
    """
    Action to purchase a development card.
    
    Players can purchase either:
    - A card from the visible cards on the board
    - A previously reserved card
    """
    action_type: ClassVar[ActionType] = ActionType.PURCHASE_CARD
    card_id: int  # ID of the card to purchase
    from_reserved: bool = False  # Whether the card is from the player's reserved cards
    payment: Dict[GemColor, int] = field(default_factory=dict)  # Gems used to pay
    
    def validate(self, game_state, player_id: int) -> bool:
        """
        Validate if the card purchase action is legal.
        
        Args:
            game_state: Current state of the game
            player_id: ID of the player performing the action
            
        Returns:
            True if the action is valid, False otherwise
        """
        player = game_state.players[player_id]
        
        # Find the card
        card = None
        if self.from_reserved:
            # Check if the card is in the player's reserved cards
            for reserved_card in player.reserved_cards:
                if reserved_card.id == self.card_id:
                    card = reserved_card
                    break
            if card is None:
                return False
        else:
            # Check if the card is on the board
            for tier in game_state.card_tiers.values():
                for board_card in tier:
                    if board_card.id == self.card_id:
                        card = board_card
                        break
                if card is not None:
                    break
            if card is None:
                return False
        
        # Check if the player has enough gems to pay for the card
        # First apply bonuses from existing cards
        effective_cost = {
            color: max(0, count - player.bonuses.get(color, 0))
            for color, count in card.cost.items()
        }
        
        # Check if the payment is valid
        total_payment = sum(self.payment.values())
        total_cost = sum(effective_cost.values())
        
        # Payment must equal cost
        if total_payment != total_cost:
            return False
        
        # Check if player has enough of each gem
        for color, count in self.payment.items():
            if player.gems.get(color, 0) < count:
                return False
        
        # Check that gold gems are used correctly
        gold_payment = self.payment.get(GemColor.GOLD, 0)
        non_gold_payment = {c: v for c, v in self.payment.items() if c != GemColor.GOLD}
        
        # For each color in effective_cost, check if payment + gold covers it
        remaining_gold = gold_payment
        for color, cost in effective_cost.items():
            paid = non_gold_payment.get(color, 0)
            if paid < cost:
                gold_needed = cost - paid
                if remaining_gold < gold_needed:
                    return False
                remaining_gold -= gold_needed
        
        # Any leftover gold is invalid
        if remaining_gold > 0:
            return False
        
        return True
    
    def execute(self, game_state, player_id: int) -> None:
        """
        Execute the card purchase action.
        
        Args:
            game_state: Current state of the game
            player_id: ID of the player performing the action
        """
        player = game_state.players[player_id]
        
        # Find the card
        card = None
        if self.from_reserved:
            # Find and remove the card from the player's reserved cards
            for i, reserved_card in enumerate(player.reserved_cards):
                if reserved_card.id == self.card_id:
                    card = reserved_card
                    player.reserved_cards.pop(i)
                    break
        else:
            # Find and remove the card from the board
            for tier, tier_cards in game_state.card_tiers.items():
                for i, board_card in enumerate(tier_cards):
                    if board_card.id == self.card_id:
                        card = board_card
                        tier_cards.pop(i)
                        # Draw a replacement card if available
                        if game_state.card_decks[tier]:
                            tier_cards.append(game_state.card_decks[tier].pop())
                        break
                if card is not None:
                    break
        
        # Pay for the card
        for color, count in self.payment.items():
            player.gems[color] -= count
            game_state.gem_pool[color] = game_state.gem_pool.get(color, 0) + count
        
        # Add the card to the player's purchased cards
        player.cards.append(card)
        
        # Update the player's bonuses
        player.bonuses[card.bonus] = player.bonuses.get(card.bonus, 0) + 1
        
        # Update the player's points
        player.points += card.points
        
        # Check for noble visits
        self._check_nobles(game_state, player_id)
    
    def _check_nobles(self, game_state, player_id: int) -> None:
        """
        Check if any nobles should visit the player.
        
        Args:
            game_state: Current state of the game
            player_id: ID of the player performing the action
        """
        player = game_state.players[player_id]
        
        # Check each noble
        nobles_to_remove = []
        for i, noble in enumerate(game_state.nobles):
            if noble.can_visit(player.bonuses):
                nobles_to_remove.append(i)
                player.nobles.append(noble)
                player.points += noble.points
        
        # Remove nobles from the board (in reverse order to avoid index issues)
        for i in reversed(nobles_to_remove):
            game_state.nobles.pop(i)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "action_type": self.action_type.name,
            "card_id": self.card_id,
            "from_reserved": self.from_reserved,
            "payment": {color.name: count for color, count in self.payment.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> PurchaseCardAction:
        """Create from dictionary representation."""
        payment = {GemColor[color]: count for color, count in data["payment"].items()}
        return cls(
            card_id=data["card_id"],
            from_reserved=data["from_reserved"],
            payment=payment
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        source = "reserved cards" if self.from_reserved else "the board"
        payment_str = ", ".join(f"{count} {color.name}" for color, count in self.payment.items())
        return f"Purchase card {self.card_id} from {source} paying {payment_str}"


@dataclass
class ReserveCardAction(Action):
    """
    Action to reserve a development card.
    
    Players can reserve either:
    - A visible card from the board
    - A card from the top of a deck (blind reserve)
    
    When reserving, the player also gets a gold gem if available.
    """
    action_type: ClassVar[ActionType] = ActionType.RESERVE_CARD
    card_id: Optional[int] = None  # ID of the card to reserve (None for blind reserve)
    tier: Optional[CardTier] = None  # Tier to reserve from (for blind reserve)
    
    def __post_init__(self):
        """Validate the action after initialization."""
        # Either card_id or tier must be specified, but not both
        if (self.card_id is None and self.tier is None) or (self.card_id is not None and self.tier is not None):
            raise ValueError("Must specify either card_id or tier, but not both")
    
    def validate(self, game_state, player_id: int) -> bool:
        """
        Validate if the reserve action is legal.
        
        Args:
            game_state: Current state of the game
            player_id: ID of the player performing the action
            
        Returns:
            True if the action is valid, False otherwise
        """
        player = game_state.players[player_id]
        
        # Check if the player has room for another reserved card
        if len(player.reserved_cards) >= MAX_RESERVED_CARDS:
            return False
        
        # Check if the card exists
        if self.card_id is not None:
            # Check if the card is on the board
            card_found = False
            for tier_cards in game_state.card_tiers.values():
                for card in tier_cards:
                    if card.id == self.card_id:
                        card_found = True
                        break
                if card_found:
                    break
            if not card_found:
                return False
        else:
            # Check if there are cards in the specified tier's deck
            if not game_state.card_decks.get(self.tier, []):
                return False
        
        return True
    
    def execute(self, game_state, player_id: int) -> None:
        """
        Execute the reserve action.
        
        Args:
            game_state: Current state of the game
            player_id: ID of the player performing the action
        """
        player = game_state.players[player_id]
        
        # Find and reserve the card
        if self.card_id is not None:
            # Reserve a visible card
            for tier, tier_cards in game_state.card_tiers.items():
                for i, card in enumerate(tier_cards):
                    if card.id == self.card_id:
                        # Remove the card from the board
                        reserved_card = tier_cards.pop(i)
                        # Draw a replacement if available
                        if game_state.card_decks[tier]:
                            tier_cards.append(game_state.card_decks[tier].pop())
                        # Add the card to the player's reserved cards
                        player.reserved_cards.append(reserved_card)
                        break
                if len(player.reserved_cards) > len(game_state.players[player_id].reserved_cards):
                    break
        else:
            # Reserve from the top of a deck
            if game_state.card_decks[self.tier]:
                reserved_card = game_state.card_decks[self.tier].pop()
                player.reserved_cards.append(reserved_card)
        
        # Give the player a gold gem if available
        if game_state.gem_pool.get(GemColor.GOLD, 0) > 0:
            game_state.gem_pool[GemColor.GOLD] -= 1
            player.gems[GemColor.GOLD] = player.gems.get(GemColor.GOLD, 0) + 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        result = {"action_type": self.action_type.name}
        if self.card_id is not None:
            result["card_id"] = self.card_id
        else:
            result["tier"] = self.tier.name
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> ReserveCardAction:
        """Create from dictionary representation."""
        if "card_id" in data:
            return cls(card_id=data["card_id"])
        else:
            return cls(tier=CardTier[data["tier"]])
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.card_id is not None:
            return f"Reserve card {self.card_id} from the board"
        else:
            return f"Reserve a card from the tier {self.tier.value} deck"


def get_all_valid_actions(game_state, player_id: int) -> List[Action]:
    """
    Get all valid actions for a player in the current game state.
    
    Args:
        game_state: Current state of the game
        player_id: ID of the player
        
    Returns:
        List of all valid actions
    """
    valid_actions = []
    player = game_state.players[player_id]
    
    # 1. Take gems actions
    # 1.1 Take 2 of the same color
    for color in REGULAR_GEMS:
        if game_state.gem_pool.get(color, 0) >= 4:
            action = TakeGemsAction(gems={color: 2})
            if action.validate(game_state, player_id):
                valid_actions.append(action)
    
    # 1.2 Take up to 3 different colors
    # Generate all combinations of 1-3 different colors
    for num_colors in range(1, MAX_GEMS_TAKE_DIFFERENT + 1):
        for colors in _get_color_combinations(REGULAR_GEMS, num_colors):
            gems = {color: 1 for color in colors if game_state.gem_pool.get(color, 0) > 0}
            if gems:
                action = TakeGemsAction(gems=gems)
                if action.validate(game_state, player_id):
                    valid_actions.append(action)
    
    # 2. Purchase card actions
    # 2.1 Purchase from board
    for tier_cards in game_state.card_tiers.values():
        for card in tier_cards:
            # Check if player can afford the card
            if _can_afford_card(player, card):
                # Generate all valid payment combinations
                for payment in _get_valid_payments(player, card):
                    action = PurchaseCardAction(
                        card_id=card.id,
                        from_reserved=False,
                        payment=payment
                    )
                    if action.validate(game_state, player_id):
                        valid_actions.append(action)
    
    # 2.2 Purchase from reserved cards
    for card in player.reserved_cards:
        # Check if player can afford the card
        if _can_afford_card(player, card):
            # Generate all valid payment combinations
            for payment in _get_valid_payments(player, card):
                action = PurchaseCardAction(
                    card_id=card.id,
                    from_reserved=True,
                    payment=payment
                )
                if action.validate(game_state, player_id):
                    valid_actions.append(action)
    
    # 3. Reserve card actions
    if len(player.reserved_cards) < MAX_RESERVED_CARDS:
        # 3.1 Reserve from board
        for tier_cards in game_state.card_tiers.values():
            for card in tier_cards:
                action = ReserveCardAction(card_id=card.id)
                if action.validate(game_state, player_id):
                    valid_actions.append(action)
        
        # 3.2 Reserve from deck
        for tier, deck in game_state.card_decks.items():
            if deck:  # Only if there are cards left in the deck
                action = ReserveCardAction(tier=tier)
                if action.validate(game_state, player_id):
                    valid_actions.append(action)
    
    return valid_actions


def _get_color_combinations(colors: List[GemColor], num_colors: int) -> List[List[GemColor]]:
    """
    Get all combinations of colors of a given length.
    
    Args:
        colors: List of colors to choose from
        num_colors: Number of colors to include in each combination
        
    Returns:
        List of color combinations
    """
    if num_colors == 0:
        return [[]]
    if not colors:
        return []
    
    first, rest = colors[0], colors[1:]
    with_first = [[first] + combo for combo in _get_color_combinations(rest, num_colors - 1)]
    without_first = _get_color_combinations(rest, num_colors)
    return with_first + without_first


def _can_afford_card(player, card) -> bool:
    """
    Check if a player can afford a card.
    
    Args:
        player: Player object
        card: Card object
        
    Returns:
        True if the player can afford the card, False otherwise
    """
    # Calculate effective cost after applying bonuses
    effective_cost = {
        color: max(0, count - player.bonuses.get(color, 0))
        for color, count in card.cost.items()
    }
    
    # Calculate total cost and available gems
    total_cost = sum(effective_cost.values())
    regular_gems = sum(player.gems.get(color, 0) for color in REGULAR_GEMS)
    gold_gems = player.gems.get(GemColor.GOLD, 0)
    
    # Check if player has enough gems in total
    if regular_gems + gold_gems < total_cost:
        return False
    
    # Check if player has enough of each color (considering gold as wild)
    remaining_gold = gold_gems
    for color, cost in effective_cost.items():
        available = player.gems.get(color, 0)
        if available < cost:
            # Need to use gold gems
            gold_needed = cost - available
            if remaining_gold < gold_needed:
                return False
            remaining_gold -= gold_needed
    
    return True


def _get_valid_payments(player, card) -> List[Dict[GemColor, int]]:
    """
    Get all valid payment combinations for a card.
    
    Args:
        player: Player object
        card: Card object
        
    Returns:
        List of valid payment dictionaries
    """
    # Calculate effective cost after applying bonuses
    effective_cost = {
        color: max(0, count - player.bonuses.get(color, 0))
        for color, count in card.cost.items()
    }
    
    # Remove zero costs
    effective_cost = {color: cost for color, cost in effective_cost.items() if cost > 0}
    
    # Start with the base payment (using regular gems where possible)
    base_payment = {
        color: min(cost, player.gems.get(color, 0))
        for color, cost in effective_cost.items()
    }
    
    # Calculate how much gold we need
    remaining_cost = {
        color: cost - base_payment.get(color, 0)
        for color, cost in effective_cost.items()
    }
    remaining_cost = {color: cost for color, cost in remaining_cost.items() if cost > 0}
    total_gold_needed = sum(remaining_cost.values())
    
    # Check if we have enough gold
    available_gold = player.gems.get(GemColor.GOLD, 0)
    if available_gold < total_gold_needed:
        return []  # Can't afford the card
    
    # Generate all valid distributions of gold gems
    result = []
    _generate_gold_distributions(
        result, base_payment.copy(), remaining_cost, available_gold, list(remaining_cost.keys()), 0
    )
    return result


def _generate_gold_distributions(
    result: List[Dict[GemColor, int]],
    current_payment: Dict[GemColor, int],
    remaining_cost: Dict[GemColor, int],
    available_gold: int,
    colors: List[GemColor],
    color_index: int
) -> None:
    """
    Recursively generate all valid distributions of gold gems.
    
    Args:
        result: List to append valid payments to
        current_payment: Current payment being built
        remaining_cost: Remaining cost for each color
        available_gold: Available gold gems
        colors: Colors that need gold
        color_index: Current color index
    """
    if color_index == len(colors):
        # We've assigned gold to all colors
        if available_gold == 0:
            # Valid payment
            result.append(current_payment.copy())
        return
    
    color = colors[color_index]
    needed = remaining_cost[color]
    
    # Try using different amounts of gold for this color
    for gold_used in range(needed + 1):
        if gold_used <= available_gold:
            # Update payment
            if gold_used > 0:
                current_payment[GemColor.GOLD] = current_payment.get(GemColor.GOLD, 0) + gold_used
            
            # Recurse to next color
            _generate_gold_distributions(
                result,
                current_payment,
                remaining_cost,
                available_gold - gold_used,
                colors,
                color_index + 1
            )
            
            # Backtrack
            if gold_used > 0:
                current_payment[GemColor.GOLD] = current_payment.get(GemColor.GOLD, 0) - gold_used


def create_action_from_dict(data: Dict) -> Action:
    """
    Create an action from a dictionary representation.
    
    Args:
        data: Dictionary representation of an action
        
    Returns:
        Action object
    """
    action_type = ActionType[data["action_type"]]
    
    if action_type == ActionType.TAKE_GEMS:
        return TakeGemsAction.from_dict(data)
    elif action_type == ActionType.PURCHASE_CARD:
        return PurchaseCardAction.from_dict(data)
    elif action_type == ActionType.RESERVE_CARD:
        return ReserveCardAction.from_dict(data)
    else:
        raise ValueError(f"Unknown action type: {action_type}")


def encode_action_for_neural_network(action: Action, game_state) -> List[float]:
    """
    Encode an action as a vector for a neural network.
    
    This creates a fixed-length vector representation of the action
    that can be used as output for a policy network.
    
    Args:
        action: Action to encode
        game_state: Current game state (needed for context)
        
    Returns:
        Vector representation of the action
    """
    # This is a simple implementation - a more sophisticated one would
    # create a one-hot encoding based on all possible actions
    
    # For now, we'll just create a simple encoding based on action type
    if isinstance(action, TakeGemsAction):
        # Encode as [1, 0, 0, gem1, gem2, gem3, gem4, gem5, 0, 0, ...]
        result = [1.0, 0.0, 0.0]
        # Add gem counts (normalized)
        for color in REGULAR_GEMS:
            result.append(action.gems.get(color, 0) / 2.0)  # Max is 2
        # Pad to fixed length
        result.extend([0.0] * 10)
        
    elif isinstance(action, PurchaseCardAction):
        # Encode as [0, 1, 0, card_index, from_reserved, 0, 0, ...]
        result = [0.0, 1.0, 0.0]
        
        # Find the card index (position in the game state)
        card_index = -1
        if action.from_reserved:
            for i, card in enumerate(game_state.players[game_state.current_player].reserved_cards):
                if card.id == action.card_id:
                    card_index = i
                    break
        else:
            index = 0
            for tier in CardTier:
                for i, card in enumerate(game_state.card_tiers[tier]):
                    if card.id == action.card_id:
                        card_index = index
                        break
                    index += 1
        
        result.append(card_index / 20.0)  # Normalize by max possible index
        result.append(1.0 if action.from_reserved else 0.0)
        
        # Pad to fixed length
        result.extend([0.0] * 10)
        
    elif isinstance(action, ReserveCardAction):
        # Encode as [0, 0, 1, has_card_id, tier, 0, 0, ...]
        result = [0.0, 0.0, 1.0]
        
        # Encode whether we're reserving a visible card or from the deck
        result.append(1.0 if action.card_id is not None else 0.0)
        
        # Encode the tier or card position
        if action.card_id is not None:
            # Find the card index
            card_index = -1
            index = 0
            for tier in CardTier:
                for i, card in enumerate(game_state.card_tiers[tier]):
                    if card.id == action.card_id:
                        card_index = index
                        break
                    index += 1
            result.append(card_index / 20.0)  # Normalize by max possible index
        else:
            # Encode the tier
            result.append((action.tier.value - 1) / 2.0)  # Normalize to [0, 1]
        
        # Pad to fixed length
        result.extend([0.0] * 10)
    
    else:
        raise ValueError(f"Unknown action type: {type(action)}")
    
    return result[:15]  # Ensure fixed length
