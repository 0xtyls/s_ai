"""
Player representation for the Splendor game.

This module defines the Player class which tracks a player's state including
gems, cards, nobles, points, and provides methods for player actions.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import json

from splendor_ai.core.constants import (
    GemColor, REGULAR_GEMS, ALL_GEMS, MAX_GEMS_TOTAL, MAX_RESERVED_CARDS
)
from splendor_ai.core.cards import Card, Noble


@dataclass
class Player:
    """
    Represents a player in the Splendor game.
    
    Tracks the player's gems, cards, nobles, points, and provides methods
    for checking affordability, calculating bonuses, and more.
    """
    id: int  # Player ID (0-indexed)
    name: str  # Player name
    gems: Dict[GemColor, int] = field(default_factory=dict)  # Gems the player has
    cards: List[Card] = field(default_factory=list)  # Purchased development cards
    reserved_cards: List[Card] = field(default_factory=list)  # Reserved cards
    nobles: List[Noble] = field(default_factory=list)  # Acquired nobles
    points: int = 0  # Prestige points
    bonuses: Dict[GemColor, int] = field(default_factory=dict)  # Gem bonuses from cards
    
    def __post_init__(self):
        """Initialize default values and validate the player state."""
        # Initialize empty gem counts
        for color in ALL_GEMS:
            if color not in self.gems:
                self.gems[color] = 0
        
        # Initialize empty bonuses
        for color in REGULAR_GEMS:
            if color not in self.bonuses:
                self.bonuses[color] = 0
        
        # Recalculate bonuses and points
        self.recalculate_bonuses()
        self.recalculate_points()
    
    def recalculate_bonuses(self) -> None:
        """Recalculate gem bonuses based on purchased cards."""
        # Reset bonuses
        self.bonuses = {color: 0 for color in REGULAR_GEMS}
        
        # Add bonuses from cards
        for card in self.cards:
            self.bonuses[card.bonus] = self.bonuses.get(card.bonus, 0) + 1
    
    def recalculate_points(self) -> None:
        """Recalculate prestige points from cards and nobles."""
        self.points = sum(card.points for card in self.cards) + sum(noble.points for noble in self.nobles)
    
    def can_afford_card(self, card: Card) -> bool:
        """
        Check if the player can afford to purchase a card.
        
        Args:
            card: The card to check
            
        Returns:
            True if the player can afford the card, False otherwise
        """
        # Calculate effective cost after applying bonuses
        effective_cost = {
            color: max(0, count - self.bonuses.get(color, 0))
            for color, count in card.cost.items()
        }
        
        # Calculate total cost and available gems
        total_cost = sum(effective_cost.values())
        regular_gems = sum(self.gems.get(color, 0) for color in REGULAR_GEMS)
        gold_gems = self.gems.get(GemColor.GOLD, 0)
        
        # Check if player has enough gems in total
        if regular_gems + gold_gems < total_cost:
            return False
        
        # Check if player has enough of each color (considering gold as wild)
        remaining_gold = gold_gems
        for color, cost in effective_cost.items():
            available = self.gems.get(color, 0)
            if available < cost:
                # Need to use gold gems
                gold_needed = cost - available
                if remaining_gold < gold_needed:
                    return False
                remaining_gold -= gold_needed
        
        return True
    
    def get_affordable_payment(self, card: Card) -> Optional[Dict[GemColor, int]]:
        """
        Get a valid payment for a card if the player can afford it.
        
        This method returns one possible way to pay for the card, prioritizing
        using regular gems before gold gems.
        
        Args:
            card: The card to purchase
            
        Returns:
            Dictionary mapping gem colors to counts, or None if card can't be afforded
        """
        if not self.can_afford_card(card):
            return None
        
        # Calculate effective cost after applying bonuses
        effective_cost = {
            color: max(0, count - self.bonuses.get(color, 0))
            for color, count in card.cost.items()
        }
        
        # Start with using regular gems where possible
        payment = {}
        remaining_cost = {}
        
        for color, cost in effective_cost.items():
            available = self.gems.get(color, 0)
            if available >= cost:
                payment[color] = cost
            else:
                payment[color] = available
                remaining_cost[color] = cost - available
        
        # Use gold gems for any remaining cost
        if remaining_cost:
            gold_needed = sum(remaining_cost.values())
            payment[GemColor.GOLD] = gold_needed
        
        return payment
    
    def get_all_affordable_payments(self, card: Card) -> List[Dict[GemColor, int]]:
        """
        Get all possible ways to pay for a card.
        
        This method returns all valid combinations of gems that can be used
        to purchase the card, considering gold gems as wild.
        
        Args:
            card: The card to purchase
            
        Returns:
            List of dictionaries mapping gem colors to counts
        """
        if not self.can_afford_card(card):
            return []
        
        # Calculate effective cost after applying bonuses
        effective_cost = {
            color: max(0, count - self.bonuses.get(color, 0))
            for color, count in card.cost.items()
        }
        
        # Remove zero costs
        effective_cost = {color: cost for color, cost in effective_cost.items() if cost > 0}
        
        # Start with the base payment (using regular gems where possible)
        base_payment = {
            color: min(cost, self.gems.get(color, 0))
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
        available_gold = self.gems.get(GemColor.GOLD, 0)
        if available_gold < total_gold_needed:
            return []  # Can't afford the card
        
        # Generate all valid distributions of gold gems
        result = []
        self._generate_gold_distributions(
            result, base_payment.copy(), remaining_cost, available_gold, list(remaining_cost.keys()), 0
        )
        return result
    
    def _generate_gold_distributions(
        self,
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
                self._generate_gold_distributions(
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
    
    def can_reserve_card(self) -> bool:
        """
        Check if the player can reserve another card.
        
        Returns:
            True if the player can reserve a card, False otherwise
        """
        return len(self.reserved_cards) < MAX_RESERVED_CARDS
    
    def can_take_gem(self, color: GemColor, count: int = 1) -> bool:
        """
        Check if the player can take gems without exceeding the limit.
        
        Args:
            color: The gem color to take
            count: The number of gems to take
            
        Returns:
            True if the player can take the gems, False otherwise
        """
        current_total = sum(self.gems.values())
        return current_total + count <= MAX_GEMS_TOTAL
    
    def add_card(self, card: Card) -> None:
        """
        Add a purchased card to the player's collection.
        
        Updates bonuses and points.
        
        Args:
            card: The card to add
        """
        self.cards.append(card)
        self.bonuses[card.bonus] = self.bonuses.get(card.bonus, 0) + 1
        self.points += card.points
    
    def add_noble(self, noble: Noble) -> None:
        """
        Add a noble to the player's collection.
        
        Updates points.
        
        Args:
            noble: The noble to add
        """
        self.nobles.append(noble)
        self.points += noble.points
    
    def reserve_card(self, card: Card) -> None:
        """
        Reserve a card.
        
        Args:
            card: The card to reserve
        """
        if not self.can_reserve_card():
            raise ValueError("Cannot reserve more cards")
        self.reserved_cards.append(card)
    
    def add_gems(self, gems: Dict[GemColor, int]) -> None:
        """
        Add gems to the player's collection.
        
        Args:
            gems: Dictionary mapping gem colors to counts
        """
        for color, count in gems.items():
            self.gems[color] = self.gems.get(color, 0) + count
    
    def remove_gems(self, gems: Dict[GemColor, int]) -> None:
        """
        Remove gems from the player's collection.
        
        Args:
            gems: Dictionary mapping gem colors to counts
        """
        for color, count in gems.items():
            if self.gems.get(color, 0) < count:
                raise ValueError(f"Not enough {color.name} gems")
            self.gems[color] -= count
    
    def get_total_gems(self) -> int:
        """
        Get the total number of gems the player has.
        
        Returns:
            Total gem count
        """
        return sum(self.gems.values())
    
    def get_total_cards(self) -> int:
        """
        Get the total number of cards the player has purchased.
        
        Returns:
            Total card count
        """
        return len(self.cards)
    
    def get_total_nobles(self) -> int:
        """
        Get the total number of nobles the player has acquired.
        
        Returns:
            Total noble count
        """
        return len(self.nobles)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the player to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the player
        """
        return {
            "id": self.id,
            "name": self.name,
            "gems": {color.name: count for color, count in self.gems.items()},
            "cards": [card.id for card in self.cards],
            "reserved_cards": [card.id for card in self.reserved_cards],
            "nobles": [noble.id for noble in self.nobles],
            "points": self.points,
            "bonuses": {color.name: count for color, count in self.bonuses.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], card_lookup: Dict[int, Card], noble_lookup: Dict[int, Noble]) -> 'Player':
        """
        Create a player from a dictionary representation.
        
        Args:
            data: Dictionary representation of the player
            card_lookup: Dictionary mapping card IDs to Card objects
            noble_lookup: Dictionary mapping noble IDs to Noble objects
            
        Returns:
            Player object
        """
        # Convert gems from string keys to GemColor enum
        gems = {GemColor[color]: count for color, count in data["gems"].items()}
        
        # Look up cards and nobles by ID
        cards = [card_lookup[card_id] for card_id in data["cards"]]
        reserved_cards = [card_lookup[card_id] for card_id in data["reserved_cards"]]
        nobles = [noble_lookup[noble_id] for noble_id in data["nobles"]]
        
        # Convert bonuses from string keys to GemColor enum
        bonuses = {GemColor[color]: count for color, count in data["bonuses"].items()}
        
        return cls(
            id=data["id"],
            name=data["name"],
            gems=gems,
            cards=cards,
            reserved_cards=reserved_cards,
            nobles=nobles,
            points=data["points"],
            bonuses=bonuses,
        )
    
    def to_json(self) -> str:
        """
        Convert the player to a JSON string.
        
        Returns:
            JSON string representation of the player
        """
        return json.dumps(self.to_dict())
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the player.
        
        Returns:
            String representation
        """
        gem_str = ", ".join(f"{count} {color.name}" for color, count in self.gems.items() if count > 0)
        bonus_str = ", ".join(f"{count} {color.name}" for color, count in self.bonuses.items() if count > 0)
        
        return (
            f"Player {self.name} (ID: {self.id})\n"
            f"Points: {self.points}\n"
            f"Gems: {gem_str}\n"
            f"Bonuses: {bonus_str}\n"
            f"Cards: {len(self.cards)}\n"
            f"Reserved Cards: {len(self.reserved_cards)}\n"
            f"Nobles: {len(self.nobles)}"
        )
    
    def get_state_features(self) -> List[float]:
        """
        Get a feature vector representing the player's state for ML models.
        
        Returns:
            List of normalized features
        """
        features = []
        
        # Gems (normalized by MAX_GEMS_TOTAL)
        for color in ALL_GEMS:
            features.append(self.gems.get(color, 0) / MAX_GEMS_TOTAL)
        
        # Bonuses (normalized by an assumed max of 10)
        for color in REGULAR_GEMS:
            features.append(self.bonuses.get(color, 0) / 10.0)
        
        # Points (normalized by 15, which is typically the winning threshold)
        features.append(self.points / 15.0)
        
        # Card counts
        features.append(len(self.cards) / 20.0)  # Assume max 20 cards
        features.append(len(self.reserved_cards) / MAX_RESERVED_CARDS)
        features.append(len(self.nobles) / 5.0)  # Assume max 5 nobles
        
        return features
