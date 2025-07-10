"""
Cards and Nobles for the Splendor game.

This module defines the data structures for development cards and nobles,
along with factory functions to create decks and utility functions for card operations.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import random
from collections import Counter

from splendor_ai.core.constants import (
    GemColor, CardTier, REGULAR_GEMS, CARDS_PER_TIER,
    TIER_POINT_RANGES, NOBLE_POINTS, NOBLE_REQUIREMENT_MIN,
    NOBLE_REQUIREMENT_MAX, NOBLE_REQUIRED_COLORS, TOTAL_NOBLES
)


@dataclass(frozen=True)
class Card:
    """
    Represents a development card in Splendor.
    
    Each card has a tier, prestige points, a gem cost, and provides a permanent gem bonus.
    """
    id: int  # Unique identifier
    tier: CardTier  # Card tier (1, 2, or 3)
    points: int  # Prestige points
    cost: Dict[GemColor, int]  # Cost in gems
    bonus: GemColor  # Gem color this card provides as bonus
    
    def __post_init__(self):
        """Validate the card after initialization."""
        # Ensure cost doesn't include gold (gold is only used as a joker when paying)
        if GemColor.GOLD in self.cost:
            raise ValueError("Card cost cannot include gold gems")
        
        # Ensure bonus is a regular gem (not gold)
        if self.bonus == GemColor.GOLD:
            raise ValueError("Card bonus cannot be gold")
        
        # Validate points are within range for tier
        min_points, max_points = TIER_POINT_RANGES[self.tier]
        if not min_points <= self.points <= max_points:
            raise ValueError(
                f"Card points ({self.points}) outside valid range "
                f"({min_points}-{max_points}) for tier {self.tier.value}"
            )
    
    @property
    def total_cost(self) -> int:
        """Get the total number of gems required to purchase this card."""
        return sum(self.cost.values())
    
    def can_afford(self, gems: Dict[GemColor, int], bonuses: Dict[GemColor, int]) -> bool:
        """
        Check if the card can be purchased with the given gems and bonuses.
        
        Args:
            gems: Dictionary of gem colors to counts the player has
            bonuses: Dictionary of gem color bonuses from previously purchased cards
            
        Returns:
            True if the card can be purchased, False otherwise
        """
        # Calculate effective cost after applying bonuses
        effective_cost = {
            color: max(0, count - bonuses.get(color, 0))
            for color, count in self.cost.items()
        }
        
        # Check if regular gems are sufficient
        remaining_cost = 0
        for color, count in effective_cost.items():
            if gems.get(color, 0) >= count:
                # Player has enough of this specific gem
                continue
            else:
                # Track how many gems are missing
                remaining_cost += count - gems.get(color, 0)
        
        # Check if gold gems can cover the remaining cost
        return gems.get(GemColor.GOLD, 0) >= remaining_cost
    
    def __str__(self) -> str:
        """String representation of the card."""
        cost_str = ", ".join(f"{count} {color.name}" for color, count in self.cost.items())
        return (f"Card(Tier: {self.tier.value}, Points: {self.points}, "
                f"Bonus: {self.bonus.name}, Cost: {cost_str})")


@dataclass(frozen=True)
class Noble:
    """
    Represents a noble in Splendor.
    
    Nobles provide prestige points when a player meets their requirements.
    Requirements are based on the player's gem bonuses (cards), not their gems.
    """
    id: int  # Unique identifier
    points: int = NOBLE_POINTS  # Always 3 points in the standard game
    requirements: Dict[GemColor, int] = field(default_factory=dict)  # Required gem bonuses
    
    def __post_init__(self):
        """Validate the noble after initialization."""
        # Ensure requirements don't include gold
        if GemColor.GOLD in self.requirements:
            raise ValueError("Noble requirements cannot include gold")
        
        # Validate requirements are within range
        for color, count in self.requirements.items():
            if not NOBLE_REQUIREMENT_MIN <= count <= NOBLE_REQUIREMENT_MAX:
                raise ValueError(
                    f"Noble requirement for {color.name} ({count}) outside valid range "
                    f"({NOBLE_REQUIREMENT_MIN}-{NOBLE_REQUIREMENT_MAX})"
                )
    
    def can_visit(self, bonuses: Dict[GemColor, int]) -> bool:
        """
        Check if the noble can visit a player with the given bonuses.
        
        Args:
            bonuses: Dictionary of gem color bonuses from purchased cards
            
        Returns:
            True if the noble can visit, False otherwise
        """
        for color, required in self.requirements.items():
            if bonuses.get(color, 0) < required:
                return False
        return True
    
    def __str__(self) -> str:
        """String representation of the noble."""
        req_str = ", ".join(f"{count} {color.name}" for color, count in self.requirements.items())
        return f"Noble(Points: {self.points}, Requirements: {req_str})"


def create_balanced_card_deck(tier: CardTier, num_cards: Optional[int] = None) -> List[Card]:
    """
    Create a balanced deck of cards for the given tier.
    
    This function creates cards with a distribution similar to the actual Splendor game.
    
    Args:
        tier: The card tier (1, 2, or 3)
        num_cards: Optional number of cards to create (defaults to standard count for tier)
        
    Returns:
        List of Card objects
    """
    if num_cards is None:
        num_cards = CARDS_PER_TIER[tier]
    
    min_points, max_points = TIER_POINT_RANGES[tier]
    cards = []
    card_id = 1000 * tier.value  # Unique ID prefix based on tier
    
    # Create a balanced distribution of points
    if tier == CardTier.TIER_1:
        # Tier 1: mostly 0 points, some 1 point
        points_distribution = [0] * 30 + [1] * 10
    elif tier == CardTier.TIER_2:
        # Tier 2: mix of 1, 2, and 3 points
        points_distribution = [1] * 10 + [2] * 15 + [3] * 5
    else:  # TIER_3
        # Tier 3: mix of 3, 4, and 5 points
        points_distribution = [3] * 8 + [4] * 8 + [5] * 4
    
    # Ensure we have enough point values
    while len(points_distribution) < num_cards:
        points_distribution.extend(points_distribution[:num_cards - len(points_distribution)])
    
    # NOTE: Do *not* shuffle the points distribution – we want deterministic
    # decks for reproducibility in unit-tests and self-play evaluation.

    # Create cards with balanced bonuses (each gem color gets equal representation)
    for i in range(num_cards):
        card_id += 1
        points = points_distribution[i]
        
        # Assign bonus color (cycle through all regular gems)
        bonus_color = REGULAR_GEMS[i % len(REGULAR_GEMS)]
        
        # Create a balanced cost
        cost = {}
        total_cost = 0
        
        # Higher tier cards are more expensive
        max_total_cost = 3 + 2 * tier.value  # Tier 1: 5, Tier 2: 7, Tier 3: 9
        min_total_cost = 2 + tier.value  # Tier 1: 3, Tier 2: 4, Tier 3: 5
        
        # Target total cost based on points (higher points = higher cost)
        target_cost = min_total_cost + ((points - min_points) / 
                                       (max_points - min_points or 1)) * (max_total_cost - min_total_cost)
        target_cost = int(target_cost)
        
        # Distribute cost among 2-3 colors (excluding the bonus color)
        # Deterministic list of colours (keep original ordering)
        available_colors = [c for c in REGULAR_GEMS if c != bonus_color]
        
        # Number of colors in the cost (2-3 for lower tiers, 3-4 for higher tiers)
        num_colors = min(2 + tier.value // 2, len(available_colors))
        cost_colors = available_colors[:num_colors]
        
        # Evenly (deterministically) distribute total cost across selected colours
        base_cost = target_cost // num_colors
        leftover  = target_cost % num_colors

        for j, color in enumerate(cost_colors):
            # First `leftover` colours get an extra gem to reach the target sum
            color_cost = base_cost + (1 if j < leftover else 0)
            if color_cost > 0:
                cost[color] = color_cost
        
        cards.append(Card(
            id=card_id,
            tier=tier,
            points=points,
            cost=cost,
            bonus=bonus_color
        ))
    
    return cards


def create_standard_card_decks() -> Dict[CardTier, List[Card]]:
    """
    Create standard decks for all three tiers.
    
    Returns:
        Dictionary mapping tiers to lists of cards
    """
    return {
        tier: create_balanced_card_deck(tier)
        for tier in CardTier
    }


def create_noble_deck(num_nobles: int = TOTAL_NOBLES) -> List[Noble]:
    """
    Create a deck of nobles with balanced requirements.
    
    Args:
        num_nobles: Number of nobles to create
        
    Returns:
        List of Noble objects
    """
    nobles = []
    
    for i in range(num_nobles):
        # Each noble requires 3 colors, with 3-4 of each color
        requirements = {}
        
        # Select 3 random colors
        colors = random.sample(REGULAR_GEMS, NOBLE_REQUIRED_COLORS)
        
        for color in colors:
            # Each color requires 3 or 4 cards
            requirements[color] = random.randint(NOBLE_REQUIREMENT_MIN, NOBLE_REQUIREMENT_MAX)
        
        nobles.append(Noble(
            id=i + 1,
            requirements=requirements
        ))
    
    return nobles


def get_historical_cards() -> Dict[CardTier, List[Card]]:
    """
    Get a predefined set of cards that match the actual Splendor game.
    
    This function creates cards with costs and bonuses that closely match
    the distribution in the physical game.
    
    Returns:
        Dictionary mapping tiers to lists of cards
    """
    # This is a simplified version - a full implementation would include
    # the exact 90 cards from the game with precise costs
    
    # Tier 1 cards (40 cards, 0-1 points)
    tier1_cards = []
    card_id = 1001
    
    # Create 8 cards for each gem color (40 total)
    for bonus_color in REGULAR_GEMS:
        for _ in range(8):
            # Determine points (mostly 0, some 1)
            points = 1 if random.random() < 0.25 else 0
            
            # Create cost (3-4 total gems, spread across 2-3 colors)
            cost = {}
            available_colors = [c for c in REGULAR_GEMS if c != bonus_color]
            num_cost_colors = random.randint(2, 3)
            cost_colors = random.sample(available_colors, num_cost_colors)
            
            # Distribute 3-4 gems across the cost colors
            total_cost = random.randint(3, 4)
            remaining = total_cost
            
            for i, color in enumerate(cost_colors):
                if i == len(cost_colors) - 1:
                    cost[color] = remaining
                else:
                    cost_value = random.randint(1, remaining - (len(cost_colors) - i - 1))
                    cost[color] = cost_value
                    remaining -= cost_value
            
            tier1_cards.append(Card(
                id=card_id,
                tier=CardTier.TIER_1,
                points=points,
                cost=cost,
                bonus=bonus_color
            ))
            card_id += 1
    
    # Tier 2 cards (30 cards, 1-3 points)
    tier2_cards = []
    card_id = 2001
    
    # Create 6 cards for each gem color (30 total)
    for bonus_color in REGULAR_GEMS:
        for _ in range(6):
            # Determine points (1-3)
            points = random.randint(1, 3)
            
            # Create cost (5-6 total gems, spread across 2-3 colors)
            cost = {}
            available_colors = [c for c in REGULAR_GEMS if c != bonus_color]
            num_cost_colors = random.randint(2, 3)
            cost_colors = random.sample(available_colors, num_cost_colors)
            
            # Distribute 5-6 gems across the cost colors
            total_cost = random.randint(5, 6)
            remaining = total_cost
            
            for i, color in enumerate(cost_colors):
                if i == len(cost_colors) - 1:
                    cost[color] = remaining
                else:
                    cost_value = random.randint(1, remaining - (len(cost_colors) - i - 1))
                    cost[color] = cost_value
                    remaining -= cost_value
            
            tier2_cards.append(Card(
                id=card_id,
                tier=CardTier.TIER_2,
                points=points,
                cost=cost,
                bonus=bonus_color
            ))
            card_id += 1
    
    # Tier 3 cards (20 cards, 3-5 points)
    tier3_cards = []
    card_id = 3001
    
    # Create 4 cards for each gem color (20 total)
    for bonus_color in REGULAR_GEMS:
        for _ in range(4):
            # Determine points (3-5)
            points = random.randint(3, 5)
            
            # Create cost (7-10 total gems, spread across 3-4 colors)
            cost = {}
            available_colors = [c for c in REGULAR_GEMS if c != bonus_color]
            num_cost_colors = random.randint(3, 4)
            cost_colors = random.sample(available_colors, num_cost_colors)
            
            # Distribute 7-10 gems across the cost colors
            total_cost = random.randint(7, 10)
            remaining = total_cost
            
            for i, color in enumerate(cost_colors):
                if i == len(cost_colors) - 1:
                    cost[color] = remaining
                else:
                    max_cost = min(5, remaining - (len(cost_colors) - i - 1))
                    cost_value = random.randint(1, max_cost)
                    cost[color] = cost_value
                    remaining -= cost_value
            
            tier3_cards.append(Card(
                id=card_id,
                tier=CardTier.TIER_3,
                points=points,
                cost=cost,
                bonus=bonus_color
            ))
            card_id += 1
    
    return {
        CardTier.TIER_1: tier1_cards,
        CardTier.TIER_2: tier2_cards,
        CardTier.TIER_3: tier3_cards,
    }


def get_historical_nobles() -> List[Noble]:
    """
    Get a predefined set of nobles that match the actual Splendor game.
    
    Returns:
        List of Noble objects
    """
    # These are the 10 nobles from the actual game
    # Each requires exactly 3 colors with exactly 4 of each color
    nobles = [
        Noble(id=1, requirements={GemColor.WHITE: 4, GemColor.RED: 4, GemColor.BLACK: 4}),
        Noble(id=2, requirements={GemColor.WHITE: 4, GemColor.BLUE: 4, GemColor.GREEN: 4}),
        Noble(id=3, requirements={GemColor.BLUE: 4, GemColor.GREEN: 4, GemColor.RED: 4}),
        Noble(id=4, requirements={GemColor.GREEN: 4, GemColor.RED: 4, GemColor.BLACK: 4}),
        Noble(id=5, requirements={GemColor.WHITE: 4, GemColor.BLUE: 4, GemColor.BLACK: 4}),
        Noble(id=6, requirements={GemColor.WHITE: 3, GemColor.BLUE: 3, GemColor.BLACK: 3}),
        Noble(id=7, requirements={GemColor.WHITE: 3, GemColor.GREEN: 3, GemColor.RED: 3}),
        Noble(id=8, requirements={GemColor.BLUE: 3, GemColor.GREEN: 3, GemColor.BLACK: 3}),
        Noble(id=9, requirements={GemColor.WHITE: 3, GemColor.GREEN: 3, GemColor.BLACK: 3}),
        Noble(id=10, requirements={GemColor.BLUE: 3, GemColor.RED: 3, GemColor.BLACK: 3}),
    ]
    return nobles


def filter_affordable_cards(cards: List[Card], 
                           gems: Dict[GemColor, int], 
                           bonuses: Dict[GemColor, int]) -> List[Card]:
    """
    Filter a list of cards to only those that can be purchased.
    
    Args:
        cards: List of cards to filter
        gems: Dictionary of gem colors to counts
        bonuses: Dictionary of gem color bonuses
        
    Returns:
        List of affordable cards
    """
    return [card for card in cards if card.can_afford(gems, bonuses)]
