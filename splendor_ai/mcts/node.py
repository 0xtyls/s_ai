"""
Monte Carlo Tree Search Node for Splendor.

This module defines the MCTSNode class which represents a node in the MCTS tree.
Each node contains a game state, statistics (visits, wins), and manages child nodes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import math
import random
import time
from copy import deepcopy

from splendor_ai.core.game import GameState, GameResult
from splendor_ai.core.actions import Action
from splendor_ai.mcts.config import MCTSConfig


class MCTSNode:
    """
    A node in the Monte Carlo Tree Search.
    
    Each node represents a game state and tracks statistics about
    simulations that pass through it, including visit count and rewards.
    """
    
    def __init__(
        self,
        state: GameState,
        parent: Optional['MCTSNode'] = None,
        action: Optional[Action] = None,
        config: Optional[MCTSConfig] = None,
        player_id: int = -1,
    ):
        """
        Initialize an MCTS node.
        
        Args:
            state: The game state this node represents
            parent: The parent node (None for root)
            action: The action that led to this state (None for root)
            config: MCTS configuration parameters
            player_id: ID of the player who is making decisions at this node
        """
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.config = config or MCTSConfig()
        
        # The player who is making decisions at this node
        # If not specified, use the current player from the state
        self.player_id = player_id if player_id >= 0 else state.current_player
        
        # Node statistics
        self.visits = 0
        self.total_reward = 0.0
        self.children: List[MCTSNode] = []
        
        # Track which actions have been tried
        self._untried_actions: Optional[List[Action]] = None
        
        # For transposition table
        self._hash: Optional[int] = None
    
    @property
    def untried_actions(self) -> List[Action]:
        """
        Get the list of untried actions from this node.
        
        This property lazily computes the valid actions the first time
        it's accessed, then filters out actions that have already been tried.
        
        Returns:
            List of untried actions
        """
        if self._untried_actions is None:
            # Get all valid actions for the current player
            self._untried_actions = self.state.get_valid_actions(self.state.current_player)
            
            # Shuffle to ensure exploration
            random.shuffle(self._untried_actions)
        
        return self._untried_actions
    
    def has_untried_actions(self) -> bool:
        """
        Check if there are untried actions from this node.
        
        Returns:
            True if there are untried actions, False otherwise
        """
        return bool(self.untried_actions)
    
    def is_terminal(self) -> bool:
        """
        Check if this node represents a terminal game state.
        
        Returns:
            True if the game is over, False otherwise
        """
        return self.state.game_over
    
    def is_fully_expanded(self) -> bool:
        """
        Check if all possible actions from this node have been tried.
        
        Returns:
            True if all actions have been tried, False otherwise
        """
        # If using progressive widening, we might not fully expand nodes with too many children
        if self.config.progressive_widening:
            max_children = max(1, int(self.visits ** self.config.widening_factor))
            return len(self.children) >= max_children and len(self.children) >= len(self.untried_actions)
        
        # Otherwise, we're fully expanded when all actions have been tried
        return not self.has_untried_actions()
    
    def select_child(self) -> 'MCTSNode':
        """
        Select a child node using the UCB1 formula.
        
        This implements the selection phase of MCTS, choosing the child
        with the highest UCB1 value.
        
        Returns:
            Selected child node
        """
        # Ensure we have children to select from
        if not self.children:
            raise ValueError("Cannot select child from node with no children")
        
        # Select child with highest UCB1 value
        return max(self.children, key=self.ucb_score)
    
    def ucb_score(self, child: 'MCTSNode') -> float:
        """
        Calculate the UCB1 score for a child node.
        
        UCB1 = average_reward + exploration_weight * sqrt(ln(parent_visits) / child_visits)
        
        Args:
            child: Child node to calculate score for
            
        Returns:
            UCB1 score
        """
        # If the child has never been visited, treat it as having infinite value
        if child.visits == 0:
            return float('inf')
        
        # Calculate exploitation term (average reward)
        exploitation = child.total_reward / child.visits
        
        # Calculate exploration term
        exploration = math.sqrt(math.log(self.visits) / child.visits)
        
        # Combine with exploration weight
        return exploitation + self.config.exploration_weight * exploration
    
    def expand(self) -> Optional['MCTSNode']:
        """
        Expand the tree by adding a new child node.
        
        This implements the expansion phase of MCTS, selecting an untried action
        and creating a new child node.
        
        Returns:
            The new child node, or None if no expansion is possible
        """
        if not self.has_untried_actions() or self.is_terminal():
            return None
        
        # Get an untried action
        action = self.untried_actions.pop()
        
        # Create a new game state by applying the action
        new_state = self.state.clone()
        if not new_state.apply_action(new_state.current_player, action):
            # Invalid action, try another one
            return self.expand()
        
        # Advance to the next player
        new_state.next_turn()
        
        # Create a new child node
        child = MCTSNode(
            state=new_state,
            parent=self,
            action=action,
            config=self.config,
            player_id=new_state.current_player
        )
        
        # Add the child to our children list
        self.children.append(child)
        
        return child
    
    def update(self, result: Union[float, Dict[int, float]]) -> None:
        """
        Update the node statistics with a simulation result.
        
        This implements the backpropagation phase of MCTS.
        
        Args:
            result: The simulation result, either a single value or a dictionary
                   mapping player IDs to rewards
        """
        self.visits += 1
        
        # If result is a dictionary, extract the reward for this node's player
        if isinstance(result, dict):
            reward = result.get(self.player_id, 0.0)
        else:
            reward = result
        
        self.total_reward += reward
    
    def best_child(self, exploration_weight: float = 0.0) -> 'MCTSNode':
        """
        Select the best child node, optionally with exploration.
        
        When exploration_weight is 0, this selects the child with the
        highest average reward (exploitation only).
        
        Args:
            exploration_weight: Weight for the exploration term (0 = pure exploitation)
            
        Returns:
            Best child node
        """
        # Use a temporary exploration weight for UCB calculation
        original_weight = self.config.exploration_weight
        self.config.exploration_weight = exploration_weight
        
        # Select the best child
        best = max(self.children, key=self.ucb_score)
        
        # Restore the original exploration weight
        self.config.exploration_weight = original_weight
        
        return best
    
    def best_action(self) -> Optional[Action]:
        """
        Get the best action from this node based on visit counts.
        
        This is typically called at the root node to determine the final move.
        
        Returns:
            The best action, or None if no children
        """
        if not self.children:
            return None
        
        # Select the child with the most visits
        # This is more robust than using average reward
        best_child = max(self.children, key=lambda c: c.visits)
        return best_child.action
    
    def tree_policy(self) -> 'MCTSNode':
        """
        Execute the tree policy to select a node for expansion or simulation.
        
        This combines the selection and expansion phases of MCTS.
        
        Returns:
            Selected node
        """
        current = self
        
        # Traverse the tree until we find a node to expand or simulate from
        while not current.is_terminal():
            if not current.is_fully_expanded():
                # If not fully expanded, expand by trying a new action
                expanded = current.expand()
                if expanded:
                    return expanded
            
            # If fully expanded or expansion failed, select the best child
            current = current.select_child()
        
        return current
    
    def simulation_policy(self, state: GameState) -> Action:
        """
        Select an action for the simulation phase.
        
        This can be a random policy or a heuristic policy depending on configuration.
        
        Args:
            state: Current game state
            
        Returns:
            Selected action
        """
        valid_actions = state.get_valid_actions(state.current_player)
        
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        if self.config.simulation_policy == "random":
            # Random policy
            return random.choice(valid_actions)
        
        elif self.config.simulation_policy == "heuristic":
            # Simple heuristic policy for Splendor
            # Prioritize:
            # 1. Purchasing cards that give points
            # 2. Reserving high-point cards
            # 3. Taking gems that help purchase affordable cards
            # 4. Random action as fallback
            
            # Get the current player
            player = state.players[state.current_player]
            
            # Group actions by type
            purchase_actions = []
            reserve_actions = []
            take_gems_actions = []
            
            for action in valid_actions:
                if action.action_type.name == "PURCHASE_CARD":
                    purchase_actions.append(action)
                elif action.action_type.name == "RESERVE_CARD":
                    reserve_actions.append(action)
                elif action.action_type.name == "TAKE_GEMS":
                    take_gems_actions.append(action)
            
            # 1. Try to purchase a card with points
            if purchase_actions:
                # Find cards with points
                point_cards = []
                for action in purchase_actions:
                    # Find the card
                    card = None
                    if action.from_reserved:
                        for c in player.reserved_cards:
                            if c.id == action.card_id:
                                card = c
                                break
                    else:
                        for tier_cards in state.card_tiers.values():
                            for c in tier_cards:
                                if c.id == action.card_id:
                                    card = c
                                    break
                            if card:
                                break
                    
                    if card and card.points > 0:
                        point_cards.append((action, card.points))
                
                if point_cards:
                    # Choose the card with the most points
                    return max(point_cards, key=lambda x: x[1])[0]
                
                # If no point cards, choose a random purchase
                return random.choice(purchase_actions)
            
            # 2. Try to reserve a high-point card
            if reserve_actions and random.random() < 0.7:  # 70% chance to reserve
                return random.choice(reserve_actions)
            
            # 3. Take gems
            if take_gems_actions:
                return random.choice(take_gems_actions)
            
            # Fallback to random
            return random.choice(valid_actions)
        
        else:
            # Default to random
            return random.choice(valid_actions)
    
    def simulate(self) -> Union[float, Dict[int, float]]:
        """
        Run a simulation from this node to a terminal state.
        
        This implements the simulation phase of MCTS.
        
        Returns:
            Simulation result (reward)
        """
        # Clone the state to avoid modifying the original
        state = self.state.clone()
        
        # Run the simulation until the game ends or we reach max depth
        depth = 0
        while not state.game_over and depth < self.config.max_depth:
            # Select an action using the simulation policy
            action = self.simulation_policy(state)
            
            # Apply the action
            state.apply_action(state.current_player, action)
            
            # Move to the next player
            state.next_turn()
            
            depth += 1
        
        # Calculate rewards based on the final state
        if state.game_over:
            # Game ended naturally
            if state.result == GameResult.WINNER:
                # Winner gets 1.0, others get 0.0
                rewards = {player.id: 0.0 for player in state.players}
                if state.winner is not None:
                    rewards[state.winner] = 1.0
                return rewards
            else:
                # Draw - everyone gets 0.5
                return {player.id: 0.5 for player in state.players}
        else:
            # Reached max depth - evaluate the current state
            return self._evaluate_state(state)
    
    def _evaluate_state(self, state: GameState) -> Dict[int, float]:
        """
        Evaluate a non-terminal game state.
        
        This is used when a simulation reaches the maximum depth without
        reaching a terminal state.
        
        Args:
            state: Game state to evaluate
            
        Returns:
            Dictionary mapping player IDs to estimated rewards
        """
        # Simple heuristic: normalize points relative to victory threshold
        victory_points = 15  # Standard Splendor victory points
        
        rewards = {}
        max_points = 0
        
        # Find the maximum points among all players
        for player in state.players:
            max_points = max(max_points, player.points)
        
        # Calculate rewards based on points
        for player in state.players:
            # Base reward on points (0.0 to 0.8)
            reward = 0.8 * (player.points / victory_points)
            
            # Bonus for having the most points (up to 0.2)
            if player.points == max_points and max_points > 0:
                reward += 0.2
            
            rewards[player.id] = reward
        
        return rewards
    
    def __hash__(self) -> int:
        """
        Calculate a hash for the node based on the game state.
        
        This is used for transposition tables.
        
        Returns:
            Hash value
        """
        if self._hash is None:
            # Create a hash based on:
            # - Current player
            # - Player gem counts
            # - Player card counts and bonuses
            # - Visible cards on the board
            
            # Start with the current player
            h = hash(self.state.current_player)
            
            # Add player information
            for player in self.state.players:
                # Gems
                for color, count in player.gems.items():
                    h = h ^ hash((player.id, 'gems', color, count))
                
                # Bonuses
                for color, count in player.bonuses.items():
                    h = h ^ hash((player.id, 'bonuses', color, count))
                
                # Points
                h = h ^ hash((player.id, 'points', player.points))
            
            # Add visible cards (just their IDs)
            for tier, cards in self.state.card_tiers.items():
                for card in cards:
                    h = h ^ hash(('board', tier, card.id))
            
            self._hash = h
        
        return self._hash
    
    def __eq__(self, other: object) -> bool:
        """
        Check if two nodes are equal based on their hash.
        
        Args:
            other: Node to compare with
            
        Returns:
            True if the nodes are equal, False otherwise
        """
        if not isinstance(other, MCTSNode):
            return False
        return hash(self) == hash(other)
    
    def __str__(self) -> str:
        """
        Get a string representation of the node.
        
        Returns:
            String representation
        """
        return (f"MCTSNode(player={self.player_id}, "
                f"visits={self.visits}, "
                f"reward={self.total_reward:.2f}, "
                f"children={len(self.children)}, "
                f"untried={len(self.untried_actions) if self._untried_actions is not None else 'unknown'})")
