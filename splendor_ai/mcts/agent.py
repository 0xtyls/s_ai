"""
Monte Carlo Tree Search Agent for Splendor.

This module provides the MCTSAgent class, which is a ready-to-use AI player
that uses Monte Carlo Tree Search to select actions in Splendor games.
The agent can be configured with different parameters and provides
statistics about its search process.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time
import random
import json
from pathlib import Path

from splendor_ai.core.game import GameState, Game
from splendor_ai.core.actions import Action
from splendor_ai.mcts.node import MCTSNode
from splendor_ai.mcts.config import MCTSConfig
from splendor_ai.mcts.search import (
    mcts_search, get_action_statistics, get_principal_variation
)


class MCTSAgent:
    """
    Monte Carlo Tree Search agent for playing Splendor.
    
    This agent uses MCTS to select actions in Splendor games. It can be
    configured with different parameters and provides statistics about
    its search process.
    """
    
    def __init__(
        self,
        config: Optional[MCTSConfig] = None,
        name: str = "MCTS Agent",
        verbose: bool = False
    ):
        """
        Initialize an MCTS agent.
        
        Args:
            config: MCTS configuration parameters
            name: Name of the agent
            verbose: Whether to print detailed information during search
        """
        self.config = config or MCTSConfig()
        self.name = name
        self.verbose = verbose
        
        # Statistics from the most recent search
        self.last_stats: Dict[str, Any] = {}
        
        # History of all actions and their statistics
        self.action_history: List[Tuple[Action, Dict[str, Any]]] = []
        
        # Root node of the last search
        self.last_root: Optional[MCTSNode] = None
    
    def select_action(self, state: GameState, player_id: int) -> Action:
        """
        Select an action using Monte Carlo Tree Search.
        
        Args:
            state: Current game state
            player_id: ID of the player making the decision
            
        Returns:
            Selected action
        """
        # Check if it's actually our turn
        if state.current_player != player_id:
            raise ValueError(f"Not player {player_id}'s turn")
        
        # Get valid actions
        valid_actions = state.get_valid_actions(player_id)
        
        # If there's only one valid action, no need to search
        if len(valid_actions) == 1:
            action = valid_actions[0]
            self.last_stats = {"iterations": 0, "forced_move": True}
            return action
        
        # Run MCTS search
        start_time = time.time()
        action, stats = mcts_search(state, player_id, self.config)
        stats["total_time"] = time.time() - start_time
        
        # Store statistics
        self.last_stats = stats
        
        # Store in history
        self.action_history.append((action, stats))
        
        # Print information if verbose
        if self.verbose:
            self._print_search_info(action, stats)
        
        return action
    
    def _print_search_info(self, action: Action, stats: Dict[str, Any]) -> None:
        """
        Print information about the search.
        
        Args:
            action: Selected action
            stats: Search statistics
        """
        print(f"\n{self.name} selected: {action}")
        print(f"Iterations: {stats['iterations']}")
        print(f"Time: {stats['time_elapsed']:.3f}s ({stats['iterations_per_second']:.1f} it/s)")
        print(f"Nodes: {stats['node_count']}")
        print(f"Max depth: {stats['max_depth']}")
        
        # Print top actions by visit count
        if 'action_visits' in stats:
            print("\nTop actions:")
            actions_by_visits = sorted(
                stats['action_visits'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for i, (action_str, visits) in enumerate(actions_by_visits[:5]):
                reward = stats['action_rewards'].get(action_str, 0)
                win_rate = reward / visits if visits > 0 else 0
                print(f"{i+1}. {action_str} - {visits} visits, {win_rate:.3f} value")
    
    def get_action_callback(self) -> Callable[[GameState, int], Action]:
        """
        Get a callback function for selecting actions.
        
        This is useful for registering the agent with a Game object.
        
        Returns:
            Callback function that takes a game state and player ID and returns an action
        """
        return lambda state, player_id: self.select_action(state, player_id)
    
    def register_with_game(self, game: Game, player_id: int) -> None:
        """
        Register this agent with a game.
        
        Args:
            game: Game object
            player_id: ID of the player to register as
        """
        game.register_agent(player_id, self.get_action_callback())
    
    def get_last_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from the most recent search.
        
        Returns:
            Dictionary of search statistics
        """
        return self.last_stats
    
    def get_principal_variation(self) -> List[Tuple[Action, float]]:
        """
        Get the principal variation (most visited path) from the last search.
        
        Returns:
            List of (action, value) pairs representing the principal variation
        """
        if self.last_root is None:
            return []
        
        return get_principal_variation(self.last_root)
    
    def get_action_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all actions from the last search.
        
        Returns:
            Dictionary mapping action strings to statistics
        """
        if self.last_root is None:
            return {}
        
        return get_action_statistics(self.last_root)
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.last_stats = {}
        self.action_history = []
        self.last_root = None
    
    def save_statistics(self, filename: str) -> None:
        """
        Save statistics to a file.
        
        Args:
            filename: Name of the file to save to
        """
        # Convert actions to strings for JSON serialization
        history = []
        for action, stats in self.action_history:
            history.append({
                "action": str(action),
                "stats": {k: v for k, v in stats.items() if not isinstance(v, dict)}
            })
        
        data = {
            "agent_name": self.name,
            "config": self.config.to_dict(),
            "history": history,
            "total_actions": len(self.action_history)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def __str__(self) -> str:
        """
        Get a string representation of the agent.
        
        Returns:
            String representation
        """
        return f"{self.name} (MCTS, {self.config.iterations} iterations)"


class MCTSAgentFactory:
    """
    Factory for creating MCTS agents with different configurations.
    
    This class provides methods for creating MCTS agents with different
    strengths and configurations.
    """
    
    @staticmethod
    def create_fast() -> MCTSAgent:
        """
        Create a fast MCTS agent with fewer iterations.
        
        Returns:
            MCTSAgent
        """
        config = MCTSConfig.fast()
        return MCTSAgent(config=config, name="Fast MCTS")
    
    @staticmethod
    def create_standard() -> MCTSAgent:
        """
        Create a standard MCTS agent with balanced parameters.
        
        Returns:
            MCTSAgent
        """
        config = MCTSConfig.default()
        return MCTSAgent(config=config, name="Standard MCTS")
    
    @staticmethod
    def create_strong() -> MCTSAgent:
        """
        Create a strong MCTS agent with more iterations.
        
        Returns:
            MCTSAgent
        """
        config = MCTSConfig.deep()
        return MCTSAgent(config=config, name="Strong MCTS")
    
    @staticmethod
    def create_custom(
        iterations: int = 1000,
        time_limit: Optional[float] = None,
        exploration_weight: float = 1.41,
        use_heuristics: bool = True,
        name: str = "Custom MCTS"
    ) -> MCTSAgent:
        """
        Create a custom MCTS agent.
        
        Args:
            iterations: Number of MCTS iterations
            time_limit: Optional time limit in seconds
            exploration_weight: UCB1 exploration parameter
            use_heuristics: Whether to use domain-specific heuristics
            name: Name of the agent
            
        Returns:
            MCTSAgent
        """
        config = MCTSConfig(
            iterations=iterations,
            time_limit=time_limit,
            exploration_weight=exploration_weight,
            use_heuristics=use_heuristics
        )
        return MCTSAgent(config=config, name=name)
