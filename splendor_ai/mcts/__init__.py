"""
Monte Carlo Tree Search (MCTS) implementation for Splendor.

This package provides a complete MCTS agent that can play Splendor without
any training. The MCTS algorithm works by:

1. Selection: Starting from the root node, select child nodes using UCB1 until reaching
   a leaf node or a node that hasn't been fully expanded.
2. Expansion: Create a new child node by taking a previously untried action.
3. Simulation: From the new node, perform a random playout to the end of the game.
4. Backpropagation: Update the statistics of all nodes in the path with the result.

The agent can be configured with different parameters to control the search depth,
exploration constant, and simulation strategy.
"""

from splendor_ai.mcts.node import MCTSNode
from splendor_ai.mcts.agent import MCTSAgent
from splendor_ai.mcts.search import (
    mcts_search, 
    select_node, 
    expand_node, 
    simulate_game, 
    backpropagate
)
from splendor_ai.mcts.config import MCTSConfig

# Default configuration
DEFAULT_CONFIG = MCTSConfig(
    iterations=1000,          # Number of MCTS iterations per move
    exploration_weight=1.41,  # UCB1 exploration parameter (sqrt(2))
    max_depth=100,            # Maximum depth for simulations
    time_limit=None,          # Optional time limit in seconds (None = no limit)
    use_heuristics=True,      # Whether to use domain-specific heuristics
    simulation_policy="random"  # Policy for simulation phase ("random" or "heuristic")
)

__all__ = [
    'MCTSAgent',
    'MCTSNode',
    'MCTSConfig',
    'mcts_search',
    'select_node',
    'expand_node',
    'simulate_game',
    'backpropagate',
    'DEFAULT_CONFIG'
]
