"""
Monte Carlo Tree Search (MCTS) algorithm for Splendor.

This module implements the core MCTS algorithm with the four standard phases:
1. Selection: Traverse the tree to find a promising node
2. Expansion: Create a new child node
3. Simulation: Run a playout to estimate the node's value
4. Backpropagation: Update statistics up the tree

The implementation includes optimizations like transposition tables
and configurable simulation strategies.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Set, Any
import time
import random
from collections import defaultdict

from splendor_ai.core.game import GameState, GameResult
from splendor_ai.core.actions import Action
from splendor_ai.mcts.node import MCTSNode
from splendor_ai.mcts.config import MCTSConfig


def mcts_search(
    state: GameState,
    player_id: int,
    config: Optional[MCTSConfig] = None
) -> Tuple[Action, Dict[str, Any]]:
    """
    Run Monte Carlo Tree Search to find the best action.
    
    This function runs the full MCTS algorithm:
    1. Create a root node from the current state
    2. Repeatedly run selection, expansion, simulation, and backpropagation
    3. Return the best action based on visit counts
    
    Args:
        state: Current game state
        player_id: ID of the player making the decision
        config: MCTS configuration parameters
        
    Returns:
        Tuple of (best action, search statistics)
    """
    # Use default config if none provided
    if config is None:
        config = MCTSConfig()
    
    # Create the root node
    root = MCTSNode(state=state, player_id=player_id, config=config)
    
    # Initialize transposition table if enabled
    transposition_table = {} if config.use_transposition_table else None
    
    # Track statistics
    stats = {
        "iterations": 0,
        "max_depth": 0,
        "total_simulation_steps": 0,
        "time_elapsed": 0,
        "node_count": 1,  # Start with the root
        "action_visits": defaultdict(int),
        "action_rewards": defaultdict(float),
    }
    
    # Start time
    start_time = time.time()
    
    # Main MCTS loop
    for i in range(config.iterations):
        # Check time limit if specified
        if config.time_limit is not None and time.time() - start_time > config.time_limit:
            stats["stopped_early"] = True
            break
        
        # 1. Selection & Expansion: Find a node to simulate from
        selected_node = select_node(root, transposition_table)
        
        # 2. Simulation: Run a playout from the selected node
        simulation_result, simulation_steps = simulate_game(selected_node, config)
        
        # 3. Backpropagation: Update statistics up the tree
        backpropagate(selected_node, simulation_result)
        
        # Update statistics
        stats["iterations"] += 1
        stats["total_simulation_steps"] += simulation_steps
        stats["max_depth"] = max(stats["max_depth"], simulation_steps)
        
        # Update node count
        if i % 10 == 0:  # Only update occasionally for efficiency
            stats["node_count"] = count_nodes(root)
    
    # Find the best action
    best_action = root.best_action()
    
    # Record statistics about each action
    for child in root.children:
        if child.action:
            action_str = str(child.action)
            stats["action_visits"][action_str] = child.visits
            if child.visits > 0:
                stats["action_rewards"][action_str] = child.total_reward / child.visits
    
    # Calculate final statistics
    stats["time_elapsed"] = time.time() - start_time
    stats["iterations_per_second"] = stats["iterations"] / max(0.001, stats["time_elapsed"])
    stats["average_simulation_steps"] = stats["total_simulation_steps"] / max(1, stats["iterations"])
    
    if not best_action:
        # Fallback to random action if MCTS failed to find a good move
        valid_actions = state.get_valid_actions(player_id)
        if valid_actions:
            best_action = random.choice(valid_actions)
            stats["used_fallback"] = True
    
    return best_action, stats


def select_node(
    root: MCTSNode,
    transposition_table: Optional[Dict[int, MCTSNode]] = None
) -> MCTSNode:
    """
    Select a node for expansion or simulation.
    
    This function implements the selection and expansion phases of MCTS.
    It traverses the tree using the UCB1 formula until it finds a node
    that hasn't been fully expanded, then expands it.
    
    Args:
        root: Root node of the MCTS tree
        transposition_table: Optional transposition table for node reuse
        
    Returns:
        Node selected for simulation
    """
    # Use the tree policy to select a node
    current = root.tree_policy()
    
    # Check transposition table if enabled
    if transposition_table is not None and not current.is_terminal():
        node_hash = hash(current)
        if node_hash in transposition_table:
            # We've seen this state before, reuse the node
            existing_node = transposition_table[node_hash]
            
            # Only reuse if it has been visited more than the current node
            if existing_node.visits > current.visits:
                current = existing_node
        else:
            # Add to transposition table
            transposition_table[node_hash] = current
    
    return current


def expand_node(node: MCTSNode) -> Optional[MCTSNode]:
    """
    Expand a node by adding a child.
    
    This function implements the expansion phase of MCTS.
    It selects an untried action and creates a new child node.
    
    Args:
        node: Node to expand
        
    Returns:
        New child node, or None if expansion is not possible
    """
    # This is a wrapper around the node's expand method
    return node.expand()


def simulate_game(
    node: MCTSNode,
    config: MCTSConfig
) -> Tuple[Union[float, Dict[int, float]], int]:
    """
    Run a simulation from a node to estimate its value.
    
    This function implements the simulation phase of MCTS.
    It runs a playout from the node using a simulation policy,
    and returns the result.
    
    Args:
        node: Node to simulate from
        config: MCTS configuration parameters
        
    Returns:
        Tuple of (simulation result, number of steps)
    """
    # If the node is terminal, no need to simulate
    if node.is_terminal():
        # Calculate rewards based on the final state
        state = node.state
        if state.result == GameResult.WINNER:
            # Winner gets 1.0, others get 0.0
            rewards = {player.id: 0.0 for player in state.players}
            if state.winner is not None:
                rewards[state.winner] = 1.0
            return rewards, 0
        else:
            # Draw - everyone gets 0.5
            return {player.id: 0.5 for player in state.players}, 0
    
    # Clone the state to avoid modifying the original
    state = node.state.clone()
    
    # Run the simulation until the game ends or we reach max depth
    steps = 0
    while not state.game_over and steps < config.max_depth:
        # Select an action using the simulation policy
        valid_actions = state.get_valid_actions(state.current_player)
        if not valid_actions:
            break
        
        # Use the node's simulation policy
        action = node.simulation_policy(state)
        
        # Apply the action
        state.apply_action(state.current_player, action)
        
        # Move to the next player
        state.next_turn()
        
        steps += 1
    
    # Calculate rewards based on the final state
    if state.game_over:
        # Game ended naturally
        if state.result == GameResult.WINNER:
            # Winner gets 1.0, others get 0.0
            rewards = {player.id: 0.0 for player in state.players}
            if state.winner is not None:
                rewards[state.winner] = 1.0
            return rewards, steps
        else:
            # Draw - everyone gets 0.5
            return {player.id: 0.5 for player in state.players}, steps
    else:
        # Reached max depth - evaluate the current state
        return _evaluate_state(state), steps


def _evaluate_state(state: GameState) -> Dict[int, float]:
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


def backpropagate(
    node: MCTSNode,
    result: Union[float, Dict[int, float]]
) -> None:
    """
    Update statistics up the tree.
    
    This function implements the backpropagation phase of MCTS.
    It updates the visit count and reward for each node in the path
    from the simulated node to the root.
    
    Args:
        node: Node to start backpropagation from
        result: Simulation result
    """
    # Update statistics for this node and all its ancestors
    current = node
    while current is not None:
        current.update(result)
        current = current.parent


def count_nodes(node: MCTSNode) -> int:
    """
    Count the total number of nodes in the tree.
    
    Args:
        node: Root node of the tree
        
    Returns:
        Total number of nodes
    """
    count = 1  # Count this node
    for child in node.children:
        count += count_nodes(child)
    return count


def get_principal_variation(root: MCTSNode, max_depth: int = 10) -> List[Tuple[Action, float]]:
    """
    Get the principal variation (most visited path) from the root.
    
    This is useful for analysis and debugging.
    
    Args:
        root: Root node of the MCTS tree
        max_depth: Maximum depth to explore
        
    Returns:
        List of (action, value) pairs representing the principal variation
    """
    result = []
    current = root
    depth = 0
    
    while current.children and depth < max_depth:
        # Find the child with the most visits
        best_child = max(current.children, key=lambda c: c.visits)
        
        # Calculate the value (win rate)
        value = best_child.total_reward / max(1, best_child.visits)
        
        # Add to the result
        if best_child.action:
            result.append((best_child.action, value))
        
        # Move to the best child
        current = best_child
        depth += 1
    
    return result


def get_action_statistics(root: MCTSNode) -> Dict[str, Dict[str, float]]:
    """
    Get statistics for all actions from the root.
    
    This is useful for analysis and debugging.
    
    Args:
        root: Root node of the MCTS tree
        
    Returns:
        Dictionary mapping action strings to statistics
    """
    result = {}
    
    for child in root.children:
        if child.action:
            action_str = str(child.action)
            result[action_str] = {
                "visits": child.visits,
                "reward": child.total_reward,
                "value": child.total_reward / max(1, child.visits),
                "exploration": root.ucb_score(child) if child.visits > 0 else float('inf')
            }
    
    return result
