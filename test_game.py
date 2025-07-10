#!/usr/bin/env python
"""
Test script for the Splendor game and MCTS agent.

This script runs a test game between MCTS agents to verify that the
game engine and AI are working correctly.
"""
import argparse
import time
import random
from typing import List, Optional, Dict, Any

from splendor_ai.core.game import Game, GameResult
from splendor_ai.core.constants import GemColor, CardTier
from splendor_ai.mcts.agent import MCTSAgent, MCTSAgentFactory
from splendor_ai.mcts.config import MCTSConfig


def display_game_state(game: Game) -> None:
    """
    Display the current game state in a readable format.
    
    Args:
        game: Game object
    """
    state = game.state
    
    print("\n" + "="*80)
    print(f"Turn {state.turn_count}, Player {state.current_player}'s turn")
    print("="*80)
    
    # Display gem pool
    print("\nGem Pool:")
    for color in GemColor:
        count = state.gem_pool.get(color, 0)
        if count > 0:
            print(f"  {color.name}: {count}")
    
    # Display nobles
    print("\nNobles:")
    for noble in state.nobles:
        req_str = ", ".join(f"{count} {color.name}" for color, count in noble.requirements.items())
        print(f"  Noble {noble.id} (3 points) - Requires: {req_str}")
    
    # Display card tiers
    print("\nCards:")
    for tier in CardTier:
        print(f"\n  Tier {tier.value}:")
        for i, card in enumerate(state.card_tiers.get(tier, [])):
            cost_str = ", ".join(f"{count} {color.name}" for color, count in card.cost.items())
            print(f"    Card {card.id} - {card.points} points, {card.bonus.name} bonus, Cost: {cost_str}")
    
    # Display players
    print("\nPlayers:")
    for i, player in enumerate(state.players):
        print(f"\n  Player {player.id} ({player.name}):")
        print(f"    Points: {player.points}")
        
        # Display gems
        gem_str = ", ".join(f"{count} {color.name}" for color, count in player.gems.items() if count > 0)
        print(f"    Gems: {gem_str or 'None'}")
        
        # Display bonuses
        bonus_str = ", ".join(f"{count} {color.name}" for color, count in player.bonuses.items() if count > 0)
        print(f"    Bonuses: {bonus_str or 'None'}")
        
        # Display cards
        if player.cards:
            print(f"    Cards ({len(player.cards)}):")
            for card in player.cards:
                print(f"      Card {card.id} - {card.points} points, {card.bonus.name} bonus")
        else:
            print("    Cards: None")
        
        # Display reserved cards
        if player.reserved_cards:
            print(f"    Reserved Cards ({len(player.reserved_cards)}):")
            for card in player.reserved_cards:
                print(f"      Card {card.id} - {card.points} points, {card.bonus.name} bonus")
        else:
            print("    Reserved Cards: None")
        
        # Display nobles
        if player.nobles:
            print(f"    Nobles ({len(player.nobles)}):")
            for noble in player.nobles:
                print(f"      Noble {noble.id} - {noble.points} points")
        else:
            print("    Nobles: None")
        
        # Indicate current player
        if i == state.current_player:
            print("    (Current Player)")


def display_action(game: Game, player_id: int, action_str: str) -> None:
    """
    Display an action in a readable format.
    
    Args:
        game: Game object
        player_id: ID of the player taking the action
        action_str: String representation of the action
    """
    player = game.state.players[player_id]
    print(f"\n{player.name} (Player {player_id}) takes action: {action_str}")


def run_test_game(
    num_players: int = 2,
    max_turns: int = 100,
    display_interval: int = 1,
    agent_iterations: int = 1000,
    random_seed: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a test game between MCTS agents.
    
    Args:
        num_players: Number of players
        max_turns: Maximum number of turns
        display_interval: How often to display the game state (in turns)
        agent_iterations: Number of MCTS iterations per move
        random_seed: Random seed for reproducibility
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary of game statistics
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Create a game
    player_names = [f"MCTS Agent {i+1}" for i in range(num_players)]
    game = Game(
        num_players=num_players,
        player_names=player_names,
        use_historical_cards=True,
        random_seed=random_seed
    )
    
    # Create MCTS agents with different configurations
    agents = []
    for i in range(num_players):
        # Vary the agents slightly to make the game more interesting
        iterations = agent_iterations + random.randint(-100, 100)
        iterations = max(100, iterations)  # Ensure at least 100 iterations
        
        config = MCTSConfig(
            iterations=iterations,
            exploration_weight=1.41,
            use_heuristics=True,
            simulation_policy="heuristic" if i % 2 == 0 else "random"
        )
        
        agent = MCTSAgent(
            config=config,
            name=f"MCTS Agent {i+1}",
            verbose=verbose
        )
        
        agents.append(agent)
        game.register_agent(i, agent.get_action_callback())
    
    # Display initial state
    if verbose:
        print("\nStarting game...")
        display_game_state(game)
    
    # Run the game
    start_time = time.time()
    turn = 0
    
    while not game.state.game_over and turn < max_turns:
        # Get current player
        player_id = game.state.current_player
        
        # Take a step (the agent will select an action)
        state, game_over = game.step()
        
        # Display the action
        if verbose or turn % display_interval == 0:
            last_action = game.state.actions_history[-1][1] if game.state.actions_history else None
            if last_action:
                display_action(game, player_id, str(last_action))
        
        # Display the game state
        if verbose or (turn % display_interval == 0 and turn > 0):
            display_game_state(game)
        
        turn += 1
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Display final state
    print("\nGame over!")
    display_game_state(game)
    
    # Display results
    print("\nResults:")
    if game.state.result == GameResult.WINNER:
        winner = game.state.players[game.state.winner]
        print(f"Winner: {winner.name} with {winner.points} points")
    else:
        print("Result: Draw")
    
    print(f"Total turns: {game.state.turn_count}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    # Display player statistics
    print("\nPlayer Statistics:")
    for player in game.state.players:
        print(f"  {player.name}:")
        print(f"    Points: {player.points}")
        print(f"    Cards: {len(player.cards)}")
        print(f"    Nobles: {len(player.nobles)}")
    
    # Get game statistics
    stats = game.get_game_statistics()
    stats["elapsed_time"] = elapsed_time
    
    return stats


def main():
    """Run the test game with command-line arguments."""
    parser = argparse.ArgumentParser(description="Test the Splendor game and MCTS agent.")
    parser.add_argument("--players", type=int, default=2, help="Number of players (2-4)")
    parser.add_argument("--turns", type=int, default=100, help="Maximum number of turns")
    parser.add_argument("--display-interval", type=int, default=5, 
                        help="How often to display the game state (in turns)")
    parser.add_argument("--iterations", type=int, default=1000, 
                        help="Number of MCTS iterations per move")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    
    # Run the test game
    stats = run_test_game(
        num_players=args.players,
        max_turns=args.turns,
        display_interval=args.display_interval,
        agent_iterations=args.iterations,
        random_seed=args.seed,
        verbose=args.verbose
    )
    
    # Print summary statistics
    print("\nGame Summary:")
    print(f"  Duration: {stats['elapsed_time']:.2f} seconds")
    print(f"  Turns: {stats['turns']}")
    # Use .get to avoid KeyError if 'result' is missing (e.g., game hit max_turns)
    print(f"  Result: {stats.get('result', 'IN_PROGRESS')}")
    if 'winner' in stats:
        print(f"  Winner: Player {stats['winner']} ({stats['winner_name']})")
        print(f"  Winner Score: {stats['winner_score']}")
    
    print("\nAction Statistics:")
    print(f"  Total Actions: {stats.get('actions_take_gems', 0) + stats.get('actions_purchase_card', 0) + stats.get('actions_reserve_card', 0)}")
    print(f"  Take Gems: {stats.get('actions_take_gems', 0)}")
    print(f"  Purchase Cards: {stats.get('actions_purchase_card', 0)}")
    print(f"  Reserve Cards: {stats.get('actions_reserve_card', 0)}")


if __name__ == "__main__":
    main()
