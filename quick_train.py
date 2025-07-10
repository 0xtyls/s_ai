#!/usr/bin/env python
"""
Quick training demonstration for Splendor AI.

This script provides a simplified demonstration of the Splendor game engine
and basic agent training concepts. It uses random agents to play against each
other and collects statistics to show the system is operational.

Example usage:
    # Run 100 games between random agents
    python quick_train.py --games 100
    
    # Run 50 games with visualization
    python quick_train.py --games 50 --visualize
"""
import os
import sys
import time
import argparse
import random
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from splendor_ai.core.game import Game, GameState, GameResult
from splendor_ai.core.actions import Action, ActionType
from splendor_ai.core.player import Player
from splendor_ai.core.constants import GemColor, ALL_GEMS
from splendor_ai.core.cards import Card, CardTier, Noble

from splendor_ai.rl.agents import RandomAgent


def parse_args():
    """Parse command-line arguments for demo configuration."""
    parser = argparse.ArgumentParser(description="Quick training demonstration for Splendor AI")
    
    # Training parameters
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games to play")
    parser.add_argument("--max-turns", type=int, default=100,
                        help="Maximum number of turns per game")
    
    # Display options
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize training statistics")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed game information")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between turns (seconds) for visualization")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def play_game(agent1, agent2, max_turns=100, verbose=False, delay=0.0):
    """Play a single game between two agents and return statistics."""
    # Create game
    game = Game(num_players=2)
    
    # Register agents
    game.register_agent(0, agent1.get_action_callback())
    game.register_agent(1, agent2.get_action_callback())
    
    # Reset game
    game.reset()
    
    # Game statistics
    turn_count = 0
    actions_taken = []
    
    # Game loop
    game_over = False
    while not game_over and turn_count < max_turns:
        # Get current player
        current_player_id = game.state.current_player
        current_agent = agent1 if current_player_id == 0 else agent2
        
        # Display game state if verbose
        if verbose:
            print(f"\nTurn {turn_count}, Player {current_player_id} ({current_agent.name})")
            print(f"Gems: {game.state.players[current_player_id].gems}")
            print(f"Points: {game.state.players[current_player_id].points}")
        
        # Add a small delay for visualization
        if delay > 0:
            time.sleep(delay)
        
        # Take action
        action = current_agent.select_action(game.state, current_player_id)
        next_state, game_over = game.step(action)
        
        # Record action
        actions_taken.append((current_player_id, action))
        
        # Display action if verbose
        if verbose:
            print(f"Action: {action}")
        
        # Update turn count
        turn_count += 1
    
    # Game result
    if game.state.result == GameResult.WINNER:
        winner_id = game.state.winner
        if verbose:
            print(f"\nGame over! Player {winner_id} wins!")
    else:
        # Determine winner by points
        player1_points = game.state.players[0].points
        player2_points = game.state.players[1].points
        
        if player1_points > player2_points:
            winner_id = 0
        elif player2_points > player1_points:
            winner_id = 1
        else:
            winner_id = -1  # Draw
        
        if verbose:
            if winner_id == -1:
                print(f"\nGame over! It's a draw with {player1_points} points each.")
            else:
                print(f"\nGame over! Player {winner_id} wins with {game.state.players[winner_id].points} points!")
    
    # Collect action statistics
    action_stats = {
        0: defaultdict(int),
        1: defaultdict(int)
    }
    
    for player_id, action in actions_taken:
        action_stats[player_id][action.action_type] += 1
    
    # Return game statistics
    return {
        "winner": winner_id,
        "turns": turn_count,
        "points": [game.state.players[0].points, game.state.players[1].points],
        "cards": [len(game.state.players[0].cards), len(game.state.players[1].cards)],
        "nobles": [len(game.state.players[0].nobles), len(game.state.players[1].nobles)],
        "gems_remaining": sum(game.state.gem_pool.values()),
        "action_stats": action_stats
    }


def run_training(args):
    """Run training demonstration with specified parameters."""
    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Create agents
    agent1 = RandomAgent(name="Random Agent 1")
    agent2 = RandomAgent(name="Random Agent 2")
    
    print(f"Running {args.games} games between {agent1.name} and {agent2.name}...")
    
    # Training statistics
    all_stats = []
    win_counts = {0: 0, 1: 0, -1: 0}  # Player 0, Player 1, Draw
    
    # Progress tracking
    start_time = time.time()
    
    # Run games
    for game_idx in range(args.games):
        # Display progress
        if not args.verbose and game_idx % 10 == 0:
            print(f"Playing game {game_idx}/{args.games}...")
        
        # Play game
        stats = play_game(
            agent1=agent1,
            agent2=agent2,
            max_turns=args.max_turns,
            verbose=args.verbose,
            delay=args.delay
        )
        
        # Update statistics
        all_stats.append(stats)
        win_counts[stats["winner"]] += 1
        
        # Display game result if verbose
        if args.verbose:
            print(f"Game {game_idx+1} result: Player {stats['winner']} wins in {stats['turns']} turns")
            print(f"Points: {stats['points']}, Cards: {stats['cards']}, Nobles: {stats['nobles']}")
            print("-" * 50)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Calculate statistics
    avg_turns = np.mean([s["turns"] for s in all_stats])
    avg_points = [
        np.mean([s["points"][0] for s in all_stats]),
        np.mean([s["points"][1] for s in all_stats])
    ]
    avg_cards = [
        np.mean([s["cards"][0] for s in all_stats]),
        np.mean([s["cards"][1] for s in all_stats])
    ]
    avg_nobles = [
        np.mean([s["nobles"][0] for s in all_stats]),
        np.mean([s["nobles"][1] for s in all_stats])
    ]
    
    # Calculate action statistics
    action_counts = {
        0: defaultdict(int),
        1: defaultdict(int)
    }
    
    for stats in all_stats:
        for player_id in [0, 1]:
            for action_type, count in stats["action_stats"][player_id].items():
                action_counts[player_id][action_type] += count
    
    # Display statistics
    print("\nTraining Statistics:")
    print(f"Total games: {args.games}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Games per second: {args.games / elapsed_time:.2f}")
    print(f"Win rate: Player 0: {win_counts[0] / args.games:.2f}, "
          f"Player 1: {win_counts[1] / args.games:.2f}, "
          f"Draws: {win_counts[-1] / args.games:.2f}")
    print(f"Average turns per game: {avg_turns:.2f}")
    print(f"Average points: Player 0: {avg_points[0]:.2f}, Player 1: {avg_points[1]:.2f}")
    print(f"Average cards: Player 0: {avg_cards[0]:.2f}, Player 1: {avg_cards[1]:.2f}")
    print(f"Average nobles: Player 0: {avg_nobles[0]:.2f}, Player 1: {avg_nobles[1]:.2f}")
    
    print("\nAction Statistics:")
    for player_id in [0, 1]:
        print(f"Player {player_id}:")
        total_actions = sum(action_counts[player_id].values())
        for action_type, count in sorted(action_counts[player_id].items(), key=lambda x: x[1], reverse=True):
            print(f"  {action_type.name}: {count} ({count / total_actions:.2f})")
    
    # Visualize statistics if requested
    if args.visualize:
        visualize_statistics(all_stats)
    
    return all_stats


def visualize_statistics(all_stats):
    """Visualize training statistics."""
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    turns = [s["turns"] for s in all_stats]
    points_p0 = [s["points"][0] for s in all_stats]
    points_p1 = [s["points"][1] for s in all_stats]
    cards_p0 = [s["cards"][0] for s in all_stats]
    cards_p1 = [s["cards"][1] for s in all_stats]
    nobles_p0 = [s["nobles"][0] for s in all_stats]
    nobles_p1 = [s["nobles"][1] for s in all_stats]
    winners = [s["winner"] for s in all_stats]
    
    # Calculate moving averages
    window_size = min(10, len(all_stats))
    if window_size > 0:
        turns_ma = np.convolve(turns, np.ones(window_size)/window_size, mode='valid')
        points_p0_ma = np.convolve(points_p0, np.ones(window_size)/window_size, mode='valid')
        points_p1_ma = np.convolve(points_p1, np.ones(window_size)/window_size, mode='valid')
        win_rate_p0 = np.convolve([1 if w == 0 else 0 for w in winners], 
                                 np.ones(window_size)/window_size, mode='valid')
        win_rate_p1 = np.convolve([1 if w == 1 else 0 for w in winners], 
                                 np.ones(window_size)/window_size, mode='valid')
    
        # Plot turns per game
        axs[0, 0].plot(range(len(turns)), turns, 'b-', alpha=0.3)
        axs[0, 0].plot(range(window_size-1, len(turns)), turns_ma, 'b-', linewidth=2)
        axs[0, 0].set_title('Turns per Game')
        axs[0, 0].set_xlabel('Game')
        axs[0, 0].set_ylabel('Turns')
        axs[0, 0].grid(True)
        
        # Plot points per game
        axs[0, 1].plot(range(len(points_p0)), points_p0, 'b-', alpha=0.3, label='Player 0')
        axs[0, 1].plot(range(window_size-1, len(points_p0)), points_p0_ma, 'b-', linewidth=2)
        axs[0, 1].plot(range(len(points_p1)), points_p1, 'r-', alpha=0.3, label='Player 1')
        axs[0, 1].plot(range(window_size-1, len(points_p1)), points_p1_ma, 'r-', linewidth=2)
        axs[0, 1].set_title('Points per Game')
        axs[0, 1].set_xlabel('Game')
        axs[0, 1].set_ylabel('Points')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # Plot win rates
        axs[1, 0].plot(range(window_size-1, len(winners)), win_rate_p0, 'b-', linewidth=2, label='Player 0')
        axs[1, 0].plot(range(window_size-1, len(winners)), win_rate_p1, 'r-', linewidth=2, label='Player 1')
        axs[1, 0].set_title('Win Rate (Moving Average)')
        axs[1, 0].set_xlabel('Game')
        axs[1, 0].set_ylabel('Win Rate')
        axs[1, 0].set_ylim(0, 1)
        axs[1, 0].legend()
        axs[1, 0].grid(True)
    
    # Plot action distribution
    action_counts = {
        0: defaultdict(int),
        1: defaultdict(int)
    }
    
    for stats in all_stats:
        for player_id in [0, 1]:
            for action_type, count in stats["action_stats"][player_id].items():
                action_counts[player_id][action_type] += count
    
    # Prepare data for bar chart
    action_types = list(ActionType)
    p0_counts = [action_counts[0][at] for at in action_types]
    p1_counts = [action_counts[1][at] for at in action_types]
    
    # Normalize
    p0_total = sum(p0_counts)
    p1_total = sum(p1_counts)
    
    if p0_total > 0 and p1_total > 0:
        p0_norm = [c / p0_total for c in p0_counts]
        p1_norm = [c / p1_total for c in p1_counts]
        
        # Plot action distribution
        x = np.arange(len(action_types))
        width = 0.35
        
        axs[1, 1].bar(x - width/2, p0_norm, width, label='Player 0')
        axs[1, 1].bar(x + width/2, p1_norm, width, label='Player 1')
        axs[1, 1].set_title('Action Distribution')
        axs[1, 1].set_xlabel('Action Type')
        axs[1, 1].set_ylabel('Frequency')
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels([at.name for at in action_types], rotation=45)
        axs[1, 1].legend()
        axs[1, 1].grid(True)
    
    # Adjust layout and show
    plt.tight_layout()
    plt.show()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Run training demonstration
    try:
        stats = run_training(args)
        print("\nTraining demonstration completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining demonstration interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
