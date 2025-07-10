#!/usr/bin/env python
"""
Demonstration script for Splendor AI agents playing against each other.

This script shows AI vs AI gameplay for testing purposes, displaying the
game state progression and final statistics.

Example usage:
    # Run a demo game between two random agents
    python demo_game.py --agent1 random --agent2 random
    
    # Run a demo game between MCTS and random agents
    python demo_game.py --agent1 mcts --agent2 random --mcts-iterations 100
    
    # Run a demo game between a trained RL agent and MCTS
    python demo_game.py --agent1 rl --agent2 mcts --model models/simple_ppo.pt
"""
import os
import sys
import time
import argparse
import random
from typing import List, Dict, Optional, Any

import torch
import numpy as np

from splendor_ai.core.game import Game, GameState, GameResult
from splendor_ai.core.actions import Action, ActionType
from splendor_ai.core.player import Player
from splendor_ai.core.constants import GemColor, ALL_GEMS
from splendor_ai.core.cards import Card, CardTier, Noble

from splendor_ai.mcts.agent import MCTSAgent
from splendor_ai.mcts.config import MCTSConfig

from splendor_ai.rl.agents import PPOAgent, A2CAgent, RandomAgent, HybridAgent
from splendor_ai.rl.config import PPOConfig, A2CConfig, NetworkConfig


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BLACK = "\033[90m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    
    @staticmethod
    def gem_color(gem_color: GemColor) -> str:
        """Get ANSI color code for a gem color."""
        if gem_color == GemColor.RED:
            return Colors.RED
        elif gem_color == GemColor.GREEN:
            return Colors.GREEN
        elif gem_color == GemColor.BLUE:
            return Colors.BLUE
        elif gem_color == GemColor.WHITE:
            return Colors.WHITE
        elif gem_color == GemColor.BLACK:
            return Colors.BLACK
        elif gem_color == GemColor.GOLD:
            return Colors.YELLOW
        else:
            return Colors.RESET


def parse_args():
    """Parse command-line arguments for demo configuration."""
    parser = argparse.ArgumentParser(description="Demonstrate Splendor AI agents playing against each other")
    
    # Agent configuration
    parser.add_argument("--agent1", type=str, default="random",
                        choices=["random", "mcts", "rl", "hybrid"],
                        help="Type of first AI agent")
    parser.add_argument("--agent2", type=str, default="random",
                        choices=["random", "mcts", "rl", "hybrid"],
                        help="Type of second AI agent")
    
    # MCTS configuration
    parser.add_argument("--mcts-iterations", type=int, default=100,
                        help="Number of MCTS iterations per move (for MCTS agent)")
    
    # RL configuration
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (for RL or hybrid agent)")
    parser.add_argument("--agent-type", type=str, default="ppo",
                        choices=["ppo", "a2c"],
                        help="Type of RL agent (for RL or hybrid agent)")
    
    # Demo configuration
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between turns (seconds)")
    parser.add_argument("--max-turns", type=int, default=100,
                        help="Maximum number of turns before ending the game")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed information for each action")
    
    return parser.parse_args()


def create_agent(agent_type, args, name=None):
    """Create an AI agent based on type."""
    if agent_type == "random":
        return RandomAgent(name=name or "Random AI")
    
    elif agent_type == "mcts":
        mcts_config = MCTSConfig(
            iterations=args.mcts_iterations,
            exploration_weight=1.0
        )
        return MCTSAgent(config=mcts_config, name=name or "MCTS AI")
    
    elif agent_type == "rl":
        # Create network configuration
        network_config = NetworkConfig(
            hidden_sizes=[128, 128]
        )
        
        # Create agent based on type
        if args.agent_type == "ppo":
            # Create PPO configuration
            ppo_config = PPOConfig(
                network=network_config,
                device="cpu"
            )
            
            # Create PPO agent
            agent = PPOAgent(
                config=ppo_config,
                state_dim=50,  # Match the state encoding in simple_train.py
                action_dim=50,  # Simplified action space
                name=name or "PPO AI"
            )
        else:  # a2c
            # Create A2C configuration
            a2c_config = A2CConfig(
                network=network_config,
                device="cpu"
            )
            
            # Create A2C agent
            agent = A2CAgent(
                config=a2c_config,
                state_dim=50,
                action_dim=50,
                name=name or "A2C AI"
            )
        
        # Load model if specified
        if args.model:
            try:
                agent.load(args.model)
                print(f"Loaded model from {args.model}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using untrained model instead")
        
        return agent
    
    elif agent_type == "hybrid":
        # Create base RL agent
        base_agent = create_agent("rl", args, name=name or "Hybrid AI")
        
        # Create MCTS configuration
        mcts_config = MCTSConfig(
            iterations=args.mcts_iterations // 5,  # Fewer iterations for hybrid
            exploration_weight=1.0
        )
        
        # Create hybrid agent
        return HybridAgent(
            rl_agent=base_agent,
            mcts_config=mcts_config,
            name=name or "Hybrid AI"
        )
    
    else:
        print(f"Unknown agent type: {agent_type}")
        sys.exit(1)


def display_gem_pool(state: GameState):
    """Display the gem pool."""
    print("\n" + Colors.BOLD + "Gem Pool:" + Colors.RESET)
    for color in ALL_GEMS:
        count = state.gem_pool.get(color, 0)
        color_name = color.name
        color_code = Colors.gem_color(color)
        print(f"  {color_code}●{Colors.RESET} {color_name}: {count}")


def display_card(card: Card, index: Optional[int] = None):
    """Display a single card."""
    # Card header with points
    if index is not None:
        header = f"[{index}] "
    else:
        header = ""
    
    header += f"Tier {card.tier.value} Card"
    if card.points > 0:
        header += f" ({card.points} pts)"
    
    print(header)
    
    # Card bonus
    bonus_color = Colors.gem_color(card.bonus)
    print(f"  Bonus: {bonus_color}●{Colors.RESET} {card.bonus.name}")
    
    # Card cost
    print("  Cost:")
    for color, count in card.cost.items():
        if count > 0:
            color_code = Colors.gem_color(color)
            print(f"    {color_code}●{Colors.RESET} {color.name}: {count}")


def display_noble(noble: Noble, index: Optional[int] = None):
    """Display a single noble."""
    # Noble header with points
    if index is not None:
        header = f"[{index}] "
    else:
        header = ""
    
    header += f"Noble ({noble.points} pts)"
    print(header)
    
    # Noble requirements
    print("  Requirements:")
    for color, count in noble.requirements.items():
        color_code = Colors.gem_color(color)
        print(f"    {color_code}●{Colors.RESET} {color.name}: {count}")


def display_player(player: Player, player_id: int, is_current: bool, name: str):
    """Display player information."""
    # Player header
    header = Colors.BOLD
    if is_current:
        header += Colors.CYAN + "▶ "
    
    header += name
    header += Colors.RESET
    print(f"\n{header} (Points: {player.points})")
    
    # Player gems
    print("  Gems:")
    for color in ALL_GEMS:
        count = player.gems.get(color, 0)
        if count > 0:
            color_code = Colors.gem_color(color)
            print(f"    {color_code}●{Colors.RESET} {color.name}: {count}")
    
    # Player bonuses
    print("  Bonuses:")
    for color in [c for c in ALL_GEMS if c != GemColor.GOLD]:
        count = player.bonuses.get(color, 0)
        if count > 0:
            color_code = Colors.gem_color(color)
            print(f"    {color_code}●{Colors.RESET} {color.name}: {count}")
    
    # Player cards
    if player.cards:
        print("  Cards:")
        for i, card in enumerate(player.cards):
            bonus_color = Colors.gem_color(card.bonus)
            print(f"    {bonus_color}●{Colors.RESET} {card.bonus.name} (Tier {card.tier.value}, {card.points} pts)")
    
    # Player reserved cards
    if player.reserved_cards:
        print("  Reserved Cards:")
        for i, card in enumerate(player.reserved_cards):
            bonus_color = Colors.gem_color(card.bonus)
            print(f"    {bonus_color}●{Colors.RESET} {card.bonus.name} (Tier {card.tier.value}, {card.points} pts)")
    
    # Player nobles
    if player.nobles:
        print("  Nobles:")
        for i, noble in enumerate(player.nobles):
            print(f"    Noble ({noble.points} pts)")


def display_game_state(state: GameState, agent1_name: str, agent2_name: str):
    """Display the current game state."""
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Display game header
    print(Colors.BOLD + Colors.YELLOW + "=== SPLENDOR AI DEMO ===" + Colors.RESET)
    print(f"Turn: {state.turn_count}")
    
    # Display gem pool
    display_gem_pool(state)
    
    # Display nobles
    print("\n" + Colors.BOLD + "Nobles:" + Colors.RESET)
    for i, noble in enumerate(state.nobles):
        display_noble(noble, i + 1)
    
    # Display visible cards
    visible_cards = state.get_visible_cards()
    print("\n" + Colors.BOLD + "Available Cards:" + Colors.RESET)
    
    # Group cards by tier
    cards_by_tier = {}
    for card in visible_cards:
        if card.tier not in cards_by_tier:
            cards_by_tier[card.tier] = []
        cards_by_tier[card.tier].append(card)
    
    # Display cards by tier
    for tier in sorted(cards_by_tier.keys(), key=lambda t: t.value):
        print(f"\nTier {tier.value}:")
        for i, card in enumerate(cards_by_tier[tier]):
            # Calculate global index
            global_index = visible_cards.index(card) + 1
            display_card(card, global_index)
    
    # Display players
    player_names = [agent1_name, agent2_name]
    for i, player in enumerate(state.players):
        display_player(player, i, state.current_player == i, player_names[i])


def format_action(action: Action) -> str:
    """Format an action as a human-readable string."""
    if action.action_type == ActionType.TAKE_GEMS:
        gems_str = ", ".join(f"{count} {color.name}" for color, count in action.gems.items())
        return f"Take gems: {gems_str}"
    
    elif action.action_type == ActionType.PURCHASE_CARD:
        card = action.card
        return f"Purchase card: {card.bonus.name} (Tier {card.tier.value}, {card.points} pts)"
    
    elif action.action_type == ActionType.RESERVE_CARD:
        if action.card:
            card = action.card
            return f"Reserve card: {card.bonus.name} (Tier {card.tier.value}, {card.points} pts)"
        else:
            return f"Reserve card from Tier {action.tier.value} deck"
    
    else:
        return str(action)


def run_demo(args):
    """Run a demonstration game between two AI agents."""
    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Create agents
    agent1 = create_agent(args.agent1, args, name=f"{args.agent1.upper()} Agent 1")
    agent2 = create_agent(args.agent2, args, name=f"{args.agent2.upper()} Agent 2")
    
    print(f"Running demo game: {agent1.name} vs {agent2.name}")
    
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
    while not game_over and turn_count < args.max_turns:
        # Display game state
        display_game_state(game.state, agent1.name, agent2.name)
        
        # Get current player
        current_player_id = game.state.current_player
        current_agent = agent1 if current_player_id == 0 else agent2
        
        # Show current player's turn
        print(f"\n{current_agent.name} is thinking...")
        
        # Add a small delay for better UX
        time.sleep(args.delay)
        
        # Take action
        action = current_agent.select_action(game.state, current_player_id)
        next_state, game_over = game.step(action)
        
        # Record action
        actions_taken.append((current_player_id, action))
        
        # Display action if verbose
        if args.verbose:
            print(f"{current_agent.name} chose: {format_action(action)}")
            input("Press Enter to continue...")
        
        # Update turn count
        turn_count += 1
    
    # Display final game state
    display_game_state(game.state, agent1.name, agent2.name)
    
    # Display game result
    print("\n" + Colors.BOLD + Colors.YELLOW + "=== GAME OVER ===" + Colors.RESET)
    
    if game.state.result == GameResult.WINNER:
        winner_id = game.state.winner
        if winner_id == 0:
            print(Colors.BOLD + Colors.GREEN + f"{agent1.name} wins!" + Colors.RESET)
        else:
            print(Colors.BOLD + Colors.RED + f"{agent2.name} wins!" + Colors.RESET)
    else:
        # Determine winner by points
        player1_points = game.state.players[0].points
        player2_points = game.state.players[1].points
        
        if player1_points > player2_points:
            print(Colors.BOLD + Colors.GREEN + f"{agent1.name} wins with {player1_points} points vs {player2_points}!" + Colors.RESET)
        elif player2_points > player1_points:
            print(Colors.BOLD + Colors.RED + f"{agent2.name} wins with {player2_points} points vs {player1_points}!" + Colors.RESET)
        else:
            print(Colors.BOLD + Colors.YELLOW + f"It's a tie with {player1_points} points each!" + Colors.RESET)
    
    # Display game statistics
    print("\nGame Statistics:")
    print(f"  Total turns: {turn_count}")
    print(f"  {agent1.name} points: {game.state.players[0].points}")
    print(f"  {agent2.name} points: {game.state.players[1].points}")
    print(f"  {agent1.name} cards: {len(game.state.players[0].cards)}")
    print(f"  {agent2.name} cards: {len(game.state.players[1].cards)}")
    print(f"  {agent1.name} nobles: {len(game.state.players[0].nobles)}")
    print(f"  {agent2.name} nobles: {len(game.state.players[1].nobles)}")
    
    # Display action breakdown if verbose
    if args.verbose:
        print("\nAction Breakdown:")
        
        # Count action types
        agent1_actions = {
            ActionType.TAKE_GEMS: 0,
            ActionType.PURCHASE_CARD: 0,
            ActionType.RESERVE_CARD: 0
        }
        
        agent2_actions = {
            ActionType.TAKE_GEMS: 0,
            ActionType.PURCHASE_CARD: 0,
            ActionType.RESERVE_CARD: 0
        }
        
        for player_id, action in actions_taken:
            if player_id == 0:
                agent1_actions[action.action_type] += 1
            else:
                agent2_actions[action.action_type] += 1
        
        # Display counts
        print(f"  {agent1.name} actions:")
        print(f"    Take gems: {agent1_actions[ActionType.TAKE_GEMS]}")
        print(f"    Purchase cards: {agent1_actions[ActionType.PURCHASE_CARD]}")
        print(f"    Reserve cards: {agent1_actions[ActionType.RESERVE_CARD]}")
        
        print(f"  {agent2.name} actions:")
        print(f"    Take gems: {agent2_actions[ActionType.TAKE_GEMS]}")
        print(f"    Purchase cards: {agent2_actions[ActionType.PURCHASE_CARD]}")
        print(f"    Reserve cards: {agent2_actions[ActionType.RESERVE_CARD]}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up colored output for Windows
    if os.name == 'nt':
        os.system('color')
    
    # Display welcome message
    print(Colors.BOLD + Colors.YELLOW + "Welcome to Splendor AI Demo!" + Colors.RESET)
    print("Watch two AI agents play against each other.")
    
    # Run demo
    try:
        run_demo(args)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        sys.exit(0)
    
    # Ask to run again
    while True:
        run_again = input("\nRun another demo? (y/n): ").lower()
        if run_again in ['y', 'yes']:
            run_demo(args)
        elif run_again in ['n', 'no']:
            print("Thanks for watching!")
            break
        else:
            print("Please enter 'y' or 'n'.")


if __name__ == "__main__":
    main()
