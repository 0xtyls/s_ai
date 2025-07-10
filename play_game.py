#!/usr/bin/env python
"""
Interactive Splendor game interface for playing against AI agents.

This script provides a command-line interface for playing Splendor against
various AI agents, including random agents, MCTS agents, and trained RL agents.

Example usage:
    # Play against a random agent
    python play_game.py --opponent random

    # Play against an MCTS agent
    python play_game.py --opponent mcts --mcts-iterations 1000

    # Play against a trained RL agent
    python play_game.py --opponent rl --model models/simple_ppo.pt
"""
import os
import sys
import time
import argparse
import random
from typing import List, Dict, Tuple, Optional, Any

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
    """Parse command-line arguments for game configuration."""
    parser = argparse.ArgumentParser(description="Play Splendor against AI agents")
    
    # Opponent configuration
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["random", "mcts", "rl", "hybrid"],
                        help="Type of AI opponent")
    
    # MCTS configuration
    parser.add_argument("--mcts-iterations", type=int, default=1000,
                        help="Number of MCTS iterations per move (for MCTS opponent)")
    
    # RL configuration
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (for RL or hybrid opponent)")
    parser.add_argument("--agent-type", type=str, default="ppo",
                        choices=["ppo", "a2c"],
                        help="Type of RL agent (for RL or hybrid opponent)")
    
    # Game configuration
    parser.add_argument("--first", action="store_true",
                        help="Human player goes first")
    parser.add_argument("--debug", action="store_true",
                        help="Show debug information")
    
    return parser.parse_args()


def create_opponent(args):
    """Create an AI opponent based on command-line arguments."""
    if args.opponent == "random":
        return RandomAgent(name="Random AI")
    
    elif args.opponent == "mcts":
        mcts_config = MCTSConfig(
            iterations=args.mcts_iterations,
            exploration_weight=1.0
        )
        return MCTSAgent(config=mcts_config, name="MCTS AI")
    
    elif args.opponent == "rl":
        if args.model is None:
            print("Error: Must specify a model path for RL opponent")
            sys.exit(1)
        
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
                name="PPO AI"
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
                name="A2C AI"
            )
        
        # Load model
        try:
            agent.load(args.model)
            print(f"Loaded model from {args.model}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model instead")
        
        return agent
    
    elif args.opponent == "hybrid":
        if args.model is None:
            print("Error: Must specify a model path for hybrid opponent")
            sys.exit(1)
        
        # Create base RL agent
        base_agent = create_opponent(argparse.Namespace(
            opponent="rl",
            model=args.model,
            agent_type=args.agent_type
        ))
        
        # Create MCTS configuration
        mcts_config = MCTSConfig(
            iterations=args.mcts_iterations // 5,  # Fewer iterations for hybrid
            exploration_weight=1.0
        )
        
        # Create hybrid agent
        return HybridAgent(
            rl_agent=base_agent,
            mcts_config=mcts_config,
            name="Hybrid AI"
        )
    
    else:
        print(f"Unknown opponent type: {args.opponent}")
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


def display_player(player: Player, player_id: int, is_current: bool):
    """Display player information."""
    # Player header
    header = Colors.BOLD
    if is_current:
        header += Colors.CYAN + "▶ "
    
    if player_id == 0:
        header += "You"
    else:
        header += f"AI Opponent"
    
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


def display_game_state(state: GameState):
    """Display the current game state."""
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Display game header
    print(Colors.BOLD + Colors.YELLOW + "=== SPLENDOR GAME ===" + Colors.RESET)
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
    for i, player in enumerate(state.players):
        display_player(player, i, state.current_player == i)


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


def get_human_action(state: GameState) -> Action:
    """Get an action from the human player."""
    valid_actions = state.get_valid_actions(0)
    
    if not valid_actions:
        print("No valid actions available!")
        input("Press Enter to continue...")
        return None
    
    # Group actions by type
    take_gems_actions = []
    purchase_actions = []
    reserve_actions = []
    
    for action in valid_actions:
        if action.action_type == ActionType.TAKE_GEMS:
            take_gems_actions.append(action)
        elif action.action_type == ActionType.PURCHASE_CARD:
            purchase_actions.append(action)
        elif action.action_type == ActionType.RESERVE_CARD:
            reserve_actions.append(action)
    
    # Display action menu
    print("\n" + Colors.BOLD + "Your turn! Choose an action:" + Colors.RESET)
    print("1. Take gems")
    print("2. Purchase card")
    print("3. Reserve card")
    
    # Get action type
    while True:
        try:
            choice = int(input("\nEnter choice (1-3): "))
            if 1 <= choice <= 3:
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Handle take gems
    if choice == 1:
        if not take_gems_actions:
            print("No valid gem-taking actions available!")
            return get_human_action(state)
        
        print("\n" + Colors.BOLD + "Choose gems to take:" + Colors.RESET)
        for i, action in enumerate(take_gems_actions):
            print(f"{i + 1}. {format_action(action)}")
        
        # Get gem choice
        while True:
            try:
                gem_choice = int(input("\nEnter choice (1-" + str(len(take_gems_actions)) + "): "))
                if 1 <= gem_choice <= len(take_gems_actions):
                    return take_gems_actions[gem_choice - 1]
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(take_gems_actions)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    # Handle purchase card
    elif choice == 2:
        if not purchase_actions:
            print("No valid card purchase actions available!")
            return get_human_action(state)
        
        print("\n" + Colors.BOLD + "Choose card to purchase:" + Colors.RESET)
        for i, action in enumerate(purchase_actions):
            print(f"{i + 1}. {format_action(action)}")
        
        # Get card choice
        while True:
            try:
                card_choice = int(input("\nEnter choice (1-" + str(len(purchase_actions)) + "): "))
                if 1 <= card_choice <= len(purchase_actions):
                    return purchase_actions[card_choice - 1]
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(purchase_actions)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    # Handle reserve card
    elif choice == 3:
        if not reserve_actions:
            print("No valid card reservation actions available!")
            return get_human_action(state)
        
        print("\n" + Colors.BOLD + "Choose card to reserve:" + Colors.RESET)
        for i, action in enumerate(reserve_actions):
            print(f"{i + 1}. {format_action(action)}")
        
        # Get reservation choice
        while True:
            try:
                reserve_choice = int(input("\nEnter choice (1-" + str(len(reserve_actions)) + "): "))
                if 1 <= reserve_choice <= len(reserve_actions):
                    return reserve_actions[reserve_choice - 1]
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(reserve_actions)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    # Fallback
    print("Something went wrong. Please try again.")
    return get_human_action(state)


def play_game(args):
    """Play a game of Splendor against an AI opponent."""
    # Create game
    game = Game(num_players=2)
    
    # Create opponent
    opponent = create_opponent(args)
    print(f"Playing against: {opponent.name}")
    
    # Register agents
    human_player_id = 0 if args.first else 1
    ai_player_id = 1 if args.first else 0
    
    game.register_agent(ai_player_id, opponent.get_action_callback())
    
    # Reset game
    game.reset()
    
    # Game loop
    game_over = False
    while not game_over:
        # Display game state
        display_game_state(game.state)
        
        # Check if it's human's turn
        if game.state.current_player == human_player_id:
            # Get human action
            action = get_human_action(game.state)
            
            # Take action
            next_state, game_over = game.step(action)
            
            # Pause briefly to show the result
            time.sleep(1)
        else:
            # AI's turn
            print(f"\n{opponent.name} is thinking...")
            
            # Add a small delay for better UX
            time.sleep(0.5)
            
            # Take AI action
            next_state, game_over = game.step()
            
            # Pause briefly to show the result
            time.sleep(1)
    
    # Display final game state
    display_game_state(game.state)
    
    # Display game result
    print("\n" + Colors.BOLD + Colors.YELLOW + "=== GAME OVER ===" + Colors.RESET)
    
    if game.state.result == GameResult.WINNER:
        winner_id = game.state.winner
        if winner_id == human_player_id:
            print(Colors.BOLD + Colors.GREEN + "You win!" + Colors.RESET)
        else:
            print(Colors.BOLD + Colors.RED + f"{opponent.name} wins!" + Colors.RESET)
    else:
        # Determine winner by points
        human_points = game.state.players[human_player_id].points
        ai_points = game.state.players[ai_player_id].points
        
        if human_points > ai_points:
            print(Colors.BOLD + Colors.GREEN + f"You win with {human_points} points vs {ai_points}!" + Colors.RESET)
        elif ai_points > human_points:
            print(Colors.BOLD + Colors.RED + f"{opponent.name} wins with {ai_points} points vs {human_points}!" + Colors.RESET)
        else:
            print(Colors.BOLD + Colors.YELLOW + f"It's a tie with {human_points} points each!" + Colors.RESET)
    
    # Display game statistics
    print("\nGame Statistics:")
    print(f"  Total turns: {game.state.turn_count}")
    print(f"  Your points: {game.state.players[human_player_id].points}")
    print(f"  {opponent.name} points: {game.state.players[ai_player_id].points}")
    print(f"  Your cards: {len(game.state.players[human_player_id].cards)}")
    print(f"  {opponent.name} cards: {len(game.state.players[ai_player_id].cards)}")
    print(f"  Your nobles: {len(game.state.players[human_player_id].nobles)}")
    print(f"  {opponent.name} nobles: {len(game.state.players[ai_player_id].nobles)}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up colored output for Windows
    if os.name == 'nt':
        os.system('color')
    
    # Display welcome message
    print(Colors.BOLD + Colors.YELLOW + "Welcome to Splendor!" + Colors.RESET)
    print("Play against an AI opponent and try to win.")
    
    # Play game
    try:
        play_game(args)
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
        sys.exit(0)
    
    # Ask to play again
    while True:
        play_again = input("\nPlay again? (y/n): ").lower()
        if play_again in ['y', 'yes']:
            play_game(args)
        elif play_again in ['n', 'no']:
            print("Thanks for playing!")
            break
        else:
            print("Please enter 'y' or 'n'.")


if __name__ == "__main__":
    main()
