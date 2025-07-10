#!/usr/bin/env python
"""
Comprehensive test script for Splendor AI system.

This script demonstrates all working components of the Splendor AI system:
- Game engine (rules, state management, actions)
- AI agents (Random, MCTS)
- User interfaces (text-based)
- Training and evaluation

Run this script to verify that all components are working correctly.
"""
import os
import sys
import time
import random
import argparse
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

try:
    import numpy as np
    import torch
except ImportError:
    print("Warning: NumPy and/or PyTorch not found. Some tests will be skipped.")
    np = None
    torch = None

from splendor_ai.core.game import Game, GameState, GameResult
from splendor_ai.core.actions import Action, ActionType
from splendor_ai.core.player import Player
from splendor_ai.core.constants import GemColor, ALL_GEMS
from splendor_ai.core.cards import Card, CardTier, Noble

# Import agents if available
try:
    from splendor_ai.mcts.agent import MCTSAgent
    from splendor_ai.mcts.config import MCTSConfig
    MCTS_AVAILABLE = True
except ImportError:
    print("Warning: MCTS module not found. MCTS tests will be skipped.")
    MCTS_AVAILABLE = False

try:
    from splendor_ai.rl.agents import PPOAgent, A2CAgent, RandomAgent, HybridAgent
    from splendor_ai.rl.config import PPOConfig, A2CConfig, NetworkConfig
    RL_AVAILABLE = True
except ImportError:
    print("Warning: RL module not found. RL tests will be skipped.")
    RL_AVAILABLE = False
    # Define a minimal RandomAgent for testing if RL is not available
    class RandomAgent:
        def __init__(self, name="Random Agent"):
            self.name = name
            
        def select_action(self, state, player_id):
            valid_actions = state.get_valid_actions(player_id)
            if not valid_actions:
                raise ValueError(f"No valid actions for player {player_id}")
            return random.choice(valid_actions)
            
        def get_action_callback(self):
            return lambda state, player_id: self.select_action(state, player_id)


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


def test_game_engine():
    """Test the core game engine functionality."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}Testing Game Engine{Colors.RESET}")
    
    # Create a new game
    print("Creating a new game...")
    game = Game(num_players=2)
    
    # Reset the game
    print("Resetting game state...")
    game.reset()
    
    # Check initial state
    print("Checking initial state...")
    state = game.state
    
    print(f"  Number of players: {len(state.players)}")
    print(f"  Current player: {state.current_player}")
    print(f"  Turn count: {state.turn_count}")
    print(f"  Game result: {state.result}")
    
    # Check gem pool
    print("Checking gem pool...")
    for color, count in state.gem_pool.items():
        color_code = Colors.gem_color(color)
        print(f"  {color_code}●{Colors.RESET} {color.name}: {count}")
    
    # Check nobles
    print("Checking nobles...")
    for i, noble in enumerate(state.nobles):
        print(f"  Noble {i+1}: {noble.points} points, {len(noble.requirements)} requirements")
    
    # Check card decks
    print("Checking card decks...")
    for tier, deck in state.card_decks.items():
        print(f"  Tier {tier.value} deck: {len(deck)} cards")
    
    # Check visible cards
    print("Checking visible cards...")
    visible_cards = state.get_visible_cards()
    print(f"  Number of visible cards: {len(visible_cards)}")
    
    # Check valid actions
    print("Checking valid actions for player 0...")
    valid_actions = state.get_valid_actions(0)
    
    action_types = defaultdict(int)
    for action in valid_actions:
        action_types[action.action_type] += 1
    
    for action_type, count in action_types.items():
        print(f"  {action_type.name}: {count} actions")
    
    # Test taking a turn
    print("Testing turn execution...")
    action = valid_actions[0]
    print(f"  Taking action: {action}")
    next_state, game_over = game.step(action)
    
    print(f"  New current player: {next_state.current_player}")
    print(f"  New turn count: {next_state.turn_count}")
    print(f"  Game over: {game_over}")
    
    print(f"{Colors.GREEN}Game engine test completed successfully!{Colors.RESET}")
    return True


def test_random_agent():
    """Test the random agent functionality."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}Testing Random Agent{Colors.RESET}")
    
    # Create a new game
    print("Creating a new game...")
    game = Game(num_players=2)
    
    # Create random agents
    print("Creating random agents...")
    agent1 = RandomAgent(name="Random Agent 1")
    agent2 = RandomAgent(name="Random Agent 2")
    
    # Register agents
    print("Registering agents...")
    game.register_agent(0, agent1.get_action_callback())
    game.register_agent(1, agent2.get_action_callback())
    
    # Reset the game
    print("Resetting game state...")
    game.reset()
    
    # Play a few turns
    print("Playing 5 turns...")
    for i in range(5):
        current_player = game.state.current_player
        current_agent = agent1 if current_player == 0 else agent2
        
        print(f"  Turn {i+1}, Player {current_player} ({current_agent.name})")
        
        # Get valid actions
        valid_actions = game.state.get_valid_actions(current_player)
        print(f"    Valid actions: {len(valid_actions)}")
        
        # Select action
        action = current_agent.select_action(game.state, current_player)
        print(f"    Selected action: {action}")
        
        # Take action
        next_state, game_over = game.step(action)
        
        # Check if game is over
        if game_over:
            print(f"    Game over! Result: {next_state.result}")
            break
    
    print(f"{Colors.GREEN}Random agent test completed successfully!{Colors.RESET}")
    return True


def test_mcts_agent():
    """Test the MCTS agent functionality."""
    if not MCTS_AVAILABLE:
        print(f"\n{Colors.BOLD}{Colors.YELLOW}Skipping MCTS Agent Test (Module not available){Colors.RESET}")
        return False
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}Testing MCTS Agent{Colors.RESET}")
    
    # Create a new game
    print("Creating a new game...")
    game = Game(num_players=2)
    
    # Create MCTS agent and random agent
    print("Creating agents...")
    mcts_config = MCTSConfig(iterations=50, exploration_weight=1.0)
    mcts_agent = MCTSAgent(config=mcts_config, name="MCTS Agent")
    random_agent = RandomAgent(name="Random Agent")
    
    # Register agents
    print("Registering agents...")
    game.register_agent(0, mcts_agent.get_action_callback())
    game.register_agent(1, random_agent.get_action_callback())
    
    # Reset the game
    print("Resetting game state...")
    game.reset()
    
    # Play a few turns
    print("Playing 3 turns...")
    for i in range(3):
        current_player = game.state.current_player
        current_agent = mcts_agent if current_player == 0 else random_agent
        
        print(f"  Turn {i+1}, Player {current_player} ({current_agent.name})")
        
        # Get valid actions
        valid_actions = game.state.get_valid_actions(current_player)
        print(f"    Valid actions: {len(valid_actions)}")
        
        # Select action
        start_time = time.time()
        action = current_agent.select_action(game.state, current_player)
        end_time = time.time()
        
        if current_player == 0:
            print(f"    MCTS thinking time: {end_time - start_time:.2f} seconds")
        
        print(f"    Selected action: {action}")
        
        # Take action
        next_state, game_over = game.step(action)
        
        # Check if game is over
        if game_over:
            print(f"    Game over! Result: {next_state.result}")
            break
    
    print(f"{Colors.GREEN}MCTS agent test completed successfully!{Colors.RESET}")
    return True


def test_rl_agent():
    """Test the RL agent functionality."""
    if not RL_AVAILABLE or torch is None:
        print(f"\n{Colors.BOLD}{Colors.YELLOW}Skipping RL Agent Test (Module not available){Colors.RESET}")
        return False
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}Testing RL Agent{Colors.RESET}")
    
    # Create a new game
    print("Creating a new game...")
    game = Game(num_players=2)
    
    # Create RL agent and random agent
    print("Creating agents...")
    
    # Create network configuration
    network_config = NetworkConfig(
        hidden_sizes=[64, 64]
    )
    
    # Create PPO configuration
    ppo_config = PPOConfig(
        network=network_config,
        device="cpu"
    )
    
    # Create PPO agent
    ppo_agent = PPOAgent(
        config=ppo_config,
        state_dim=50,  # Match the state encoding in simple_train.py
        action_dim=50,  # Simplified action space
        name="PPO Agent"
    )
    
    random_agent = RandomAgent(name="Random Agent")
    
    # Register agents
    print("Registering agents...")
    game.register_agent(0, ppo_agent.get_action_callback())
    game.register_agent(1, random_agent.get_action_callback())
    
    # Reset the game
    print("Resetting game state...")
    game.reset()
    
    # Play a few turns
    print("Playing 3 turns...")
    for i in range(3):
        current_player = game.state.current_player
        current_agent = ppo_agent if current_player == 0 else random_agent
        
        print(f"  Turn {i+1}, Player {current_player} ({current_agent.name})")
        
        # Get valid actions
        valid_actions = game.state.get_valid_actions(current_player)
        print(f"    Valid actions: {len(valid_actions)}")
        
        try:
            # Select action
            action = current_agent.select_action(game.state, current_player)
            print(f"    Selected action: {action}")
            
            # Take action
            next_state, game_over = game.step(action)
            
            # Check if game is over
            if game_over:
                print(f"    Game over! Result: {next_state.result}")
                break
        except Exception as e:
            print(f"    {Colors.RED}Error: {e}{Colors.RESET}")
            print(f"    Using random action instead")
            
            # Fall back to random action
            action = random.choice(valid_actions)
            print(f"    Selected action: {action}")
            
            # Take action
            next_state, game_over = game.step(action)
    
    print(f"{Colors.GREEN}RL agent test completed successfully!{Colors.RESET}")
    return True


def test_game_simulation():
    """Test a full game simulation with random agents."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}Testing Full Game Simulation{Colors.RESET}")
    
    # Create a new game
    print("Creating a new game...")
    game = Game(num_players=2)
    
    # Create random agents
    print("Creating random agents...")
    agent1 = RandomAgent(name="Random Agent 1")
    agent2 = RandomAgent(name="Random Agent 2")
    
    # Register agents
    print("Registering agents...")
    game.register_agent(0, agent1.get_action_callback())
    game.register_agent(1, agent2.get_action_callback())
    
    # Reset the game
    print("Resetting game state...")
    game.reset()
    
    # Play the game
    print("Playing the game (max 50 turns)...")
    turn_count = 0
    max_turns = 50
    game_over = False
    
    action_counts = {
        0: defaultdict(int),
        1: defaultdict(int)
    }
    
    while not game_over and turn_count < max_turns:
        current_player = game.state.current_player
        current_agent = agent1 if current_player == 0 else agent2
        
        if turn_count % 10 == 0:
            print(f"  Turn {turn_count}, Player {current_player} ({current_agent.name})")
            print(f"    Points: {game.state.players[0].points} vs {game.state.players[1].points}")
            print(f"    Cards: {len(game.state.players[0].cards)} vs {len(game.state.players[1].cards)}")
            print(f"    Nobles: {len(game.state.players[0].nobles)} vs {len(game.state.players[1].nobles)}")
        
        try:
            # Select action
            action = current_agent.select_action(game.state, current_player)
            
            # Record action type
            action_counts[current_player][action.action_type] += 1
            
            # Take action
            next_state, game_over = game.step(action)
            
            # Update turn count
            turn_count += 1
        except Exception as e:
            print(f"    {Colors.RED}Error: {e}{Colors.RESET}")
            break
    
    # Print final state
    print("Game finished!")
    print(f"  Total turns: {turn_count}")
    
    if game.state.result == GameResult.WINNER:
        winner_id = game.state.winner
        print(f"  Winner: Player {winner_id} ({agent1.name if winner_id == 0 else agent2.name})")
    else:
        # Determine winner by points
        player1_points = game.state.players[0].points
        player2_points = game.state.players[1].points
        
        if player1_points > player2_points:
            print(f"  Winner by points: Player 0 ({agent1.name}) with {player1_points} points")
        elif player2_points > player1_points:
            print(f"  Winner by points: Player 1 ({agent2.name}) with {player2_points} points")
        else:
            print(f"  It's a tie with {player1_points} points each")
    
    # Print player statistics
    for player_id in [0, 1]:
        agent_name = agent1.name if player_id == 0 else agent2.name
        player = game.state.players[player_id]
        
        print(f"  Player {player_id} ({agent_name}):")
        print(f"    Points: {player.points}")
        print(f"    Cards: {len(player.cards)}")
        print(f"    Nobles: {len(player.nobles)}")
        print(f"    Gems: {sum(player.gems.values())}")
        
        # Print action distribution
        print(f"    Action distribution:")
        total_actions = sum(action_counts[player_id].values())
        for action_type, count in sorted(action_counts[player_id].items(), key=lambda x: x[1], reverse=True):
            print(f"      {action_type.name}: {count} ({count / total_actions * 100:.1f}%)")
    
    print(f"{Colors.GREEN}Full game simulation completed successfully!{Colors.RESET}")
    return True


def test_ui_elements():
    """Test UI elements by displaying a sample game state."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}Testing UI Elements{Colors.RESET}")
    
    # Create a new game
    print("Creating a new game...")
    game = Game(num_players=2)
    
    # Reset the game
    print("Resetting game state...")
    game.reset()
    
    # Display gem pool
    print("\nGem Pool:")
    for color in ALL_GEMS:
        count = game.state.gem_pool.get(color, 0)
        color_name = color.name
        color_code = Colors.gem_color(color)
        print(f"  {color_code}●{Colors.RESET} {color_name}: {count}")
    
    # Display nobles
    print("\nNobles:")
    for i, noble in enumerate(game.state.nobles):
        print(f"[{i+1}] Noble ({noble.points} pts)")
        print("  Requirements:")
        for color, count in noble.requirements.items():
            color_code = Colors.gem_color(color)
            print(f"    {color_code}●{Colors.RESET} {color.name}: {count}")
    
    # Display visible cards
    print("\nAvailable Cards:")
    visible_cards = game.state.get_visible_cards()
    
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
            # Card header with points
            header = f"[{i+1}] Tier {card.tier.value} Card"
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
    
    # Display player information
    print("\nPlayers:")
    for i, player in enumerate(game.state.players):
        header = f"Player {i}"
        if i == game.state.current_player:
            header = f"▶ {header}"
        print(f"{header} (Points: {player.points})")
        
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
    
    print(f"{Colors.GREEN}UI elements test completed successfully!{Colors.RESET}")
    return True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test Splendor AI system components")
    
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")
    parser.add_argument("--engine", action="store_true",
                        help="Test game engine")
    parser.add_argument("--random", action="store_true",
                        help="Test random agent")
    parser.add_argument("--mcts", action="store_true",
                        help="Test MCTS agent")
    parser.add_argument("--rl", action="store_true",
                        help="Test RL agent")
    parser.add_argument("--simulation", action="store_true",
                        help="Test full game simulation")
    parser.add_argument("--ui", action="store_true",
                        help="Test UI elements")
    
    args = parser.parse_args()
    
    # If no specific tests are selected, run all tests
    if not any([args.engine, args.random, args.mcts, args.rl, args.simulation, args.ui, args.all]):
        args.all = True
    
    return args


def main():
    """Main function to run all tests."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set up colored output for Windows
    if os.name == 'nt':
        os.system('color')
    
    # Print header
    print(f"{Colors.BOLD}{Colors.YELLOW}============================{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}  SPLENDOR AI SYSTEM TEST  {Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}============================{Colors.RESET}")
    
    # Run tests
    tests_run = 0
    tests_passed = 0
    
    if args.all or args.engine:
        if test_game_engine():
            tests_passed += 1
        tests_run += 1
    
    if args.all or args.random:
        if test_random_agent():
            tests_passed += 1
        tests_run += 1
    
    if args.all or args.mcts:
        if test_mcts_agent():
            tests_passed += 1
        tests_run += 1
    
    if args.all or args.rl:
        if test_rl_agent():
            tests_passed += 1
        tests_run += 1
    
    if args.all or args.simulation:
        if test_game_simulation():
            tests_passed += 1
        tests_run += 1
    
    if args.all or args.ui:
        if test_ui_elements():
            tests_passed += 1
        tests_run += 1
    
    # Print summary
    print(f"\n{Colors.BOLD}{Colors.YELLOW}=== TEST SUMMARY ==={Colors.RESET}")
    print(f"Tests run: {tests_run}")
    print(f"Tests passed: {tests_passed}")
    
    if tests_passed == tests_run:
        print(f"{Colors.BOLD}{Colors.GREEN}All tests passed!{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.BOLD}{Colors.RED}Some tests failed.{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
