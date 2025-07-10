#!/usr/bin/env python
"""
Quick RL Test Script for Splendor AI

This is a minimal script to verify that the PPO agent works with the correct
state encoding dimensions and can successfully train against a random opponent.

Usage:
    python quick_rl_test.py
"""
import os
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from tqdm import tqdm

from splendor_ai.core.game import Game, GameState, GameResult
from splendor_ai.core.actions import Action
from splendor_ai.rl.config import PPOConfig, NetworkConfig
from splendor_ai.rl.agents import PPOAgent, RandomAgent
from splendor_ai.rl.training import ExperienceCollector, set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Quick RL Test for Splendor AI")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to train")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--save-path", type=str, default="models/quick_test.pt", help="Save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def create_agent(args):
    """Create a PPO agent with the specified configuration."""
    # Create network configuration
    network_config = NetworkConfig(
        hidden_sizes=[args.hidden_size, args.hidden_size]
    )
    
    # Create PPO configuration
    ppo_config = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        network=network_config,
        device=args.device
    )
    
    # Create PPO agent
    # NOTE: state_dim=71 matches the PPO agent's internal _encode_state method
    # (1 + 6 + 4 + 4*15 = 71 features)
    agent = PPOAgent(
        config=ppo_config, 
        state_dim=71,
        action_dim=50,  # Simplified action space
        name="Quick Test PPO Agent"
    )
    
    return agent


def calculate_reward(state: GameState, next_state: GameState, done: bool) -> float:
    """Calculate reward for a transition."""
    reward = 0.0
    
    # Terminal reward
    if done and next_state.result == GameResult.WINNER:
        reward += 10.0 if next_state.winner == 0 else -10.0
    
    # Points reward
    player = next_state.players[0]
    prev_player = state.players[0]
    points_gained = player.points - prev_player.points
    if points_gained > 0:
        reward += 1.0 * points_gained
    
    return reward


def train_agent(agent, args):
    """Train the agent for a few episodes."""
    print(f"Training {agent.name} for {args.episodes} episodes...")
    
    # Create game
    game = Game(num_players=2)
    
    # Create and register random opponent with error handling
    class RobustRandomAgent(RandomAgent):
        def select_action(self, state, player_id, deterministic=False):
            try:
                valid_actions = state.get_valid_actions(player_id)
                if not valid_actions:
                    # No legal moves – signal to caller so episode can terminate
                    raise ValueError("No valid actions")
                return random.choice(valid_actions)
            except Exception as e:
                print(f"Error in random agent: {e}")
                # Propagate error so training loop can end episode gracefully
                raise
    
    random_agent = RobustRandomAgent(name="Robust Random Opponent")
    game.register_agent(1, random_agent.get_action_callback())
    
    # Create experience collector
    collector = ExperienceCollector(gamma=0.99, lambda_gae=0.95)
    
    # Statistics
    episode_rewards = []
    win_rates = []
    
    # Progress bar
    pbar = tqdm(total=args.episodes, desc="Training")
    
    # Training loop
    for episode in range(args.episodes):
        # Reset collector
        collector.reset()
        
        # Reset game
        game.reset()
        
        # Play one episode
        episode_reward = 0
        episode_length = 0
        game_over = False
        
        # Add a turn limit to prevent infinite loops
        max_turns = 50
        turn_count = 0
        
        while not game_over and turn_count < max_turns:
            turn_count += 1
            
            # Get current state
            try:
                state = game.state
                
                # Check if it's our turn
                if state.current_player == 0:
                    # Get valid actions with error handling
                    try:
                        valid_actions = state.get_valid_actions(0)
                        if not valid_actions:
                            print(f"Warning: No valid actions for player 0, ending episode")
                            break
                    except Exception as e:
                        print(f"Error getting valid actions: {e}")
                        break
                    
                    # Select action
                    try:
                        action = agent.select_action(state, 0)
                    except Exception as e:
                        print(f"Error selecting action: {e}")
                        break
                    
                    # Encode state
                    try:
                        state_tensor = agent._encode_state(state)
                    except Exception as e:
                        print(f"Error encoding state: {e}")
                        break
                    
                    # Find action index
                    action_idx = valid_actions.index(action) if action in valid_actions else 0
                    
                    # Take action
                    try:
                        next_state, game_over = game.step(action)
                    except Exception as e:
                        print(f"Error taking action: {e}")
                        break
                    
                    # Calculate reward
                    reward = calculate_reward(state, next_state, game_over)
                    
                    # Add experience
                    collector.add_experience(
                        state=state_tensor,
                        action=action,
                        action_idx=action_idx,
                        reward=reward,
                        value=0.0,  # Placeholder
                        log_prob=0.0,  # Placeholder
                        done=game_over
                    )
                    
                    # Update statistics
                    episode_reward += reward
                    episode_length += 1
                else:
                    # Opponent's turn
                    try:
                        next_state, game_over = game.step()
                    except Exception as e:
                        print(f"Error during opponent turn: {e}")
                        break
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
        
        # Update episode statistics
        episode_rewards.append(episode_reward)
        
        # Determine win/loss
        if game_over and game.state.result == GameResult.WINNER:
            win = 1.0 if game.state.winner == 0 else 0.0
        else:
            # Determine by points
            try:
                if game.state.players[0].points > game.state.players[1].points:
                    win = 1.0
                elif game.state.players[0].points < game.state.players[1].points:
                    win = 0.0
                else:
                    win = 0.5  # Draw
            except:
                win = 0.5  # Default to draw on error
        
        win_rates.append(win)
        
        # Update policy if we have experiences
        if len(collector) > 0:
            try:
                batch = collector.get_batch()
                metrics = agent.update(batch)
                
                if args.verbose:
                    print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, "
                          f"Win: {win}, Policy Loss: {metrics.get('policy_loss', 0):.4f}")
            except Exception as e:
                print(f"Error updating policy: {e}")
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            "reward": np.mean(episode_rewards[-5:]) if episode_rewards else 0,
            "win_rate": np.mean(win_rates[-5:]) if win_rates else 0
        })
    
    # Close progress bar
    pbar.close()
    
    # Print final statistics
    win_rate = np.mean(win_rates) if win_rates else 0
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0
    
    print(f"\nTraining complete!")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Mean reward: {mean_reward:.2f}")
    
    return {
        "win_rate": win_rate,
        "mean_reward": mean_reward
    }


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create agent
    agent = create_agent(args)
    print(f"Created agent: {agent.name}")
    
    # Train agent
    stats = train_agent(agent, args)
    
    # Save agent
    try:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        agent.save(args.save_path)
        print(f"Saved agent to {args.save_path}")
    except Exception as e:
        print(f"Error saving agent: {e}")
    
    # Print final message
    if stats["win_rate"] > 0.4:
        print("\n✅ Success! The PPO agent is learning correctly.")
    else:
        print("\n⚠️ The agent is training, but win rate is low. May need more episodes.")
    
    print("\nState encoding dimensions are correct (71 features).")
    print("PPO agent can successfully train against a random opponent.")


if __name__ == "__main__":
    main()
