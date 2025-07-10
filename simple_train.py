#!/usr/bin/env python
"""
Simple training script for Splendor AI reinforcement learning agents.

This script provides a simplified approach to training RL agents for Splendor,
adapting to our specific Game implementation rather than requiring a standard
RL environment interface.

Example usage:
    # Train a PPO agent with default settings
    python simple_train.py --episodes 100 --save-path models/simple_ppo.pt
"""
import os
import time
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
from tqdm import tqdm

from splendor_ai.core.game import Game, GameState, GameResult
from splendor_ai.core.actions import Action
from splendor_ai.core.player import Player
# GemColor and CardTier were only required for the removed custom encoder

from splendor_ai.rl.config import PPOConfig, NetworkConfig
from splendor_ai.rl.agents import PPOAgent, RandomAgent
from splendor_ai.rl.training import ExperienceCollector, set_seed

# ---------------------------------------------------------------------------
# Local helper for robust opponent handling
# ---------------------------------------------------------------------------

class NoValidActionError(Exception):
    """Raised when a player has no valid actions (edge-case game state)."""


class RobustRandomAgent(RandomAgent):
    """
    Random agent that *gracefully* signals when it has no legal moves instead
    of crashing the entire training loop.
    """

    def select_action(self, state: GameState, player_id: int, deterministic: bool = False):
        valid_actions = state.get_valid_actions(player_id)
        if not valid_actions:
            # Surface this situation to the caller so the episode can end
            raise NoValidActionError(f"No valid actions for player {player_id}")
        return random.choice(valid_actions)

def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Simple training script for Splendor AI")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to train for")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for policy updates")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="Hidden layer size for neural networks")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    
    # Evaluation and saving
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Number of episodes for evaluation")
    parser.add_argument("--save-path", type=str, default="models/simple_agent.pt",
                        help="Path to save the trained model")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    
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
    
    # Create PPO agent with a simple network for testing
    agent = PPOAgent(
        config=ppo_config, 
        # State dimension must match PPOAgent._encode_state (1 + 6 + 4 + 4*15 = 71)
        # to avoid mat1/mat2 shape mismatch during forward pass.
        state_dim=71,
        action_dim=50,  # Simplified action space
        name="Simple PPO Agent"
    )
    
    return agent


def calculate_reward(state: GameState, next_state: GameState, done: bool) -> float:
    """Calculate reward for a transition."""
    reward = 0.0
    
    # Terminal reward
    if done:
        if next_state.result == GameResult.WINNER:
            if next_state.winner == 0:
                # Win
                reward += 1.0
            else:
                # Loss
                reward -= 1.0
        else:
            # Draw
            reward += 0.0
    
    # Reward shaping
    # Points reward
    player = next_state.players[0]
    prev_player = state.players[0]
    points_gained = player.points - prev_player.points
    
    if points_gained > 0:
        reward += 0.1 * points_gained
    
    # Card purchase reward
    cards_gained = len(player.cards) - len(prev_player.cards)
    if cards_gained > 0:
        reward += 0.01 * cards_gained
    
    # Noble acquisition reward
    nobles_gained = len(player.nobles) - len(prev_player.nobles)
    if nobles_gained > 0:
        reward += 0.05 * nobles_gained
    
    return reward


def train_agent(agent, args):
    """Train the agent for the specified number of episodes."""
    print(f"Training {agent.name} for {args.episodes} episodes...")
    
    # Create game
    game = Game(num_players=2)
    
    # Register random opponent
    random_agent = RobustRandomAgent(name="Robust Random Opponent")
    game.register_agent(1, random_agent.get_action_callback())
    
    # Create experience collector
    collector = ExperienceCollector(gamma=0.99, lambda_gae=0.95)
    
    # Training statistics
    total_episodes = 0
    total_steps = 0
    episode_rewards = []
    win_rates = []
    
    # Progress bar
    pbar = tqdm(total=args.episodes, desc="Training")
    
    # Training loop
    while total_episodes < args.episodes:
        # Reset collector
        collector.reset()
        
        # Reset game
        game.reset()
        
        # Play one episode
        episode_reward = 0
        episode_length = 0
        game_over = False
        
        # Store previous state for reward calculation
        prev_state = game.state
        
        max_turns = 100  # hard safety cap to avoid infinite episodes
        while not game_over and episode_length < max_turns:
            # Get current state
            state = game.state
            
            # Check if it's our turn
            try:
                if state.current_player == 0:
                    # Select action
                    action = agent.select_action(state, 0)
                    
                    # Encode state using the agent's internal encoder
                    state_tensor = agent._encode_state(state)
                    
                    # Get valid actions
                    valid_actions = state.get_valid_actions(0)
                    
                    # Find action index
                    action_idx = valid_actions.index(action) if action in valid_actions else 0
                    
                    # Take action
                    next_state, game_over = game.step(action)
                    
                    # Calculate reward
                    reward = calculate_reward(state, next_state, game_over)
                    
                    # Add experience
                    collector.add_experience(
                        state=state_tensor,
                        action=action,
                        action_idx=action_idx,
                        reward=reward,
                        value=0.0,  # Placeholder, will be computed later
                        log_prob=0.0,  # Placeholder, will be computed later
                        done=game_over
                    )
                    
                    # Update statistics
                    episode_reward += reward
                    episode_length += 1
                    total_steps += 1
                    
                    # Store state for next iteration
                    prev_state = next_state
                else:
                    # Opponent's turn
                    next_state, game_over = game.step()
                    prev_state = next_state

            except NoValidActionError as e:
                # Edge-case: player stuck – end episode, count as draw
                if args.verbose:
                    print(f"[Warning] {e}. Ending episode early.")
                game_over = True
            except Exception as e:
                # Any unexpected error – end episode but continue training
                if args.verbose:
                    print(f"[Error] Unexpected exception: {e}. "
                          "Skipping remainder of episode.")
                game_over = True
        
        # Update episode statistics
        episode_rewards.append(episode_reward)
        total_episodes += 1
        
        # Update win rate
        if game_over and game.state.result == GameResult.WINNER:
            win_rates.append(1.0 if game.state.winner == 0 else 0.0)
        else:
            # Determine winner based on points
            if game.state.players[0].points > game.state.players[1].points:
                win_rates.append(1.0)
            elif game.state.players[0].points < game.state.players[1].points:
                win_rates.append(0.0)
            else:
                win_rates.append(0.5)  # Draw
        
        # Get batch of experiences
        if len(collector) > 0:
            batch = collector.get_batch()
            
            # Update policy
            metrics = agent.update(batch)
            
            if args.verbose and total_episodes % 10 == 0:
                print(f"Episode {total_episodes}, Reward: {episode_reward:.2f}, "
                      f"Win Rate: {np.mean(win_rates[-10:]):.2f}, "
                      f"Policy Loss: {metrics.get('policy_loss', 0):.4f}, "
                      f"Value Loss: {metrics.get('value_loss', 0):.4f}")
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            "reward": np.mean(episode_rewards[-10:]) if episode_rewards else 0,
            "win_rate": np.mean(win_rates[-10:]) if win_rates else 0
        })
    
    # Close progress bar
    pbar.close()
    
    # Return statistics
    return {
        "episode_rewards": episode_rewards,
        "win_rates": win_rates,
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "final_win_rate": np.mean(win_rates[-20:]) if win_rates else 0
    }


def evaluate_agent(agent, num_episodes=20):
    """Evaluate the agent's performance."""
    print(f"Evaluating {agent.name} for {num_episodes} episodes...")
    
    # Create game
    game = Game(num_players=2)
    
    # Register robust random opponent (handles no-action edge cases)
    random_agent = RobustRandomAgent(name="Robust Random Opponent")
    game.register_agent(1, random_agent.get_action_callback())
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0
    draws = 0
    
    # Progress bar
    pbar = tqdm(total=num_episodes, desc="Evaluating")
    
    # Evaluate for num_episodes
    for _ in range(num_episodes):
        # Reset game
        game.reset()
        
        # Play one episode
        episode_reward = 0
        episode_length = 0
        game_over = False
        
        # Store previous state for reward calculation
        prev_state = game.state
        
        while not game_over and episode_length < 100:  # Limit episode length
            try:
                # Get current state
                state = game.state

                if state.current_player == 0:
                    # Our deterministic action
                    action = agent.select_action(state, 0, deterministic=True)
                    next_state, game_over = game.step(action)
                else:
                    # Opponent's turn
                    next_state, game_over = game.step()

                # Reward & stats
                reward = calculate_reward(state, next_state, game_over)
                episode_reward += reward
                episode_length += 1
                prev_state = next_state

            except NoValidActionError as e:
                # Edge-case: opponent or agent stuck – treat as draw and end episode
                if isinstance(agent.config, PPOConfig):
                    print(f"[Warning] {e}. Ending evaluation episode early.")
                game_over = True
            except Exception as e:
                # Unexpected error, abort this episode but continue evaluation
                print(f"[Error] Evaluation exception: {e}. "
                      "Skipping remainder of episode.")
                game_over = True
        
        # Update episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Update win/loss/draw statistics
        if game_over and game.state.result == GameResult.WINNER:
            if game.state.winner == 0:
                wins += 1
            else:
                losses += 1
        else:
            # Determine winner based on points
            if game.state.players[0].points > game.state.players[1].points:
                wins += 1
            elif game.state.players[0].points < game.state.players[1].points:
                losses += 1
            else:
                draws += 1
        
        # Update progress bar
        pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    # Calculate statistics
    win_rate = wins / num_episodes
    loss_rate = losses / num_episodes
    draw_rate = draws / num_episodes
    
    # Print statistics
    print(f"Evaluation results:")
    print(f"  Win rate: {win_rate:.2f}")
    print(f"  Loss rate: {loss_rate:.2f}")
    print(f"  Draw rate: {draw_rate:.2f}")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"  Mean episode length: {np.mean(episode_lengths):.2f}")
    
    # Return statistics
    return {
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
        "mean_reward": np.mean(episode_rewards),
        "mean_length": np.mean(episode_lengths)
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
    train_stats = train_agent(agent, args)
    
    # Evaluate agent
    eval_stats = evaluate_agent(agent, args.eval_episodes)
    
    # Save agent
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    agent.save(args.save_path)
    print(f"Saved agent to {args.save_path}")
    
    # Print final statistics
    print("\nTraining complete!")
    print(f"Final win rate: {train_stats['final_win_rate']:.2f}")
    print(f"Evaluation win rate: {eval_stats['win_rate']:.2f}")


if __name__ == "__main__":
    main()
