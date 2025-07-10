#!/usr/bin/env python
"""
Training script for Splendor AI reinforcement learning agents.

This script provides a command-line interface for training different types of
reinforcement learning agents (PPO, A2C, and hybrid) for the Splendor board game.
It supports various training configurations, self-play, model saving/loading,
and evaluation.

Example usage:
    # Train a PPO agent with default settings
    python train_rl.py --agent ppo --episodes 1000 --save-dir models/ppo

    # Train an A2C agent with self-play
    python train_rl.py --agent a2c --self-play --episodes 2000 --save-dir models/a2c

    # Train a hybrid agent (MCTS + RL)
    python train_rl.py --agent hybrid --base-agent ppo --episodes 500 --save-dir models/hybrid

    # Continue training from a saved model
    python train_rl.py --agent ppo --load models/ppo/agent_best.pt --episodes 500
    
    # Evaluate a trained agent
    python train_rl.py --agent ppo --load models/ppo/agent_best.pt --eval-only --eval-episodes 100
"""
import os
import sys
import time
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from splendor_ai.core.game import Game, GameState, GameResult
from splendor_ai.core.actions import Action
from splendor_ai.core.player import Player

from splendor_ai.mcts.agent import MCTSAgent
from splendor_ai.mcts.config import MCTSConfig

from splendor_ai.rl.config import (
    RLConfig, PPOConfig, A2CConfig, NetworkConfig, 
    TrainingConfig, SplendorRLConfig
)
from splendor_ai.rl.agents import (
    RLAgent, PPOAgent, A2CAgent, RandomAgent, HybridAgent
)
from splendor_ai.rl.models import (
    PolicyNetwork, ValueNetwork, SplendorNetwork, 
    BoardGameNetwork, create_splendor_network
)
from splendor_ai.rl.training import (
    ExperienceCollector, ReplayBuffer, Trainer, 
    SelfPlayTrainer, train_agent, evaluate_agent,
    save_model, load_model, set_seed
)


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train RL agents for Splendor")
    
    # Agent configuration
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "a2c", "random", "hybrid"],
                        help="Type of agent to train")
    parser.add_argument("--base-agent", type=str, default="ppo", choices=["ppo", "a2c"],
                        help="Base agent type for hybrid agent")
    
    # Training configuration
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes to train for")
    parser.add_argument("--steps-per-update", type=int, default=2048,
                        help="Number of steps to collect before each policy update")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for policy updates")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for each policy update (PPO)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda parameter")
    parser.add_argument("--clip-ratio", type=float, default=0.2,
                        help="PPO clip ratio")
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="Value loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum gradient norm")
    
    # Network configuration
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden layer size for neural networks")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of hidden layers")
    parser.add_argument("--network-type", type=str, default="mlp", choices=["mlp", "attention", "transformer"],
                        help="Type of neural network architecture")
    
    # Self-play configuration
    parser.add_argument("--self-play", action="store_true",
                        help="Use self-play training")
    parser.add_argument("--num-opponents", type=int, default=5,
                        help="Number of opponents in self-play pool")
    parser.add_argument("--update-opponent-every", type=int, default=10,
                        help="Update opponent pool every N policy updates")
    parser.add_argument("--opponent-sampling", type=str, default="random", choices=["random", "latest", "elo"],
                        help="Method for sampling opponents from the pool")
    
    # Evaluation configuration
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Number of episodes for evaluation")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluate every N policy updates")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate, don't train")
    parser.add_argument("--eval-opponents", type=str, default="random,mcts",
                        help="Comma-separated list of opponents to evaluate against")
    
    # Saving and loading
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save model every N policy updates")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to load model from")
    
    # Logging and visualization
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory to save logs")
    parser.add_argument("--use-tensorboard", action="store_true",
                        help="Use TensorBoard for logging")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda)")
    parser.add_argument("--num-players", type=int, default=2,
                        help="Number of players in the game")
    
    return parser.parse_args()


def create_agent(args) -> RLAgent:
    """
    Create an agent based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Created agent
    """
    # Create network configuration
    # NetworkConfig expects a list named `hidden_sizes`. It does not accept
    # `network_type` or `device`, so we construct it accordingly.
    # We replicate the user-provided hidden size for the specified number of
    # layers (default 2). This keeps CLI behaviour identical while respecting
    # the dataclass definition.
    network_config = NetworkConfig(
        hidden_sizes=[args.hidden_size] * args.num_layers
    )
    
    # Create agent based on type
    if args.agent == "ppo":
        # Create PPO configuration
        ppo_config = PPOConfig(
            learning_rate=args.lr,
            gamma=args.gamma,
            lambda_gae=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            value_loss_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            epochs=args.epochs,
            batch_size=args.batch_size,
            network=network_config,
            device=args.device
        )
        
        # Create PPO agent
        agent = PPOAgent(config=ppo_config, name="PPO Agent")
    
    elif args.agent == "a2c":
        # Create A2C configuration
        a2c_config = A2CConfig(
            learning_rate=args.lr,
            gamma=args.gamma,
            lambda_gae=args.gae_lambda,
            value_loss_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            network=network_config,
            device=args.device
        )
        
        # Create A2C agent
        agent = A2CAgent(config=a2c_config, name="A2C Agent")
    
    elif args.agent == "random":
        # Create random agent
        agent = RandomAgent(name="Random Agent")
    
    elif args.agent == "hybrid":
        # Create base agent
        if args.base_agent == "ppo":
            # Create PPO configuration
            ppo_config = PPOConfig(
                learning_rate=args.lr,
                gamma=args.gamma,
                lambda_gae=args.gae_lambda,
                clip_ratio=args.clip_ratio,
                value_loss_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                epochs=args.epochs,
                batch_size=args.batch_size,
                network=network_config,
                device=args.device
            )
            
            # Create PPO agent
            base_agent = PPOAgent(config=ppo_config, name="PPO Base")
        
        else:  # a2c
            # Create A2C configuration
            a2c_config = A2CConfig(
                learning_rate=args.lr,
                gamma=args.gamma,
                lambda_gae=args.gae_lambda,
                value_loss_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                network=network_config,
                device=args.device
            )
            
            # Create A2C agent
            base_agent = A2CAgent(config=a2c_config, name="A2C Base")
        
        # Create MCTS configuration
        mcts_config = MCTSConfig(
            iterations=100,
            exploration_weight=1.0
        )
        
        # Create hybrid agent
        agent = HybridAgent(
            rl_agent=base_agent,
            mcts_config=mcts_config,
            name=f"Hybrid {args.base_agent.upper()} Agent"
        )
    
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")
    
    # Load model if specified
    if args.load is not None:
        print(f"Loading model from {args.load}")
        agent.load(args.load)
    
    return agent


def create_training_config(args) -> TrainingConfig:
    """
    Create a training configuration based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Training configuration
    """
    # Calculate total timesteps from episodes
    # Assuming average episode length of 30 steps
    total_timesteps = args.episodes * 30
    
    # Create training configuration
    config = TrainingConfig(
        total_timesteps=total_timesteps,
        steps_per_update=args.steps_per_update,
        eval_episodes=args.eval_episodes,
        checkpoint_dir=args.save_dir,
        log_dir=args.log_dir,
        use_tensorboard=args.use_tensorboard,
        seed=args.seed,
        num_opponents=args.num_opponents,
        update_opponent_every=args.update_opponent_every,
        opponent_sampling=args.opponent_sampling,
        max_episode_length=100  # Prevent infinite games
    )
    
    # Create directories
    config.create_directories()
    
    return config


def create_game(args) -> Game:
    """
    Create a game instance based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Game instance
    """
    # Create game
    game = Game(num_players=args.num_players)
    
    return game


def train(args, agent: RLAgent, game_class: type) -> Dict[str, Any]:
    """
    Train an agent.
    
    Args:
        args: Command-line arguments
        agent: Agent to train
        game_class: Game class to use
        
    Returns:
        Training statistics
    """
    # Create training configuration
    config = create_training_config(args)
    
    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Print training configuration
    if args.verbose:
        print("Training configuration:")
        for key, value in config.__dict__.items():
            print(f"  {key}: {value}")
    
    # Train agent
    print(f"Training {agent.name} for {args.episodes} episodes...")
    
    if args.self_play:
        # Create self-play trainer
        trainer = SelfPlayTrainer(
            agent=agent,
            game_class=game_class,
            config=config,
            game_kwargs={"num_players": args.num_players}
        )
    else:
        # Create game instance
        game = game_class(num_players=args.num_players)
        
        # Register agent with game
        game.register_agent(0, agent.get_action_callback())
        
        # Register random opponent
        random_agent = RandomAgent(name="Random Opponent")
        game.register_agent(1, random_agent.get_action_callback())
        
        # Create trainer
        trainer = Trainer(
            agent=agent,
            env=game,
            config=config
        )
    
    # Train agent
    stats = trainer.train()
    
    # Save final model
    final_path = os.path.join(args.save_dir, f"{agent.name}_final.pt")
    agent.save(final_path)
    print(f"Saved final model to {final_path}")
    
    return stats


def evaluate(args, agent: RLAgent, game_class: type) -> Dict[str, float]:
    """
    Evaluate an agent against various opponents.
    
    Args:
        args: Command-line arguments
        agent: Agent to evaluate
        game_class: Game class to use
        
    Returns:
        Evaluation statistics
    """
    # Parse opponent types
    opponent_types = args.eval_opponents.split(",")
    
    # Create game instance
    game = game_class(num_players=args.num_players)
    
    # Register agent with game
    game.register_agent(0, agent.get_action_callback())
    
    # Evaluate against each opponent type
    all_stats = []
    
    for opponent_type in opponent_types:
        # Create opponent
        if opponent_type == "random":
            opponent = RandomAgent(name="Random Opponent")
        elif opponent_type == "mcts":
            mcts_config = MCTSConfig(
                iterations=100,
                exploration_weight=1.0
            )
            opponent = MCTSAgent(config=mcts_config, name="MCTS Opponent")
        else:
            print(f"Unknown opponent type: {opponent_type}, skipping")
            continue
        
        # Register opponent with game
        game.register_agent(1, opponent.get_action_callback())
        
        # Statistics
        episode_rewards = []
        episode_lengths = []
        wins = 0
        losses = 0
        draws = 0
        
        # Evaluate for eval_episodes
        print(f"Evaluating against {opponent.name} for {args.eval_episodes} episodes...")
        for _ in tqdm(range(args.eval_episodes)):
            # Reset game
            game.reset()
            
            # Reset episode statistics
            episode_reward = 0
            episode_length = 0
            game_over = False
            
            # Play game
            while not game_over and episode_length < 100:  # Prevent infinite games
                # Get current state
                state = game.state
                
                # Check if it's our turn
                if state.current_player == 0:
                    # Select action
                    action = agent.select_action(state, 0, deterministic=True)
                    
                    # Take action
                    _, game_over = game.step(action)
                    
                    # Update statistics
                    episode_length += 1
                else:
                    # Opponent's turn
                    _, game_over = game.step()
            
            # Update statistics
            episode_lengths.append(episode_length)
            
            # Check game result
            if game_over:
                if game.state.result == GameResult.WINNER:
                    if game.state.winner == 0:
                        wins += 1
                        episode_reward = 1.0
                    else:
                        losses += 1
                        episode_reward = -1.0
                else:
                    draws += 1
                    episode_reward = 0.0
            else:
                # Game didn't finish (reached max length)
                # Determine winner based on points
                if game.state.players[0].points > game.state.players[1].points:
                    wins += 1
                    episode_reward = 1.0
                elif game.state.players[0].points < game.state.players[1].points:
                    losses += 1
                    episode_reward = -1.0
                else:
                    draws += 1
                    episode_reward = 0.0
            
            episode_rewards.append(episode_reward)
        
        # Calculate statistics
        stats = {
            "opponent": opponent.name,
            "mean_reward": np.mean(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "win_rate": wins / args.eval_episodes,
            "loss_rate": losses / args.eval_episodes,
            "draw_rate": draws / args.eval_episodes,
            "num_episodes": args.eval_episodes
        }
        
        # Print statistics
        print(f"Evaluation against {opponent.name}:")
        print(f"  Win rate: {stats['win_rate']:.2f}")
        print(f"  Loss rate: {stats['loss_rate']:.2f}")
        print(f"  Draw rate: {stats['draw_rate']:.2f}")
        print(f"  Mean reward: {stats['mean_reward']:.2f}")
        print(f"  Mean episode length: {stats['mean_length']:.2f}")
        
        all_stats.append(stats)
    
    # Calculate overall statistics
    overall_stats = {
        "mean_reward": np.mean([s["mean_reward"] for s in all_stats]),
        "mean_length": np.mean([s["mean_length"] for s in all_stats]),
        "win_rate": np.mean([s["win_rate"] for s in all_stats]),
        "loss_rate": np.mean([s["loss_rate"] for s in all_stats]),
        "draw_rate": np.mean([s["draw_rate"] for s in all_stats]),
        "num_episodes": args.eval_episodes * len(opponent_types),
        "opponent_stats": all_stats
    }
    
    # Print overall statistics
    print("\nOverall evaluation:")
    print(f"  Win rate: {overall_stats['win_rate']:.2f}")
    print(f"  Loss rate: {overall_stats['loss_rate']:.2f}")
    print(f"  Draw rate: {overall_stats['draw_rate']:.2f}")
    print(f"  Mean reward: {overall_stats['mean_reward']:.2f}")
    print(f"  Mean episode length: {overall_stats['mean_length']:.2f}")
    
    return overall_stats


def visualize_training(stats: Dict[str, Any], save_dir: str) -> None:
    """
    Visualize training progress.
    
    Args:
        stats: Training statistics
        save_dir: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    if "training_stats" in stats:
        # Extract from agent stats
        training_stats = stats["training_stats"]
        
        # Plot policy loss
        if "policy_loss" in training_stats:
            plt.figure(figsize=(10, 6))
            plt.plot(training_stats["policy_loss"])
            plt.title("Policy Loss")
            plt.xlabel("Update")
            plt.ylabel("Loss")
            plt.savefig(os.path.join(save_dir, "policy_loss.png"))
            plt.close()
        
        # Plot value loss
        if "value_loss" in training_stats:
            plt.figure(figsize=(10, 6))
            plt.plot(training_stats["value_loss"])
            plt.title("Value Loss")
            plt.xlabel("Update")
            plt.ylabel("Loss")
            plt.savefig(os.path.join(save_dir, "value_loss.png"))
            plt.close()
        
        # Plot entropy
        if "entropy" in training_stats:
            plt.figure(figsize=(10, 6))
            plt.plot(training_stats["entropy"])
            plt.title("Entropy")
            plt.xlabel("Update")
            plt.ylabel("Entropy")
            plt.savefig(os.path.join(save_dir, "entropy.png"))
            plt.close()
        
        # Plot KL divergence
        if "approx_kl" in training_stats:
            plt.figure(figsize=(10, 6))
            plt.plot(training_stats["approx_kl"])
            plt.title("Approximate KL Divergence")
            plt.xlabel("Update")
            plt.ylabel("KL")
            plt.savefig(os.path.join(save_dir, "kl.png"))
            plt.close()
    
    # Plot win rate over time if available
    if "win_rates" in stats:
        plt.figure(figsize=(10, 6))
        plt.plot(stats["win_rates"])
        plt.title("Win Rate")
        plt.xlabel("Evaluation")
        plt.ylabel("Win Rate")
        plt.savefig(os.path.join(save_dir, "win_rate.png"))
        plt.close()
    
    # Plot reward over time if available
    if "rewards" in stats:
        plt.figure(figsize=(10, 6))
        plt.plot(stats["rewards"])
        plt.title("Mean Reward")
        plt.xlabel("Evaluation")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(save_dir, "reward.png"))
        plt.close()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create agent
    agent = create_agent(args)
    print(f"Created agent: {agent.name}")
    
    # Create game class
    game_class = Game
    
    # Evaluation only
    if args.eval_only:
        if args.load is None:
            print("Error: Must specify a model to load for evaluation")
            sys.exit(1)
        
        # Evaluate agent
        evaluate(args, agent, game_class)
        return
    
    # Train agent
    stats = train(args, agent, game_class)
    
    # Print training statistics
    print("\nTraining complete!")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Best win rate: {stats['best_win_rate']:.2f}")
    
    # Visualize training
    visualize_training(stats, args.save_dir)
    
    # Evaluate final agent
    print("\nEvaluating final agent...")
    evaluate(args, agent, game_class)


if __name__ == "__main__":
    main()
