"""
Training infrastructure for reinforcement learning agents.

This module provides the training infrastructure for RL agents, including:

1. Experience collection from gameplay
2. Replay buffer for storing and sampling experiences
3. Training loop for updating agent policies
4. Self-play training for competitive learning
5. Evaluation mechanisms for measuring performance
6. Model management for saving and loading

The training system supports both single-agent and multi-agent learning,
with a focus on self-play for board games like Splendor.
"""
import os
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import deque
import pickle
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from splendor_ai.core.game import Game, GameState, GameResult
from splendor_ai.core.actions import Action
from splendor_ai.rl.config import RLConfig, TrainingConfig, SplendorRLConfig
from splendor_ai.rl.agents import RLAgent, RandomAgent


class ExperienceCollector:
    """
    Collects experiences during gameplay for training.
    
    This class tracks states, actions, rewards, and other information
    during gameplay, and processes them into training data.
    """
    
    def __init__(self, gamma: float = 0.99, lambda_gae: float = 0.95):
        """
        Initialize the experience collector.
        
        Args:
            gamma: Discount factor for future rewards
            lambda_gae: Lambda parameter for Generalized Advantage Estimation
        """
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.reset()
    
    def reset(self) -> None:
        """Reset the collector, clearing all stored experiences."""
        self.states = []
        self.actions = []
        self.action_indices = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.masks = []  # Action masks
    
    def add_experience(
        self,
        state: torch.Tensor,
        action: Union[Action, int],
        action_idx: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        mask: Optional[torch.Tensor] = None
    ) -> None:
        """
        Add an experience to the collector.
        
        Args:
            state: State tensor
            action: Action taken (either Action object or index)
            action_idx: Index of the action in the action space
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of the action
            done: Whether the episode is done
            mask: Action mask
        """
        self.states.append(state)
        self.actions.append(action)
        self.action_indices.append(action_idx)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        if mask is not None:
            self.masks.append(mask)
    
    def compute_returns_and_advantages(
        self,
        last_value: float = 0.0,
        use_gae: bool = True
    ) -> Tuple[List[float], List[float]]:
        """
        Compute returns and advantages for collected experiences.
        
        Args:
            last_value: Value estimate for the final state
            use_gae: Whether to use Generalized Advantage Estimation
            
        Returns:
            Tuple of (returns, advantages)
        """
        # Compute returns and advantages
        returns = []
        advantages = []
        
        if use_gae:
            # Generalized Advantage Estimation
            gae = 0
            next_value = last_value
            for i in reversed(range(len(self.rewards))):
                # Calculate TD error
                delta = self.rewards[i] + self.gamma * next_value * (1 - self.dones[i]) - self.values[i]
                
                # Update GAE
                gae = delta + self.gamma * self.lambda_gae * (1 - self.dones[i]) * gae
                
                # Update return
                returns.insert(0, gae + self.values[i])
                advantages.insert(0, gae)
                
                next_value = self.values[i]
        else:
            # Monte Carlo returns
            next_return = last_value
            for i in reversed(range(len(self.rewards))):
                next_return = self.rewards[i] + self.gamma * next_return * (1 - self.dones[i])
                returns.insert(0, next_return)
            
            # Advantages are returns minus values
            advantages = [ret - val for ret, val in zip(returns, self.values)]
        
        return returns, advantages
    
    def get_batch(
        self,
        last_value: float = 0.0,
        use_gae: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Get a batch of experiences for training.
        
        Args:
            last_value: Value estimate for the final state
            use_gae: Whether to use Generalized Advantage Estimation
            
        Returns:
            Dictionary of batched experiences
        """
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(last_value, use_gae)
        
        # Convert to tensors
        batch = {
            "states": torch.cat(self.states),
            "actions": torch.tensor(self.action_indices),
            "log_probs": torch.tensor(self.log_probs),
            "returns": torch.tensor(returns),
            "advantages": torch.tensor(advantages),
            "values": torch.tensor(self.values)
        }
        
        # Add masks if available
        if self.masks:
            batch["masks"] = torch.cat(self.masks)
        
        return batch
    
    def __len__(self) -> int:
        """Get the number of experiences collected."""
        return len(self.states)


class ReplayBuffer:
    """
    Buffer for storing and sampling experiences.
    
    This class provides a fixed-size buffer for storing experiences
    and sampling them for training.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, experience: Dict[str, torch.Tensor]) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            experience: Dictionary of experience data
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary of batched experiences
        """
        batch_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), batch_size)
        
        # Get samples
        samples = [self.buffer[i] for i in indices]
        
        # Combine samples into a batch
        batch = {}
        for key in samples[0].keys():
            batch[key] = torch.cat([sample[key] for sample in samples])
        
        return batch
    
    def __len__(self) -> int:
        """Get the number of experiences in the buffer."""
        return len(self.buffer)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the buffer to disk.
        
        Args:
            path: Path to save the buffer
        """
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the buffer from disk.
        
        Args:
            path: Path to load the buffer from
        """
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)
        self.position = len(self.buffer) % self.capacity


class Trainer:
    """
    Trainer for reinforcement learning agents.
    
    This class manages the training process, including experience collection,
    policy updates, and evaluation.
    """
    
    def __init__(
        self,
        agent: RLAgent,
        env: Any,  # Environment or game
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            agent: RL agent to train
            env: Environment or game
            config: Training configuration
        """
        self.agent = agent
        self.env = env
        self.config = config or TrainingConfig()
        
        # Create directories
        self.config.create_directories()
        
        # Create experience collector
        if hasattr(agent, 'config') and hasattr(agent.config, 'gamma'):
            gamma = agent.config.gamma
            lambda_gae = getattr(agent.config, 'lambda_gae', 0.95)
        else:
            gamma = 0.99
            lambda_gae = 0.95
        
        self.collector = ExperienceCollector(gamma=gamma, lambda_gae=lambda_gae)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Create tensorboard writer
        if self.config.use_tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(self.config.log_dir, f"{agent.name}_{time.strftime('%Y%m%d_%H%M%S')}"))
        else:
            self.writer = None
        
        # Set random seed
        if self.config.seed is not None:
            set_seed(self.config.seed)
        
        # Training stats
        self.total_steps = 0
        self.total_episodes = 0
        self.updates = 0
        self.best_reward = float('-inf')
        self.best_win_rate = 0.0
        
        # Opponent pool for self-play
        self.opponent_pool = []
    
    def collect_experience(
        self,
        num_steps: int,
        deterministic: bool = False
    ) -> Dict[str, float]:
        """
        Collect experience by playing games.
        
        Args:
            num_steps: Number of steps to collect
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary of statistics
        """
        # Reset collector
        self.collector.reset()
        
        # Reset environment
        state = self.env.reset()
        
        # Statistics
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        # Collect experience
        for _ in range(num_steps):
            # Select action
            action, action_idx, log_prob, value, mask = self._select_action(state, deterministic)
            
            # Take action
            next_state, reward, done, info = self.env.step(action)
            
            # Add experience
            self.collector.add_experience(
                state=state,
                action=action,
                action_idx=action_idx,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                mask=mask
            )
            
            # Update state
            state = next_state
            
            # Update statistics
            current_episode_reward += reward
            current_episode_length += 1
            
            # Check if episode is done
            if done:
                # Reset environment
                state = self.env.reset()
                
                # Update statistics
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0
                current_episode_length = 0
                
                # Update total episodes
                self.total_episodes += 1
        
        # Update total steps
        self.total_steps += num_steps
        
        # Calculate statistics
        stats = {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "num_episodes": len(episode_rewards),
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes
        }
        
        return stats
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update the agent's policy based on collected experiences.
        
        Returns:
            Dictionary of training metrics
        """
        # Get batch of experiences
        batch = self.collector.get_batch(use_gae=True)
        
        # Update agent
        metrics = self.agent.update(batch)
        
        # Update counter
        self.updates += 1
        
        return metrics
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        eval_interval: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the agent for a specified number of timesteps.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            eval_interval: How often to evaluate the agent
            
        Returns:
            Dictionary of training statistics
        """
        # Use config values if not specified
        total_timesteps = total_timesteps or self.config.total_timesteps
        eval_interval = eval_interval or getattr(self.config, "eval_interval", 10)
        save_interval = getattr(self.config, "save_interval", 10)
        
        # Initialize progress bar
        pbar = tqdm(total=total_timesteps, desc=f"Training {self.agent.name}")
        pbar.update(self.total_steps)
        
        # Training loop
        while self.total_steps < total_timesteps:
            # Collect experience
            steps_per_update = self.config.steps_per_update
            collect_stats = self.collect_experience(steps_per_update)
            
            # Update policy
            update_metrics = self.update_policy()
            
            # Log statistics
            if self.writer is not None:
                # Log collection statistics
                for key, value in collect_stats.items():
                    self.writer.add_scalar(f"train/{key}", value, self.total_steps)
                
                # Log update metrics
                for key, value in update_metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, self.total_steps)
            
            # Evaluate agent
            if eval_interval > 0 and self.updates % eval_interval == 0:
                eval_stats = self.evaluate()
                
                # Log evaluation statistics
                if self.writer is not None:
                    for key, value in eval_stats.items():
                        self.writer.add_scalar(f"eval/{key}", value, self.total_steps)
                
                # Save best model
                if eval_stats.get("mean_reward", 0.0) > self.best_reward:
                    self.best_reward = eval_stats["mean_reward"]
                    self.save_checkpoint(os.path.join(self.config.checkpoint_dir, f"{self.agent.name}_best_reward.pt"))
                
                if eval_stats.get("win_rate", 0.0) > self.best_win_rate:
                    self.best_win_rate = eval_stats["win_rate"]
                    self.save_checkpoint(os.path.join(self.config.checkpoint_dir, f"{self.agent.name}_best_win_rate.pt"))
            
            # Save checkpoint
            if self.updates % save_interval == 0:
                self.save_checkpoint(os.path.join(self.config.checkpoint_dir, f"{self.agent.name}_{self.updates}.pt"))
            
            # Update progress bar
            pbar.update(steps_per_update)
            pbar.set_postfix({
                "reward": collect_stats.get("mean_reward", 0.0),
                "episodes": self.total_episodes,
                "updates": self.updates
            })
        
        # Close progress bar
        pbar.close()
        
        # Final evaluation
        final_stats = self.evaluate()
        
        # Save final model
        self.save_checkpoint(os.path.join(self.config.checkpoint_dir, f"{self.agent.name}_final.pt"))
        
        # Close tensorboard writer
        if self.writer is not None:
            self.writer.close()
        
        # Return statistics
        return {
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "updates": self.updates,
            "best_reward": self.best_reward,
            "best_win_rate": self.best_win_rate,
            "final_stats": final_stats
        }
    
    def evaluate(
        self,
        num_episodes: Optional[int] = None,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Args:
            num_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary of evaluation statistics
        """
        # Use config value if not specified
        num_episodes = num_episodes or self.config.eval_episodes
        
        # Reset environment
        state = self.env.reset()
        
        # Statistics
        episode_rewards = []
        episode_lengths = []
        wins = 0
        
        # Evaluate for num_episodes
        for _ in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            
            # Reset episode statistics
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Run episode
            while not done and episode_length < self.config.max_episode_length:
                # Select action
                action, _, _, _, _ = self._select_action(state, deterministic)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Update state
                state = next_state
                
                # Update statistics
                episode_reward += reward
                episode_length += 1
            
            # Update statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check if agent won
            if info.get("winner", -1) == 0:  # Assuming agent is player 0
                wins += 1
        
        # Calculate statistics
        stats = {
            "mean_reward": np.mean(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "win_rate": wins / num_episodes,
            "num_episodes": num_episodes
        }
        
        return stats
    
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Save a training checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save agent
        self.agent.save(path)
        
        # Save trainer state
        trainer_state = {
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "updates": self.updates,
            "best_reward": self.best_reward,
            "best_win_rate": self.best_win_rate,
            "config": self.config.__dict__
        }
        
        # Save trainer state
        with open(f"{path}_trainer.json", 'w') as f:
            json.dump(trainer_state, f)
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Load a training checkpoint.
        
        Args:
            path: Path to load the checkpoint from
        """
        # Load agent
        self.agent.load(path)
        
        # Load trainer state
        try:
            with open(f"{path}_trainer.json", 'r') as f:
                trainer_state = json.load(f)
            
            # Restore trainer state
            self.total_steps = trainer_state["total_steps"]
            self.total_episodes = trainer_state["total_episodes"]
            self.updates = trainer_state["updates"]
            self.best_reward = trainer_state["best_reward"]
            self.best_win_rate = trainer_state["best_win_rate"]
            
            # Restore config
            if "config" in trainer_state:
                for key, value in trainer_state["config"].items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Warning: Trainer state file not found or invalid: {path}_trainer.json")
    
    def _select_action(
        self,
        state: Any,
        deterministic: bool = False
    ) -> Tuple[Any, int, float, float, Optional[torch.Tensor]]:
        """
        Select an action using the agent.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (action, action index, log probability, value, mask)
        """
        # This is a placeholder - in practice, you would implement
        # a more sophisticated action selection method
        
        # Get action from agent
        action = self.agent.select_action(state, 0, deterministic)
        
        # Placeholder values
        action_idx = 0
        log_prob = 0.0
        value = 0.0
        mask = None
        
        return action, action_idx, log_prob, value, mask


class SelfPlayTrainer(Trainer):
    """
    Trainer for self-play reinforcement learning.
    
    This class extends the base Trainer with self-play capabilities,
    allowing agents to learn by playing against themselves or past versions.
    """
    
    def __init__(
        self,
        agent: RLAgent,
        game_class: type,
        config: Optional[TrainingConfig] = None,
        game_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the self-play trainer.
        
        Args:
            agent: RL agent to train
            game_class: Game class to use
            config: Training configuration
            game_kwargs: Additional arguments for game creation
        """
        super().__init__(agent, None, config)
        
        # Store game class and kwargs
        self.game_class = game_class
        self.game_kwargs = game_kwargs or {}
        
        # Create a game instance
        self.game = self.game_class(**self.game_kwargs)
        
        # Register agent with game
        self.game.register_agent(0, self.agent.get_action_callback())
        
        # Create opponent pool
        self.opponent_pool = []
        
        # Add random agent as initial opponent
        random_agent = RandomAgent(name="Random")
        self.opponent_pool.append(random_agent)
        
        # Register opponent with game
        self.game.register_agent(1, random_agent.get_action_callback())
    
    def collect_experience(
        self,
        num_steps: int,
        deterministic: bool = False
    ) -> Dict[str, float]:
        """
        Collect experience by playing games against opponents.
        
        Args:
            num_steps: Number of steps to collect
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary of statistics
        """
        # Reset collector
        self.collector.reset()
        
        # Statistics
        episode_rewards = []
        episode_lengths = []
        wins = 0
        losses = 0
        draws = 0
        steps_collected = 0
        
        # Collect experience
        while steps_collected < num_steps:
            # Select opponent
            opponent = self._select_opponent()
            
            # Register opponent with game
            self.game.register_agent(1, opponent.get_action_callback())
            
            # Reset game
            self.game.reset()
            
            # Play game
            game_over = False
            episode_reward = 0
            episode_length = 0
            
            while not game_over and episode_length < self.config.max_episode_length:
                # Get current state
                state = self.game.state
                
                # Check if it's our turn
                if state.current_player == 0:
                    # Select action
                    action, action_idx, log_prob, value, mask = self._select_action(state, deterministic)
                    
                    # Take action
                    _, game_over = self.game.step(action)
                    
                    # Calculate reward
                    reward = self._calculate_reward(state, self.game.state, game_over)
                    
                    # Add experience
                    self.collector.add_experience(
                        state=self._encode_state(state),
                        action=action,
                        action_idx=action_idx,
                        reward=reward,
                        value=value,
                        log_prob=log_prob,
                        done=game_over,
                        mask=mask
                    )
                    
                    # Update statistics
                    episode_reward += reward
                    episode_length += 1
                    steps_collected += 1
                else:
                    # Opponent's turn
                    _, game_over = self.game.step()
            
            # Update statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check game result
            if game_over:
                if self.game.state.result == GameResult.WINNER:
                    if self.game.state.winner == 0:
                        wins += 1
                    else:
                        losses += 1
                else:
                    draws += 1
            
            # Update total episodes
            self.total_episodes += 1
        
        # Update total steps
        self.total_steps += steps_collected
        
        # Calculate statistics
        stats = {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "win_rate": wins / (wins + losses + draws) if (wins + losses + draws) > 0 else 0.0,
            "num_episodes": len(episode_rewards),
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes
        }
        
        return stats
    
    def update_opponent_pool(self) -> None:
        """Update the opponent pool with a snapshot of the current agent."""
        # Create a copy of the current agent
        agent_copy = type(self.agent)(
            config=self.agent.config,
            name=f"{self.agent.name}_v{self.updates}"
        )
        
        # Save current agent to temporary file
        temp_path = os.path.join(self.config.checkpoint_dir, "temp_agent.pt")
        self.agent.save(temp_path)
        
        # Load copy from temporary file
        agent_copy.load(temp_path)
        
        # Add to opponent pool
        self.opponent_pool.append(agent_copy)
        
        # Limit pool size
        if len(self.opponent_pool) > self.config.num_opponents:
            # Remove oldest opponent (excluding random agent)
            self.opponent_pool.pop(1)
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        eval_interval: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the agent using self-play.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            eval_interval: How often to evaluate the agent
            
        Returns:
            Dictionary of training statistics
        """
        # Use config values if not specified
        total_timesteps = total_timesteps or self.config.total_timesteps
        # Some configs (e.g. the vanilla TrainingConfig) do not define
        # ``eval_interval`` / ``save_interval``.  Fallback to sensible defaults.
        eval_interval = eval_interval or getattr(self.config, "eval_interval", 10)
        save_interval = getattr(self.config, "save_interval", 10)
        
        # Initialize progress bar
        pbar = tqdm(total=total_timesteps, desc=f"Training {self.agent.name}")
        pbar.update(self.total_steps)
        
        # Training loop
        while self.total_steps < total_timesteps:
            # Collect experience
            steps_per_update = self.config.steps_per_update
            collect_stats = self.collect_experience(steps_per_update)
            
            # Update policy
            update_metrics = self.update_policy()
            
            # Log statistics
            if self.writer is not None:
                # Log collection statistics
                for key, value in collect_stats.items():
                    self.writer.add_scalar(f"train/{key}", value, self.total_steps)
                
                # Log update metrics
                for key, value in update_metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, self.total_steps)
            
            # Update opponent pool
            if self.updates % self.config.update_opponent_every == 0:
                self.update_opponent_pool()
            
            # Evaluate agent
            if eval_interval > 0 and self.updates % eval_interval == 0:
                eval_stats = self.evaluate()
                
                # Log evaluation statistics
                if self.writer is not None:
                    for key, value in eval_stats.items():
                        self.writer.add_scalar(f"eval/{key}", value, self.total_steps)
                
                # Save best model
                if eval_stats.get("win_rate", 0.0) > self.best_win_rate:
                    self.best_win_rate = eval_stats["win_rate"]
                    self.save_checkpoint(os.path.join(self.config.checkpoint_dir, f"{self.agent.name}_best_win_rate.pt"))
            
            # Save checkpoint
            if self.updates % save_interval == 0:
                self.save_checkpoint(os.path.join(self.config.checkpoint_dir, f"{self.agent.name}_{self.updates}.pt"))
            
            # Update progress bar
            pbar.update(steps_per_update)
            pbar.set_postfix({
                "win_rate": collect_stats.get("win_rate", 0.0),
                "episodes": self.total_episodes,
                "updates": self.updates
            })
        
        # Close progress bar
        pbar.close()
        
        # Final evaluation
        final_stats = self.evaluate()
        
        # Save final model
        self.save_checkpoint(os.path.join(self.config.checkpoint_dir, f"{self.agent.name}_final.pt"))
        
        # Close tensorboard writer
        if self.writer is not None:
            self.writer.close()
        
        # Return statistics
        return {
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "updates": self.updates,
            "best_win_rate": self.best_win_rate,
            "final_stats": final_stats
        }
    
    def evaluate(
        self,
        num_episodes: Optional[int] = None,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the agent against all opponents.
        
        Args:
            num_episodes: Number of episodes to evaluate per opponent
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary of evaluation statistics
        """
        # Use config value if not specified
        num_episodes = num_episodes or self.config.eval_episodes
        
        # Statistics
        all_stats = []
        
        # Evaluate against each opponent
        for opponent in self.opponent_pool:
            # Register opponent with game
            self.game.register_agent(1, opponent.get_action_callback())
            
            # Statistics
            episode_rewards = []
            episode_lengths = []
            wins = 0
            losses = 0
            draws = 0
            
            # Evaluate for num_episodes
            for _ in range(num_episodes):
                # Reset game
                self.game.reset()
                
                # Reset episode statistics
                episode_reward = 0
                episode_length = 0
                game_over = False
                
                # Play game
                while not game_over and episode_length < self.config.max_episode_length:
                    # Get current state
                    state = self.game.state
                    
                    # Check if it's our turn
                    if state.current_player == 0:
                        # Select action
                        action = self.agent.select_action(state, 0, deterministic)
                        
                        # Take action
                        _, game_over = self.game.step(action)
                        
                        # Calculate reward
                        reward = self._calculate_reward(state, self.game.state, game_over)
                        
                        # Update statistics
                        episode_reward += reward
                        episode_length += 1
                    else:
                        # Opponent's turn
                        _, game_over = self.game.step()
                
                # Update statistics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Check game result
                if game_over:
                    if self.game.state.result == GameResult.WINNER:
                        if self.game.state.winner == 0:
                            wins += 1
                        else:
                            losses += 1
                    else:
                        draws += 1
            
            # Calculate statistics
            stats = {
                "opponent": opponent.name,
                "mean_reward": np.mean(episode_rewards),
                "mean_length": np.mean(episode_lengths),
                "win_rate": wins / num_episodes,
                "loss_rate": losses / num_episodes,
                "draw_rate": draws / num_episodes,
                "num_episodes": num_episodes
            }
            
            all_stats.append(stats)
        
        # Calculate overall statistics
        overall_stats = {
            "mean_reward": np.mean([s["mean_reward"] for s in all_stats]),
            "mean_length": np.mean([s["mean_length"] for s in all_stats]),
            "win_rate": np.mean([s["win_rate"] for s in all_stats]),
            "loss_rate": np.mean([s["loss_rate"] for s in all_stats]),
            "draw_rate": np.mean([s["draw_rate"] for s in all_stats]),
            "num_episodes": num_episodes * len(self.opponent_pool),
            "opponent_stats": all_stats
        }
        
        return overall_stats
    
    def _select_opponent(self) -> RLAgent:
        """
        Select an opponent from the pool.
        
        Returns:
            Selected opponent
        """
        if self.config.opponent_sampling == "latest":
            # Select the most recent opponent
            return self.opponent_pool[-1]
        
        elif self.config.opponent_sampling == "random":
            # Select a random opponent
            return random.choice(self.opponent_pool)
        
        elif self.config.opponent_sampling == "elo":
            # Select opponent based on Elo rating (not implemented)
            # For now, just select randomly
            return random.choice(self.opponent_pool)
        
        else:
            # Default to random
            return random.choice(self.opponent_pool)
    
    def _calculate_reward(
        self,
        state: GameState,
        next_state: GameState,
        done: bool
    ) -> float:
        """
        Calculate reward for a transition.
        
        Args:
            state: Current state
            next_state: Next state
            done: Whether the episode is done
            
        Returns:
            Reward value
        """
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
        
        # Reward shaping (optional)
        if hasattr(self, 'reward_shaping') and self.reward_shaping:
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
    
    def _encode_state(self, state: GameState) -> torch.Tensor:
        """
        Encode a game state as a tensor.
        
        Args:
            state: Game state
            
        Returns:
            Encoded state tensor
        """
        # This is a placeholder - in practice, you would implement
        # a more sophisticated state encoding
        
        # Simple feature extraction
        features = []
        
        # Game progress
        features.append(state.turn_count / 30.0)  # Normalize by assumed max turns
        
        # Gem pool
        for color in state.gem_pool:
            features.append(state.gem_pool[color] / 10.0)  # Normalize by max gems
        
        # Current player
        features.append(1.0 if state.current_player == 0 else 0.0)
        
        # Player 0 (our agent)
        player = state.players[0]
        
        # Gems
        for color in player.gems:
            features.append(player.gems[color] / 10.0)
        
        # Bonuses
        for color in player.bonuses:
            features.append(player.bonuses[color] / 10.0)
        
        # Points
        features.append(player.points / 15.0)
        
        # Card counts
        features.append(len(player.cards) / 20.0)
        features.append(len(player.reserved_cards) / 3.0)
        features.append(len(player.nobles) / 5.0)
        
        # Player 1 (opponent)
        if len(state.players) > 1:
            player = state.players[1]
            
            # Gems
            for color in player.gems:
                features.append(player.gems[color] / 10.0)
            
            # Bonuses
            for color in player.bonuses:
                features.append(player.bonuses[color] / 10.0)
            
            # Points
            features.append(player.points / 15.0)
            
            # Card counts
            features.append(len(player.cards) / 20.0)
            features.append(len(player.reserved_cards) / 3.0)
            features.append(len(player.nobles) / 5.0)
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _select_action(
        self,
        state: GameState,
        deterministic: bool = False
    ) -> Tuple[Action, int, float, float, Optional[torch.Tensor]]:
        """
        Select an action and return additional information.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (action, action index, log probability, value, mask)
        """
        # Get action from agent
        action = self.agent.select_action(state, 0, deterministic)
        
        # Encode state
        state_tensor = self._encode_state(state)
        
        # Get valid actions
        valid_actions = state.get_valid_actions(0)
        
        # Find action index
        action_idx = valid_actions.index(action) if action in valid_actions else 0
        
        # Placeholder values
        log_prob = 0.0
        value = 0.0
        
        # Create action mask
        mask = torch.zeros(1, 300)  # Assuming action space size of 300
        for i, _ in enumerate(valid_actions):
            if i < 300:
                mask[0, i] = 1.0
        
        return action, action_idx, log_prob, value, mask


def train_agent(
    agent: RLAgent,
    env_or_game: Any,
    config: Optional[TrainingConfig] = None,
    self_play: bool = False
) -> Dict[str, Any]:
    """
    Train an agent using the appropriate trainer.
    
    Args:
        agent: RL agent to train
        env_or_game: Environment or game class
        config: Training configuration
        self_play: Whether to use self-play training
        
    Returns:
        Dictionary of training statistics
    """
    if self_play:
        # Create self-play trainer
        trainer = SelfPlayTrainer(agent, env_or_game, config)
    else:
        # Create standard trainer
        trainer = Trainer(agent, env_or_game, config)
    
    # Train agent
    stats = trainer.train()
    
    return stats


def evaluate_agent(
    agent: RLAgent,
    env_or_game: Any,
    num_episodes: int = 10,
    deterministic: bool = True
) -> Dict[str, float]:
    """
    Evaluate an agent's performance.
    
    Args:
        agent: RL agent to evaluate
        env_or_game: Environment or game
        num_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        
    Returns:
        Dictionary of evaluation statistics
    """
    # Create trainer
    trainer = Trainer(agent, env_or_game)
    
    # Evaluate agent
    stats = trainer.evaluate(num_episodes, deterministic)
    
    return stats


def compute_returns(
    rewards: List[float],
    dones: List[bool],
    last_value: float = 0.0,
    gamma: float = 0.99
) -> List[float]:
    """
    Compute returns (discounted sum of future rewards).
    
    Args:
        rewards: List of rewards
        dones: List of done flags
        last_value: Value estimate for the final state
        gamma: Discount factor
        
    Returns:
        List of returns
    """
    returns = []
    next_return = last_value
    
    for i in reversed(range(len(rewards))):
        next_return = rewards[i] + gamma * next_return * (1 - dones[i])
        returns.insert(0, next_return)
    
    return returns


def compute_advantages(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float = 0.0,
    gamma: float = 0.99,
    lambda_gae: float = 0.95
) -> List[float]:
    """
    Compute advantages using Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        last_value: Value estimate for the final state
        gamma: Discount factor
        lambda_gae: Lambda parameter for GAE
        
    Returns:
        List of advantages
    """
    advantages = []
    gae = 0
    next_value = last_value
    
    for i in reversed(range(len(rewards))):
        # Calculate TD error
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        
        # Update GAE
        gae = delta + gamma * lambda_gae * (1 - dones[i]) * gae
        
        # Add advantage
        advantages.insert(0, gae)
        
        # Update next value
        next_value = values[i]
    
    return advantages


def save_model(
    agent: RLAgent,
    path: Union[str, Path],
    optimizer: Optional[Any] = None,
    extra_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a model to disk.
    
    Args:
        agent: RL agent to save
        path: Path to save the model
        optimizer: Optional optimizer to save
        extra_data: Optional additional data to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save agent
    agent.save(path)
    
    # Save additional data if provided
    if extra_data is not None:
        with open(f"{path}_extra.json", 'w') as f:
            json.dump(extra_data, f)


def load_model(
    agent: RLAgent,
    path: Union[str, Path],
    optimizer: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Load a model from disk.
    
    Args:
        agent: RL agent to load into
        path: Path to load the model from
        optimizer: Optional optimizer to load into
        
    Returns:
        Dictionary of additional data if available
    """
    # Load agent
    agent.load(path)
    
    # Load additional data if available
    extra_data = {}
    try:
        with open(f"{path}_extra.json", 'r') as f:
            extra_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    return extra_data


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
