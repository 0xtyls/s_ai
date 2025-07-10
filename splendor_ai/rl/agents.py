"""
Reinforcement Learning agents for Splendor.

This module provides various reinforcement learning agents for playing Splendor:

1. Base RLAgent class defining the common interface
2. PPOAgent implementing Proximal Policy Optimization
3. A2CAgent implementing Advantage Actor-Critic
4. RandomAgent for baseline comparison
5. HybridAgent combining RL policy with MCTS (AlphaZero-style)

Each agent can select actions from game states, update its policy from
collected experiences, and save/load its model.
"""
import os
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from splendor_ai.core.game import GameState, Game, GameResult
from splendor_ai.core.actions import Action, ActionType, get_all_valid_actions
from splendor_ai.core.constants import GemColor, CardTier

from splendor_ai.mcts.agent import MCTSAgent
from splendor_ai.mcts.config import MCTSConfig
from splendor_ai.mcts.search import mcts_search

from splendor_ai.rl.config import RLConfig, PPOConfig, A2CConfig, NetworkConfig
from splendor_ai.rl.models import (
    PolicyNetwork, ValueNetwork, SplendorNetwork, 
    BoardGameNetwork, create_splendor_network
)


class RLAgent(ABC):
    """
    Abstract base class for all reinforcement learning agents.
    
    This class defines the common interface that all RL agents must implement.
    """
    
    @abstractmethod
    def select_action(
        self, 
        state: GameState, 
        player_id: int,
        deterministic: bool = False
    ) -> Action:
        """
        Select an action from the current game state.
        
        Args:
            state: Current game state
            player_id: ID of the player making the decision
            deterministic: Whether to select the most probable action
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent's policy based on collected experiences.
        
        Args:
            batch: Dictionary of experience data (states, actions, rewards, etc.)
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the agent's model and configuration.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the agent's model and configuration.
        
        Args:
            path: Path to load the model from
        """
        pass
    
    def get_action_callback(self) -> Callable[[GameState, int], Action]:
        """
        Get a callback function for selecting actions.
        
        This is useful for registering the agent with a Game object.
        
        Returns:
            Callback function that takes a game state and player ID and returns an action
        """
        return lambda state, player_id: self.select_action(state, player_id)
    
    def register_with_game(self, game: Game, player_id: int) -> None:
        """
        Register this agent with a game.
        
        Args:
            game: Game object
            player_id: ID of the player to register as
        """
        game.register_agent(player_id, self.get_action_callback())


class RandomAgent(RLAgent):
    """
    Agent that selects actions randomly.
    
    This agent serves as a baseline for comparison with more sophisticated agents.
    """
    
    def __init__(self, name: str = "Random Agent"):
        """
        Initialize the random agent.
        
        Args:
            name: Name of the agent
        """
        self.name = name
    
    def select_action(
        self, 
        state: GameState, 
        player_id: int,
        deterministic: bool = False
    ) -> Action:
        """
        Select a random valid action.
        
        Args:
            state: Current game state
            player_id: ID of the player making the decision
            deterministic: Ignored for random agent
            
        Returns:
            Randomly selected action
        """
        valid_actions = state.get_valid_actions(player_id)
        if not valid_actions:
            raise ValueError(f"No valid actions for player {player_id}")
        
        return random.choice(valid_actions)
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        No-op update method (random agent doesn't learn).
        
        Args:
            batch: Dictionary of experience data (ignored)
            
        Returns:
            Empty dictionary
        """
        return {}
    
    def save(self, path: Union[str, Path]) -> None:
        """
        No-op save method (random agent has no model).
        
        Args:
            path: Path to save the model (ignored)
        """
        pass
    
    def load(self, path: Union[str, Path]) -> None:
        """
        No-op load method (random agent has no model).
        
        Args:
            path: Path to load the model from (ignored)
        """
        pass


class PPOAgent(RLAgent):
    """
    Agent implementing Proximal Policy Optimization (PPO).
    
    PPO is a policy gradient method that uses a clipped surrogate objective
    to ensure stable updates. It's one of the most popular and effective
    RL algorithms for a wide range of tasks.
    """
    
    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        name: str = "PPO Agent"
    ):
        """
        Initialize the PPO agent.
        
        Args:
            config: PPO configuration
            state_dim: State dimension (if None, will use SplendorNetwork)
            action_dim: Action dimension (if None, will use SplendorNetwork)
            name: Name of the agent
        """
        self.config = config or PPOConfig()
        self.name = name
        self.device = torch.device(self.config.device)
        
        # Create network
        if state_dim is not None and action_dim is not None:
            # Create generic network
            self.network = BoardGameNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                config=self.config.network
            ).to(self.device)
        else:
            # Create Splendor-specific network
            self.network = create_splendor_network(self.config.network).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Learning rate scheduler
        if self.config.lr_schedule == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=1000
            )
        elif self.config.lr_schedule == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.999
            )
        elif self.config.lr_schedule == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=1000,
                eta_min=1e-5
            )
        else:
            self.scheduler = None
        
        # Training stats
        self.updates = 0
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "explained_variance": []
        }
    
    def select_action(
        self, 
        state: GameState, 
        player_id: int,
        deterministic: bool = False
    ) -> Action:
        """
        Select an action using the policy network.
        
        Args:
            state: Current game state
            player_id: ID of the player making the decision
            deterministic: Whether to select the most probable action
            
        Returns:
            Selected action
        """
        # Check if it's actually our turn
        if state.current_player != player_id:
            raise ValueError(f"Not player {player_id}'s turn")
        
        # Get valid actions
        valid_actions = state.get_valid_actions(player_id)
        if not valid_actions:
            raise ValueError(f"No valid actions for player {player_id}")
        
        # If there's only one valid action, no need to use the network
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        # Use the network to select an action
        with torch.no_grad():
            if isinstance(self.network, SplendorNetwork):
                # Use Splendor-specific network
                action, _, _, _ = self.network.get_action_and_value(
                    state=state,
                    deterministic=deterministic
                )
                return action
            else:
                # Encode state and create action mask
                state_tensor = self._encode_state(state)
                mask = self._create_action_mask(state, valid_actions)
                
                # Forward pass
                action_logits, _ = self.network(state_tensor, mask)
                
                # Create distribution
                dist = Categorical(logits=action_logits)
                
                # Sample action
                if deterministic:
                    action_idx = dist.probs.argmax(dim=-1)
                else:
                    action_idx = dist.sample()
                
                # Map action index to action
                action = self._decode_action(action_idx.item(), valid_actions)
                
                return action
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the policy using PPO.
        
        Args:
            batch: Dictionary of experience data
            
        Returns:
            Dictionary of training metrics
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Unpack batch
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        masks = batch.get("masks")  # Optional
        
        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epochs of optimization
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        total_clip_fraction = 0
        # keep track of how many mini-batches we actually processed this update
        num_minibatches = 0
        
        for _ in range(self.config.epochs):
            # Get mini-batches
            batch_size = self.config.batch_size
            indices = torch.randperm(states.size(0))
            
            for start_idx in range(0, states.size(0), batch_size):
                # Get mini-batch indices
                end_idx = min(start_idx + batch_size, states.size(0))
                mb_indices = indices[start_idx:end_idx]
                
                # Get mini-batch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_masks = masks[mb_indices] if masks is not None else None
                
                # Forward pass
                action_logits, values = self.network(mb_states, mb_masks)
                # Ensure values shape matches returns (B,) not (B,1)
                if values.dim() == 2 and values.size(-1) == 1:
                    values = values.squeeze(-1)
                
                # Create distribution
                dist = Categorical(logits=action_logits)
                
                # Get log probabilities and entropy
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # Calculate ratio and clipped ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio
                )
                
                # Calculate surrogate losses
                surrogate1 = ratio * mb_advantages
                surrogate2 = clipped_ratio * mb_advantages
                
                # Calculate policy loss
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Calculate value loss
                if self.config.value_clip:
                    # Get old values
                    old_values = batch["values"][mb_indices]
                    if old_values.dim() == 2 and old_values.size(-1) == 1:
                        old_values = old_values.squeeze(-1)
                    
                    # Calculate clipped value loss
                    values_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.config.value_clip_ratio,
                        self.config.value_clip_ratio
                    )
                    value_loss1 = F.mse_loss(values, mb_returns)
                    value_loss2 = F.mse_loss(values_clipped, mb_returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    # Standard value loss
                    value_loss = F.mse_loss(values, mb_returns)
                
                # Calculate total loss
                loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss -
                    self.config.entropy_coef * entropy
                )
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                if self.config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # Calculate approximate KL divergence
                with torch.no_grad():
                    log_ratio = new_log_probs - mb_old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    
                    # Calculate clipping fraction
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean().item()
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += approx_kl
                total_clip_fraction += clip_fraction
                num_minibatches += 1
                
                # Early stopping based on KL divergence
                if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                    break
            
            # Early stopping for the whole epoch using averaged KL
            if self.config.target_kl is not None and num_minibatches > 0:
                if (total_kl / num_minibatches) > self.config.target_kl:
                    break
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        # ------------------------------------------------------------------
        # Calculate average metrics
        # Use the actual number of processed mini-batches; fall back to 1 to
        # avoid division-by-zero in edge cases where no update occurred.
        # ------------------------------------------------------------------
        denom = max(1, num_minibatches)
        metrics = {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
            "approx_kl": total_kl / denom,
            "clip_fraction": total_clip_fraction / denom,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }
        
        # Update training stats
        for k, v in metrics.items():
            if k in self.training_stats:
                self.training_stats[k].append(v)
        
        self.updates += 1
        
        return metrics
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the agent's model and configuration.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "updates": self.updates,
            "training_stats": self.training_stats,
            "config": self.config.to_dict()
        }, path)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the agent's model and configuration.
        
        Args:
            path: Path to load the model from
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load training stats
        self.updates = checkpoint["updates"]
        self.training_stats = checkpoint["training_stats"]
        
        # Load config
        if "config" in checkpoint:
            self.config = PPOConfig.from_dict(checkpoint["config"])
    
    def _encode_state(self, state: GameState) -> torch.Tensor:
        """
        Encode a game state as a tensor.
        
        Args:
            state: Game state
            
        Returns:
            Encoded state tensor
        """
        if isinstance(self.network, SplendorNetwork):
            return self.network.encode_state(state)
        else:
            # Simple feature extraction for generic network
            # This is a placeholder - in practice, you would implement
            # a more sophisticated state encoding
            features = []
            
            # Game progress
            features.append(state.turn_count / 30.0)  # Normalize by assumed max turns
            
            # Gem pool
            for color in GemColor:
                features.append(state.gem_pool.get(color, 0) / 10.0)  # Normalize by max gems
            
            # Current player
            for i in range(4):  # Max players
                features.append(1.0 if i == state.current_player else 0.0)
            
            # Players
            for i in range(4):  # Max players
                if i < len(state.players):
                    player = state.players[i]
                    
                    # Gems
                    for color in GemColor:
                        features.append(player.gems.get(color, 0) / 10.0)
                    
                    # Bonuses
                    for color in [c for c in GemColor if c != GemColor.GOLD]:
                        features.append(player.bonuses.get(color, 0) / 10.0)
                    
                    # Points
                    features.append(player.points / 15.0)
                    
                    # Card counts
                    features.append(len(player.cards) / 20.0)
                    features.append(len(player.reserved_cards) / 3.0)
                    features.append(len(player.nobles) / 5.0)
                else:
                    # Padding for non-existent players
                    features.extend([0.0] * (len(GemColor) + len(GemColor) - 1 + 1 + 3))
            
            return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def _create_action_mask(self, state: GameState, valid_actions: List[Action]) -> torch.Tensor:
        """
        Create a mask for valid actions.
        
        Args:
            state: Game state
            valid_actions: List of valid actions
            
        Returns:
            Action mask tensor (1 for valid actions, 0 for invalid)
        """
        if isinstance(self.network, SplendorNetwork):
            return self.network.create_action_mask(state)
        else:
            # Simple action mask for generic network
            # This is a placeholder - in practice, you would implement
            # a more sophisticated action encoding
            action_dim = self.network.action_dim
            mask = torch.zeros(1, action_dim, device=self.device)
            
            # Map each valid action to an index
            for i, action in enumerate(valid_actions):
                if i < action_dim:
                    mask[0, i] = 1
            
            return mask
    
    def _decode_action(self, action_idx: int, valid_actions: List[Action]) -> Action:
        """
        Decode an action index to an Action object.
        
        Args:
            action_idx: Action index
            valid_actions: List of valid actions
            
        Returns:
            Action object
        """
        if isinstance(self.network, SplendorNetwork):
            return self.network.decode_action(action_idx, valid_actions[0].state)
        else:
            # Simple action decoding for generic network
            # This is a placeholder - in practice, you would implement
            # a more sophisticated action decoding
            if 0 <= action_idx < len(valid_actions):
                return valid_actions[action_idx]
            else:
                return valid_actions[0]  # Fallback to first valid action


class A2CAgent(RLAgent):
    """
    Agent implementing Advantage Actor-Critic (A2C).
    
    A2C is a synchronous version of the Asynchronous Advantage Actor-Critic (A3C)
    algorithm. It uses multiple workers to collect experiences and then updates
    the policy synchronously.
    """
    
    def __init__(
        self,
        config: Optional[A2CConfig] = None,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        name: str = "A2C Agent"
    ):
        """
        Initialize the A2C agent.
        
        Args:
            config: A2C configuration
            state_dim: State dimension (if None, will use SplendorNetwork)
            action_dim: Action dimension (if None, will use SplendorNetwork)
            name: Name of the agent
        """
        self.config = config or A2CConfig()
        self.name = name
        self.device = torch.device(self.config.device)
        
        # Create network
        if state_dim is not None and action_dim is not None:
            # Create generic network
            self.network = BoardGameNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                config=self.config.network
            ).to(self.device)
        else:
            # Create Splendor-specific network
            self.network = create_splendor_network(self.config.network).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Learning rate scheduler
        if self.config.lr_schedule == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=1000
            )
        elif self.config.lr_schedule == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.999
            )
        elif self.config.lr_schedule == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=1000,
                eta_min=1e-5
            )
        else:
            self.scheduler = None
        
        # Training stats
        self.updates = 0
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": []
        }
    
    def select_action(
        self, 
        state: GameState, 
        player_id: int,
        deterministic: bool = False
    ) -> Action:
        """
        Select an action using the policy network.
        
        Args:
            state: Current game state
            player_id: ID of the player making the decision
            deterministic: Whether to select the most probable action
            
        Returns:
            Selected action
        """
        # Check if it's actually our turn
        if state.current_player != player_id:
            raise ValueError(f"Not player {player_id}'s turn")
        
        # Get valid actions
        valid_actions = state.get_valid_actions(player_id)
        if not valid_actions:
            raise ValueError(f"No valid actions for player {player_id}")
        
        # If there's only one valid action, no need to use the network
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        # Use the network to select an action
        with torch.no_grad():
            if isinstance(self.network, SplendorNetwork):
                # Use Splendor-specific network
                action, _, _, _ = self.network.get_action_and_value(
                    state=state,
                    deterministic=deterministic
                )
                return action
            else:
                # Encode state and create action mask
                state_tensor = self._encode_state(state)
                mask = self._create_action_mask(state, valid_actions)
                
                # Forward pass
                action_logits, _ = self.network(state_tensor, mask)
                
                # Create distribution
                dist = Categorical(logits=action_logits)
                
                # Sample action
                if deterministic:
                    action_idx = dist.probs.argmax(dim=-1)
                else:
                    action_idx = dist.sample()
                
                # Map action index to action
                action = self._decode_action(action_idx.item(), valid_actions)
                
                return action
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the policy using A2C.
        
        Args:
            batch: Dictionary of experience data
            
        Returns:
            Dictionary of training metrics
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Unpack batch
        states = batch["states"]
        actions = batch["actions"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        masks = batch.get("masks")  # Optional
        
        # Normalize returns
        if self.config.normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        action_logits, values = self.network(states, masks)
        
        # Create distribution
        dist = Categorical(logits=action_logits)
        
        # Get log probabilities and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Calculate policy loss
        policy_loss = -(log_probs * advantages).mean()
        
        # Calculate value loss
        value_loss = F.mse_loss(values, returns)
        
        # Calculate total loss
        loss = (
            policy_loss +
            self.config.value_loss_coef * value_loss -
            self.config.entropy_coef * entropy
        )
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.max_grad_norm
            )
        
        self.optimizer.step()
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Calculate metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }
        
        # Update training stats
        for k, v in metrics.items():
            if k in self.training_stats:
                self.training_stats[k].append(v)
        
        self.updates += 1
        
        return metrics
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the agent's model and configuration.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "updates": self.updates,
            "training_stats": self.training_stats,
            "config": self.config.to_dict()
        }, path)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the agent's model and configuration.
        
        Args:
            path: Path to load the model from
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load training stats
        self.updates = checkpoint["updates"]
        self.training_stats = checkpoint["training_stats"]
        
        # Load config
        if "config" in checkpoint:
            self.config = A2CConfig.from_dict(checkpoint["config"])
    
    def _encode_state(self, state: GameState) -> torch.Tensor:
        """
        Encode a game state as a tensor.
        
        Args:
            state: Game state
            
        Returns:
            Encoded state tensor
        """
        if isinstance(self.network, SplendorNetwork):
            return self.network.encode_state(state)
        else:
            # Simple feature extraction for generic network
            # This is a placeholder - in practice, you would implement
            # a more sophisticated state encoding
            features = []
            
            # Game progress
            features.append(state.turn_count / 30.0)  # Normalize by assumed max turns
            
            # Gem pool
            for color in GemColor:
                features.append(state.gem_pool.get(color, 0) / 10.0)  # Normalize by max gems
            
            # Current player
            for i in range(4):  # Max players
                features.append(1.0 if i == state.current_player else 0.0)
            
            # Players
            for i in range(4):  # Max players
                if i < len(state.players):
                    player = state.players[i]
                    
                    # Gems
                    for color in GemColor:
                        features.append(player.gems.get(color, 0) / 10.0)
                    
                    # Bonuses
                    for color in [c for c in GemColor if c != GemColor.GOLD]:
                        features.append(player.bonuses.get(color, 0) / 10.0)
                    
                    # Points
                    features.append(player.points / 15.0)
                    
                    # Card counts
                    features.append(len(player.cards) / 20.0)
                    features.append(len(player.reserved_cards) / 3.0)
                    features.append(len(player.nobles) / 5.0)
                else:
                    # Padding for non-existent players
                    features.extend([0.0] * (len(GemColor) + len(GemColor) - 1 + 1 + 3))
            
            return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def _create_action_mask(self, state: GameState, valid_actions: List[Action]) -> torch.Tensor:
        """
        Create a mask for valid actions.
        
        Args:
            state: Game state
            valid_actions: List of valid actions
            
        Returns:
            Action mask tensor (1 for valid actions, 0 for invalid)
        """
        if isinstance(self.network, SplendorNetwork):
            return self.network.create_action_mask(state)
        else:
            # Simple action mask for generic network
            # This is a placeholder - in practice, you would implement
            # a more sophisticated action encoding
            action_dim = self.network.action_dim
            mask = torch.zeros(1, action_dim, device=self.device)
            
            # Map each valid action to an index
            for i, action in enumerate(valid_actions):
                if i < action_dim:
                    mask[0, i] = 1
            
            return mask
    
    def _decode_action(self, action_idx: int, valid_actions: List[Action]) -> Action:
        """
        Decode an action index to an Action object.
        
        Args:
            action_idx: Action index
            valid_actions: List of valid actions
            
        Returns:
            Action object
        """
        if isinstance(self.network, SplendorNetwork):
            return self.network.decode_action(action_idx, valid_actions[0].state)
        else:
            # Simple action decoding for generic network
            # This is a placeholder - in practice, you would implement
            # a more sophisticated action decoding
            if 0 <= action_idx < len(valid_actions):
                return valid_actions[action_idx]
            else:
                return valid_actions[0]  # Fallback to first valid action


class HybridAgent(RLAgent):
    """
    Hybrid agent combining RL policy with MCTS (AlphaZero-style).
    
    This agent uses a trained policy network to guide MCTS exploration
    and a value network to replace random rollouts. This approach was
    popularized by AlphaGo and AlphaZero.
    """
    
    def __init__(
        self,
        rl_agent: RLAgent,
        mcts_config: Optional[MCTSConfig] = None,
        name: str = "Hybrid Agent"
    ):
        """
        Initialize the hybrid agent.
        
        Args:
            rl_agent: Reinforcement learning agent (policy provider)
            mcts_config: MCTS configuration
            name: Name of the agent
        """
        self.rl_agent = rl_agent
        self.mcts_config = mcts_config or MCTSConfig()
        self.name = name
        
        # Ensure the RL agent has a network
        if not hasattr(self.rl_agent, "network"):
            raise ValueError("RL agent must have a network attribute")
        
        # Set device
        if hasattr(self.rl_agent, "device"):
            self.device = self.rl_agent.device
        else:
            self.device = torch.device("cpu")
    
    def select_action(
        self, 
        state: GameState, 
        player_id: int,
        deterministic: bool = False
    ) -> Action:
        """
        Select an action using MCTS guided by the policy network.
        
        Args:
            state: Current game state
            player_id: ID of the player making the decision
            deterministic: Whether to use deterministic MCTS (less exploration)
            
        Returns:
            Selected action
        """
        # Check if it's actually our turn
        if state.current_player != player_id:
            raise ValueError(f"Not player {player_id}'s turn")
        
        # Get valid actions
        valid_actions = state.get_valid_actions(player_id)
        if not valid_actions:
            raise ValueError(f"No valid actions for player {player_id}")
        
        # If there's only one valid action, no need to use MCTS
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        # Create a temporary MCTS config with adjusted exploration
        mcts_config = MCTSConfig(**self.mcts_config.to_dict())
        if deterministic:
            mcts_config.exploration_weight = 0.1  # Less exploration
        
        # Run MCTS with policy guidance
        best_action, _ = mcts_search(
            state=state,
            player_id=player_id,
            config=mcts_config,
            policy_fn=self._get_policy_probs,
            value_fn=self._get_state_value
        )
        
        return best_action
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the underlying RL agent.
        
        Args:
            batch: Dictionary of experience data
            
        Returns:
            Dictionary of training metrics
        """
        return self.rl_agent.update(batch)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the agent's model and configuration.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save RL agent
        rl_path = f"{path}_rl"
        self.rl_agent.save(rl_path)
        
        # Save hybrid configuration
        torch.save({
            "mcts_config": self.mcts_config.to_dict(),
            "rl_path": rl_path,
            "name": self.name
        }, path)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the agent's model and configuration.
        
        Args:
            path: Path to load the model from
        """
        # Load hybrid configuration
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load RL agent
        rl_path = checkpoint["rl_path"]
        self.rl_agent.load(rl_path)
        
        # Load MCTS config
        if "mcts_config" in checkpoint:
            self.mcts_config = MCTSConfig.from_dict(checkpoint["mcts_config"])
        
        # Load name
        if "name" in checkpoint:
            self.name = checkpoint["name"]
    
    def _get_policy_probs(
        self,
        state: GameState,
        valid_actions: List[Action]
    ) -> List[float]:
        """
        Get action probabilities from the policy network.
        
        Args:
            state: Game state
            valid_actions: List of valid actions
            
        Returns:
            List of action probabilities
        """
        with torch.no_grad():
            # Encode state
            if isinstance(self.rl_agent.network, SplendorNetwork):
                state_tensor = self.rl_agent.network.encode_state(state)
                mask = self.rl_agent.network.create_action_mask(state)
            else:
                state_tensor = self.rl_agent._encode_state(state)
                mask = self.rl_agent._create_action_mask(state, valid_actions)
            
            # Forward pass
            action_logits, _ = self.rl_agent.network(state_tensor, mask)
            
            # Convert to probabilities
            probs = F.softmax(action_logits, dim=-1).cpu().numpy()[0]
            
            # Map probabilities to valid actions
            action_probs = []
            for action in valid_actions:
                if isinstance(self.rl_agent.network, SplendorNetwork):
                    # Use SplendorNetwork's action encoding
                    action_idx = self._get_action_idx(action)
                    if 0 <= action_idx < len(probs):
                        action_probs.append(probs[action_idx])
                    else:
                        action_probs.append(1.0 / len(valid_actions))  # Uniform fallback
                else:
                    # Simple mapping for generic network
                    action_idx = valid_actions.index(action)
                    if action_idx < len(probs):
                        action_probs.append(probs[action_idx])
                    else:
                        action_probs.append(1.0 / len(valid_actions))  # Uniform fallback
            
            # Normalize probabilities
            total = sum(action_probs)
            if total > 0:
                action_probs = [p / total for p in action_probs]
            else:
                action_probs = [1.0 / len(valid_actions) for _ in valid_actions]
            
            return action_probs
    
    def _get_state_value(self, state: GameState, player_id: int) -> float:
        """
        Get state value from the value network.
        
        Args:
            state: Game state
            player_id: ID of the player
            
        Returns:
            State value
        """
        with torch.no_grad():
            # Encode state
            if isinstance(self.rl_agent.network, SplendorNetwork):
                state_tensor = self.rl_agent.network.encode_state(state)
            else:
                state_tensor = self.rl_agent._encode_state(state)
            
            # Forward pass
            _, value = self.rl_agent.network(state_tensor)
            
            # Convert to float
            return value.item()
    
    def _get_action_idx(self, action: Action) -> int:
        """
        Get the index of an action in the policy network's output.
        
        Args:
            action: Action
            
        Returns:
            Action index
        """
        # This is a simplified version - in practice, you would implement
        # a more sophisticated action encoding
        if action.action_type == ActionType.TAKE_GEMS:
            # Take gems actions: indices 0-99
            # Encode based on gems taken
            gems_encoding = sum(1 << i for i, color in enumerate(
                [c for c in GemColor if c != GemColor.GOLD]
            ) if color in action.gems)
            return gems_encoding % 100
        
        elif action.action_type == ActionType.PURCHASE_CARD:
            # Purchase card actions: indices 100-199
            # Encode based on card ID
            return 100 + (action.card_id % 100)
        
        elif action.action_type == ActionType.RESERVE_CARD:
            # Reserve card actions: indices 200-299
            # Encode based on card ID or tier
            if action.card_id is not None:
                return 200 + (action.card_id % 100)
            else:
                return 290 + action.tier.value
        
        # Fallback
        return 0
