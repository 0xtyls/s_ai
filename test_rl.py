#!/usr/bin/env python
"""
Test script for Splendor AI reinforcement learning components.

This script tests the core functionality of the RL system:
1. Agent creation (PPO, A2C, Random, Hybrid)
2. Basic training loop functionality
3. Experience collection and policy updates
4. Evaluation against baseline agents

This is a lightweight test to verify system integration, not performance.
"""
import os
import time
import random
import unittest
from pathlib import Path

import numpy as np
import torch

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


class TestRLComponents(unittest.TestCase):
    """Test case for RL components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        set_seed(42)
        
        # Create temporary directory for saving models
        self.temp_dir = Path("temp_test_rl")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Create network config
        # minimal config – only parameters that actually exist in NetworkConfig
        self.network_config = NetworkConfig(
            hidden_sizes=[64, 64]  # Small for testing (2 layers of 64 units)
        )
        
        # Create PPO config
        self.ppo_config = PPOConfig(
            learning_rate=3e-4,
            gamma=0.99,
            lambda_gae=0.95,
            clip_ratio=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            epochs=3,  # Small for testing
            batch_size=32,
            network=self.network_config,
            device="cpu"
        )
        
        # Create A2C config
        self.a2c_config = A2CConfig(
            learning_rate=3e-4,
            gamma=0.99,
            lambda_gae=0.95,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            network=self.network_config,
            device="cpu"
        )
        
        # Create training config
        self.training_config = TrainingConfig(
            total_timesteps=100,  # Small for testing
            steps_per_update=32,
            eval_episodes=2,
            checkpoint_dir=str(self.temp_dir),
            log_dir=str(self.temp_dir),
            use_tensorboard=False,
            seed=42,
            num_opponents=2,
            update_opponent_every=1,
            opponent_sampling="random",
            max_episode_length=20  # Small for testing
        )
        
        # Create game
        self.game = Game(num_players=2)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_agent_creation(self):
        """Test agent creation."""
        # Create PPO agent
        ppo_agent = PPOAgent(config=self.ppo_config, name="PPO Test")
        self.assertIsInstance(ppo_agent, PPOAgent)
        self.assertEqual(ppo_agent.name, "PPO Test")
        
        # Create A2C agent
        a2c_agent = A2CAgent(config=self.a2c_config, name="A2C Test")
        self.assertIsInstance(a2c_agent, A2CAgent)
        self.assertEqual(a2c_agent.name, "A2C Test")
        
        # Create Random agent
        random_agent = RandomAgent(name="Random Test")
        self.assertIsInstance(random_agent, RandomAgent)
        self.assertEqual(random_agent.name, "Random Test")
        
        # Create Hybrid agent
        mcts_config = MCTSConfig(num_simulations=10)  # Small for testing
        hybrid_agent = HybridAgent(rl_agent=ppo_agent, mcts_config=mcts_config, name="Hybrid Test")
        self.assertIsInstance(hybrid_agent, HybridAgent)
        self.assertEqual(hybrid_agent.name, "Hybrid Test")
    
    def test_experience_collector(self):
        """Test experience collector."""
        # Create collector
        collector = ExperienceCollector(gamma=0.99, lambda_gae=0.95)
        
        # Add experiences
        for i in range(5):
            state = torch.rand(1, 10)  # Dummy state
            action = i
            action_idx = i
            reward = float(i)
            value = float(i) * 0.5
            log_prob = -float(i) * 0.1
            done = i == 4  # Last step is done
            
            collector.add_experience(
                state=state,
                action=action,
                action_idx=action_idx,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done
            )
        
        # Check length
        self.assertEqual(len(collector), 5)
        
        # Compute returns and advantages
        returns, advantages = collector.compute_returns_and_advantages(last_value=0.0)
        
        # Check returns and advantages
        self.assertEqual(len(returns), 5)
        self.assertEqual(len(advantages), 5)
        
        # Get batch
        batch = collector.get_batch(last_value=0.0)
        
        # Check batch
        self.assertIn("states", batch)
        self.assertIn("actions", batch)
        self.assertIn("log_probs", batch)
        self.assertIn("returns", batch)
        self.assertIn("advantages", batch)
        self.assertIn("values", batch)
    
    def test_replay_buffer(self):
        """Test replay buffer."""
        # Create buffer
        buffer = ReplayBuffer(capacity=100)
        
        # Add experiences
        for i in range(5):
            experience = {
                "states": torch.rand(1, 10),
                "actions": torch.tensor([i]),
                "rewards": torch.tensor([float(i)]),
                "dones": torch.tensor([i == 4])
            }
            
            buffer.add(experience)
        
        # Check length
        self.assertEqual(len(buffer), 5)
        
        # Sample batch
        batch = buffer.sample(batch_size=3)
        
        # Check batch
        self.assertIn("states", batch)
        self.assertIn("actions", batch)
        self.assertIn("rewards", batch)
        self.assertIn("dones", batch)
        self.assertEqual(batch["actions"].shape[0], 3)
    
    def test_agent_select_action(self):
        """Test agent action selection."""
        # Create agents
        ppo_agent = PPOAgent(config=self.ppo_config, name="PPO Test")
        random_agent = RandomAgent(name="Random Test")
        
        # Reset game
        self.game.reset()
        
        # Get state
        state = self.game.state
        
        # Select actions
        ppo_action = ppo_agent.select_action(state, player_id=0)
        random_action = random_agent.select_action(state, player_id=0)
        
        # Check actions
        self.assertIsInstance(ppo_action, Action)
        self.assertIsInstance(random_action, Action)
        
        # Check actions are valid
        valid_actions = state.get_valid_actions(0)
        self.assertIn(ppo_action, valid_actions)
        self.assertIn(random_action, valid_actions)
    
    def test_agent_save_load(self):
        """Test agent save and load."""
        # Create agent
        ppo_agent = PPOAgent(config=self.ppo_config, name="PPO Test")
        
        # Save agent
        save_path = self.temp_dir / "ppo_test.pt"
        ppo_agent.save(save_path)
        
        # Check file exists
        self.assertTrue(save_path.exists())
        
        # Create new agent
        new_agent = PPOAgent(config=self.ppo_config, name="PPO New")
        
        # Load agent
        new_agent.load(save_path)
        
        # Check name (should not be overwritten)
        self.assertEqual(new_agent.name, "PPO New")
        
        # Check both agents have same network parameters
        for p1, p2 in zip(ppo_agent.network.parameters(), new_agent.network.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
    
    def test_basic_training(self):
        """Test basic training loop."""
        # Create agent
        ppo_agent = PPOAgent(config=self.ppo_config, name="PPO Test")
        
        # Register agent with game
        self.game.register_agent(0, ppo_agent.get_action_callback())
        
        # Register random opponent
        random_agent = RandomAgent(name="Random Opponent")
        self.game.register_agent(1, random_agent.get_action_callback())
        
        # Create trainer
        trainer = Trainer(
            agent=ppo_agent,
            env=self.game,
            config=self.training_config
        )
        
        # Collect experience
        stats = trainer.collect_experience(num_steps=10)
        
        # Check stats
        self.assertIn("mean_reward", stats)
        self.assertIn("mean_length", stats)
        self.assertIn("num_episodes", stats)
        
        # Check experience collector
        self.assertGreater(len(trainer.collector), 0)
        
        # Update policy
        metrics = trainer.update_policy()
        
        # Check metrics
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertIn("entropy", metrics)
    
    def test_self_play_training(self):
        """Test self-play training."""
        # Create agent
        ppo_agent = PPOAgent(config=self.ppo_config, name="PPO Test")
        
        # Create self-play trainer
        trainer = SelfPlayTrainer(
            agent=ppo_agent,
            game_class=Game,
            config=self.training_config,
            game_kwargs={"num_players": 2}
        )
        
        # Collect experience
        stats = trainer.collect_experience(num_steps=10)
        
        # Check stats
        self.assertIn("mean_reward", stats)
        self.assertIn("mean_length", stats)
        self.assertIn("win_rate", stats)
        self.assertIn("num_episodes", stats)
        
        # Check experience collector
        self.assertGreater(len(trainer.collector), 0)
        
        # Update policy
        metrics = trainer.update_policy()
        
        # Check metrics
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertIn("entropy", metrics)
        
        # Update opponent pool
        trainer.update_opponent_pool()
        
        # Check opponent pool
        self.assertGreaterEqual(len(trainer.opponent_pool), 2)  # Random + new opponent
    
    def test_evaluation(self):
        """Test agent evaluation."""
        # Create agent
        ppo_agent = PPOAgent(config=self.ppo_config, name="PPO Test")
        
        # Register agent with game
        self.game.register_agent(0, ppo_agent.get_action_callback())
        
        # Register random opponent
        random_agent = RandomAgent(name="Random Opponent")
        self.game.register_agent(1, random_agent.get_action_callback())
        
        # Create trainer
        trainer = Trainer(
            agent=ppo_agent,
            env=self.game,
            config=self.training_config
        )
        
        # Evaluate agent
        stats = trainer.evaluate(num_episodes=2)
        
        # Check stats
        self.assertIn("mean_reward", stats)
        self.assertIn("mean_length", stats)
        self.assertIn("win_rate", stats)
        self.assertIn("num_episodes", stats)


def run_quick_test():
    """Run a quick test without unittest framework."""
    print("Running quick RL system test...")
    
    # Set random seed
    set_seed(42)
    
    # Create PPO agent
    network_config = NetworkConfig(hidden_sizes=[64, 64])
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        epochs=3,
        batch_size=32,
        network=network_config
    )
    # Create PPO agent with a GENERIC network suitable for 10-dim states
    # and a small action space for this synthetic test.
    agent = PPOAgent(config=ppo_config, state_dim=10, action_dim=8)
    
    print("Creating experience collector...")
    # Create experience collector
    collector = ExperienceCollector()
    
    # Add dummy experiences
    for i in range(5):
        collector.add_experience(
            state=torch.rand(1, 10),
            action=i,
            action_idx=i,
            reward=float(i),
            value=float(i) * 0.5,
            log_prob=-float(i) * 0.1,
            done=i == 4
        )
    
    # Compute returns and advantages
    returns, advantages = collector.compute_returns_and_advantages()
    
    # Get batch
    batch = collector.get_batch()
    
    print("Updating policy...")
    # Update policy
    metrics = agent.update(batch)
    
    print("Metrics:", metrics)
    print("Test completed successfully!")
    return True


if __name__ == "__main__":
    # Run quick test
    success = run_quick_test()
    
    if success:
        print("\nRunning full unittest suite...")
        unittest.main()
    else:
        print("Quick test failed, skipping unittest suite.")
