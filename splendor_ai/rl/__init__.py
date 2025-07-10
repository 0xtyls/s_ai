"""
Reinforcement Learning package for Splendor AI.

This package provides reinforcement learning agents and training infrastructure
for the Splendor board game. It includes:

1. Neural network models (policy and value networks)
2. RL algorithms (PPO, A2C)
3. Training utilities (experience collection, replay buffer)
4. Self-play training loop
5. Model saving/loading and evaluation

The RL agents can be used standalone or combined with MCTS for hybrid approaches
similar to AlphaZero.
"""

__version__ = "0.1.0"

# Core RL components
from splendor_ai.rl.agents import (
    RLAgent, PPOAgent, A2CAgent, 
    RandomAgent, HybridAgent
)
from splendor_ai.rl.models import (
    PolicyNetwork, ValueNetwork, 
    SplendorNetwork, BoardGameNetwork
)
from splendor_ai.rl.training import (
    Trainer, SelfPlayTrainer,
    ExperienceCollector, ReplayBuffer,
    train_agent, evaluate_agent,
    set_seed, save_model, load_model,
    compute_returns, compute_advantages
)
from splendor_ai.rl.config import (
    RLConfig, PPOConfig, A2CConfig,
    TrainingConfig, DEFAULT_CONFIG
)

# Default configurations
DEFAULT_PPO_CONFIG = PPOConfig()
DEFAULT_A2C_CONFIG = A2CConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()

# Convenient factory functions
def create_ppo_agent(config=None):
    """Create a PPO agent with optional custom configuration."""
    from splendor_ai.rl.agents import PPOAgent
    return PPOAgent(config or DEFAULT_PPO_CONFIG)

def create_a2c_agent(config=None):
    """Create an A2C agent with optional custom configuration."""
    from splendor_ai.rl.agents import A2CAgent
    return A2CAgent(config or DEFAULT_A2C_CONFIG)

def create_hybrid_agent(rl_agent, mcts_config=None):
    """Create a hybrid agent that combines RL policy with MCTS."""
    from splendor_ai.rl.agents import HybridAgent
    from splendor_ai.mcts.config import MCTSConfig
    return HybridAgent(rl_agent, mcts_config or MCTSConfig())

def create_trainer(agent, env, config=None):
    """Create a trainer for the given agent and environment."""
    from splendor_ai.rl.training import Trainer
    return Trainer(agent, env, config or DEFAULT_TRAINING_CONFIG)

# Version info as a tuple for programmatic access
VERSION_INFO = tuple(map(int, __version__.split('.')))

__all__ = [
    # Agents
    'RLAgent', 'PPOAgent', 'A2CAgent', 'RandomAgent', 'HybridAgent',
    
    # Models
    'PolicyNetwork', 'ValueNetwork', 'SplendorNetwork', 'BoardGameNetwork',
    
    # Training
    'Trainer', 'SelfPlayTrainer', 'ExperienceCollector', 'ReplayBuffer',
    'train_agent', 'evaluate_agent',
    
    # Configuration
    'RLConfig', 'PPOConfig', 'A2CConfig', 'TrainingConfig', 'DEFAULT_CONFIG',
    
    # Utilities
    'set_seed', 'save_model', 'load_model', 'compute_returns', 'compute_advantages',
    
    # Factory functions
    'create_ppo_agent', 'create_a2c_agent', 'create_hybrid_agent', 'create_trainer',
    
    # Default configurations
    'DEFAULT_PPO_CONFIG', 'DEFAULT_A2C_CONFIG', 'DEFAULT_TRAINING_CONFIG',
    
    # Version info
    '__version__', 'VERSION_INFO'
]
