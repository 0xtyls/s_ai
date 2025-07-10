"""
Configuration classes for reinforcement learning algorithms.

This module provides dataclasses for configuring different RL algorithms
and training processes. Each configuration class includes validation and
sensible defaults.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple, Callable, Literal
import math
import json
import os
from pathlib import Path


@dataclass
class RLConfig:
    """
    Base configuration for all reinforcement learning algorithms.
    
    This class defines common parameters used across different RL algorithms.
    """
    # General parameters
    gamma: float = 0.99
    """Discount factor for future rewards"""
    
    lambda_gae: float = 0.95
    """Lambda parameter for Generalized Advantage Estimation"""
    
    normalize_advantages: bool = True
    """Whether to normalize advantages"""
    
    clip_grad_norm: float = 0.5
    """Maximum norm for gradient clipping"""
    
    entropy_coef: float = 0.01
    """Coefficient for entropy regularization"""
    
    value_loss_coef: float = 0.5
    """Coefficient for value function loss"""
    
    max_grad_norm: float = 0.5
    """Maximum gradient norm for clipping"""
    
    # Device and parallelization
    device: str = "cpu"
    """Device to run the model on (cpu or cuda)"""
    
    num_workers: int = 1
    """Number of parallel workers for data collection"""
    
    # Logging and checkpointing
    log_interval: int = 10
    """How often to log training statistics (in updates)"""
    
    save_interval: int = 100
    """How often to save model checkpoints (in updates)"""
    
    eval_interval: int = 50
    """How often to evaluate the model (in updates)"""
    
    # Exploration parameters
    exploration_initial: float = 1.0
    """Initial exploration rate"""
    
    exploration_final: float = 0.1
    """Final exploration rate"""
    
    exploration_decay: float = 0.99
    """Decay rate for exploration"""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.gamma <= 0 or self.gamma > 1:
            raise ValueError("gamma must be in (0, 1]")
        
        if self.lambda_gae <= 0 or self.lambda_gae > 1:
            raise ValueError("lambda_gae must be in (0, 1]")
        
        if self.entropy_coef < 0:
            raise ValueError("entropy_coef must be non-negative")
        
        if self.value_loss_coef < 0:
            raise ValueError("value_loss_coef must be non-negative")
        
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        
        if self.device not in ["cpu", "cuda"]:
            if not self.device.startswith("cuda:"):
                raise ValueError("device must be 'cpu', 'cuda', or 'cuda:n'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RLConfig':
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'RLConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class NetworkConfig:
    """
    Configuration for neural network architecture.
    
    This class defines parameters for the policy and value networks.
    """
    # Network architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    """Sizes of hidden layers"""
    
    activation: str = "relu"
    """Activation function (relu, tanh, leaky_relu)"""
    
    use_batch_norm: bool = False
    """Whether to use batch normalization"""
    
    dropout_rate: float = 0.0
    """Dropout rate (0 = no dropout)"""
    
    # Initialization
    init_type: str = "orthogonal"
    """Weight initialization method (orthogonal, xavier, kaiming)"""
    
    gain: float = math.sqrt(2)
    """Gain factor for weight initialization"""
    
    # Advanced options
    use_layer_norm: bool = False
    """Whether to use layer normalization"""
    
    use_residual: bool = False
    """Whether to use residual connections"""
    
    shared_backbone: bool = True
    """Whether policy and value networks share backbone"""
    
    # Specific to board games
    use_transformer: bool = False
    """Whether to use transformer architecture"""
    
    transformer_heads: int = 4
    """Number of attention heads for transformer"""
    
    transformer_layers: int = 2
    """Number of transformer layers"""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")
        
        if self.activation not in ["relu", "tanh", "leaky_relu", "elu", "selu"]:
            raise ValueError("activation must be one of: relu, tanh, leaky_relu, elu, selu")
        
        if self.dropout_rate < 0 or self.dropout_rate >= 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        
        if self.init_type not in ["orthogonal", "xavier", "kaiming", "normal", "uniform"]:
            raise ValueError("init_type must be one of: orthogonal, xavier, kaiming, normal, uniform")


@dataclass
class PPOConfig(RLConfig):
    """
    Configuration for Proximal Policy Optimization (PPO) algorithm.
    
    This class extends the base RLConfig with PPO-specific parameters.
    """
    # PPO-specific parameters
    clip_ratio: float = 0.2
    """Clip ratio for PPO objective"""
    
    epochs: int = 4
    """Number of epochs to optimize on the same data"""
    
    batch_size: int = 64
    """Batch size for optimization"""
    
    learning_rate: float = 3e-4
    """Learning rate"""
    
    lr_schedule: str = "constant"
    """Learning rate schedule (constant, linear, exponential, cosine)"""
    
    target_kl: Optional[float] = 0.01
    """Target KL divergence (None = no early stopping)"""
    
    value_clip: bool = True
    """Whether to clip value function updates"""
    
    value_clip_ratio: float = 0.2
    """Clip ratio for value function"""
    
    # Network configuration
    network: NetworkConfig = field(default_factory=NetworkConfig)
    """Neural network configuration"""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        super().__post_init__()
        
        if self.clip_ratio <= 0:
            raise ValueError("clip_ratio must be positive")
        
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.lr_schedule not in ["constant", "linear", "exponential", "cosine"]:
            raise ValueError("lr_schedule must be one of: constant, linear, exponential, cosine")
        
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be positive or None")


@dataclass
class A2CConfig(RLConfig):
    """
    Configuration for Advantage Actor-Critic (A2C) algorithm.
    
    This class extends the base RLConfig with A2C-specific parameters.
    """
    # A2C-specific parameters
    learning_rate: float = 7e-4
    """Learning rate"""
    
    lr_schedule: str = "constant"
    """Learning rate schedule (constant, linear, exponential, cosine)"""
    
    normalize_returns: bool = True
    """Whether to normalize returns"""
    
    use_gae: bool = True
    """Whether to use Generalized Advantage Estimation"""
    
    # Network configuration
    network: NetworkConfig = field(default_factory=NetworkConfig)
    """Neural network configuration"""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        super().__post_init__()
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.lr_schedule not in ["constant", "linear", "exponential", "cosine"]:
            raise ValueError("lr_schedule must be one of: constant, linear, exponential, cosine")


@dataclass
class TrainingConfig:
    """
    Configuration for the training process.
    
    This class defines parameters for the overall training process,
    including data collection, evaluation, and logging.
    """
    # Training parameters
    total_timesteps: int = 1_000_000
    """Total number of timesteps to train for"""
    
    num_envs: int = 8
    """Number of parallel environments"""
    
    steps_per_update: int = 128
    """Number of steps to collect before each update"""
    
    max_episode_length: int = 500
    """Maximum length of an episode"""
    
    # Self-play parameters
    self_play: bool = True
    """Whether to use self-play for training"""
    
    opponent_sampling: str = "latest"
    """How to sample opponents (latest, random, elo)"""
    
    num_opponents: int = 5
    """Number of past versions to keep as opponents"""
    
    update_opponent_every: int = 10
    """How often to update the opponent pool (in updates)"""
    
    # Evaluation parameters
    eval_episodes: int = 10
    """Number of episodes for evaluation"""
    
    eval_deterministic: bool = True
    """Whether to use deterministic actions for evaluation"""
    
    # Logging and visualization
    log_dir: str = "runs"
    """Directory for logs"""
    
    use_tensorboard: bool = True
    """Whether to use TensorBoard for logging"""
    
    save_video: bool = False
    """Whether to save videos of episodes"""
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    """Directory for checkpoints"""
    
    keep_checkpoints: int = 5
    """Number of checkpoints to keep"""
    
    # Reproducibility
    seed: Optional[int] = None
    """Random seed for reproducibility"""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        
        if self.steps_per_update <= 0:
            raise ValueError("steps_per_update must be positive")
        
        if self.max_episode_length <= 0:
            raise ValueError("max_episode_length must be positive")
        
        if self.opponent_sampling not in ["latest", "random", "elo"]:
            raise ValueError("opponent_sampling must be one of: latest, random, elo")
        
        if self.num_opponents <= 0:
            raise ValueError("num_opponents must be positive")
        
        if self.update_opponent_every <= 0:
            raise ValueError("update_opponent_every must be positive")
        
        if self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive")
        
        if self.keep_checkpoints <= 0:
            raise ValueError("keep_checkpoints must be positive")
    
    def create_directories(self) -> None:
        """Create necessary directories for logs and checkpoints."""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


@dataclass
class SplendorRLConfig:
    """
    Configuration specific to Splendor game reinforcement learning.
    
    This class defines parameters specific to the Splendor game,
    such as state representation and action encoding.
    """
    # State representation
    use_gem_counts: bool = True
    """Whether to include gem counts in state"""
    
    use_card_counts: bool = True
    """Whether to include card counts in state"""
    
    use_noble_counts: bool = True
    """Whether to include noble counts in state"""
    
    use_player_bonuses: bool = True
    """Whether to include player bonuses in state"""
    
    use_opponent_cards: bool = True
    """Whether to include opponent cards in state"""
    
    use_deck_knowledge: bool = False
    """Whether to include knowledge of remaining deck cards"""
    
    # Action space
    simplified_actions: bool = False
    """Whether to use simplified action space"""
    
    mask_invalid_actions: bool = True
    """Whether to mask invalid actions"""
    
    # Reward shaping
    reward_shaping: bool = True
    """Whether to use reward shaping"""
    
    points_reward: float = 1.0
    """Reward for gaining prestige points"""
    
    card_reward: float = 0.1
    """Reward for purchasing cards"""
    
    noble_reward: float = 0.5
    """Reward for acquiring nobles"""
    
    gem_reward: float = 0.01
    """Reward for collecting gems"""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.points_reward < 0:
            raise ValueError("points_reward must be non-negative")
        
        if self.card_reward < 0:
            raise ValueError("card_reward must be non-negative")
        
        if self.noble_reward < 0:
            raise ValueError("noble_reward must be non-negative")
        
        if self.gem_reward < 0:
            raise ValueError("gem_reward must be non-negative")


# Default configurations
DEFAULT_CONFIG = RLConfig()
DEFAULT_PPO_CONFIG = PPOConfig()
DEFAULT_A2C_CONFIG = A2CConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_SPLENDOR_CONFIG = SplendorRLConfig()
