"""
Neural network models for reinforcement learning in Splendor.

This module defines the neural network architectures used for policy and value
functions in reinforcement learning agents. It includes:

1. Base network classes with various initialization options
2. Policy networks for action selection
3. Value networks for state evaluation
4. Combined networks with shared backbones
5. Specialized networks for Splendor game state representation
6. Attention mechanisms for better state understanding
"""
import math
from typing import List, Dict, Tuple, Optional, Union, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from splendor_ai.core.constants import GemColor, CardTier, ALL_GEMS, REGULAR_GEMS
from splendor_ai.core.game import GameState
from splendor_ai.core.actions import Action, ActionType
from splendor_ai.rl.config import NetworkConfig


def init_weights(module: nn.Module, init_type: str = 'orthogonal', gain: float = 1.0) -> None:
    """
    Initialize network weights using various methods.
    
    Args:
        module: The module to initialize
        init_type: Initialization method ('orthogonal', 'xavier', 'kaiming', etc.)
        gain: Gain factor for initialization
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        if init_type == 'orthogonal':
            nn.init.orthogonal_(module.weight.data, gain=gain)
        elif init_type == 'xavier':
            nn.init.xavier_uniform_(module.weight.data, gain=gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
        elif init_type == 'normal':
            nn.init.normal_(module.weight.data, mean=0, std=0.1)
        elif init_type == 'uniform':
            nn.init.uniform_(module.weight.data, -0.1, 0.1)
        
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)


def get_activation(activation: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        activation: Name of activation function
        
    Returns:
        Activation module
    """
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(0.2)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'selu':
        return nn.SELU()
    else:
        raise ValueError(f"Unknown activation: {activation}")


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable architecture.
    
    This is a basic building block for policy and value networks.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
        activation: str = 'relu',
        output_activation: Optional[str] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        init_type: str = 'orthogonal',
        gain: float = math.sqrt(2),
        use_residual: bool = False
    ):
        """
        Initialize the MLP.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_sizes: List of hidden layer sizes
            activation: Activation function
            output_activation: Output activation function (None for no activation)
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            dropout_rate: Dropout rate (0 = no dropout)
            init_type: Weight initialization method
            gain: Gain factor for weight initialization
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.init_type = init_type
        self.gain = gain
        self.use_residual = use_residual
        
        # Build the network
        layers = []
        prev_size = input_dim
        
        for i, size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, size))
            
            # Normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(size))
            
            # Activation
            layers.append(get_activation(activation))
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_dim))
        
        # Output activation
        if output_activation is not None:
            layers.append(get_activation(output_activation))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, init_type, gain))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if self.use_residual and len(self.hidden_sizes) > 1:
            # Simple residual connection for networks with multiple hidden layers
            identity = x
            x = self.model[0:3](x)  # First linear + norm + activation
            
            # Apply middle layers
            for i in range(1, len(self.hidden_sizes)):
                if i == 1 and x.shape == identity.shape:
                    x = x + identity  # Residual connection
                x = self.model[i*3:(i+1)*3](x)  # Next linear + norm + activation
            
            # Output layer
            x = self.model[-1](x)
            
            # Output activation if specified
            if self.output_activation_name is not None:
                x = self.model[-1](x)
            
            return x
        else:
            return self.model(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module for board game state representation.
    
    This allows the network to focus on relevant parts of the game state.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        """
        Initialize the multi-head attention module.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        init_weights(self.q_proj, 'xavier')
        init_weights(self.k_proj, 'xavier')
        init_weights(self.v_proj, 'xavier')
        init_weights(self.out_proj, 'xavier')
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the attention module.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional mask tensor of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """
    Transformer block for board game state representation.
    
    This combines multi-head attention with feed-forward layers.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0
    ):
        """
        Initialize the transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, 'xavier'))
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional mask tensor of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Multi-head attention with residual connection and normalization
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection and normalization
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x


class BoardGameNetwork(nn.Module):
    """
    Base network for board game state representation.
    
    This network can process board game states with varying numbers of
    elements (cards, pieces, etc.) using attention mechanisms.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: NetworkConfig
    ):
        """
        Initialize the board game network.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            config: Network configuration
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Feature extraction backbone
        if config.use_transformer:
            # Transformer-based backbone
            self.embedding = nn.Linear(state_dim, config.hidden_sizes[0])
            
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(
                    embed_dim=config.hidden_sizes[0],
                    num_heads=config.transformer_heads,
                    ff_dim=config.hidden_sizes[0] * 4,
                    dropout=config.dropout_rate
                )
                for _ in range(config.transformer_layers)
            ])
            
            self.pooling = nn.Linear(config.hidden_sizes[0], config.hidden_sizes[0])
            
            backbone_output_dim = config.hidden_sizes[0]
        else:
            # MLP backbone
            self.backbone = MLP(
                input_dim=state_dim,
                output_dim=config.hidden_sizes[-1],
                hidden_sizes=config.hidden_sizes[:-1],
                activation=config.activation,
                use_batch_norm=config.use_batch_norm,
                use_layer_norm=config.use_layer_norm,
                dropout_rate=config.dropout_rate,
                init_type=config.init_type,
                gain=config.gain,
                use_residual=config.use_residual
            )
            
            backbone_output_dim = config.hidden_sizes[-1]
        
        # Policy head
        self.policy_head = nn.Linear(backbone_output_dim, action_dim)
        
        # Value head
        self.value_head = nn.Linear(backbone_output_dim, 1)
        
        # Initialize heads
        init_weights(self.policy_head, config.init_type, 0.01)
        init_weights(self.value_head, config.init_type, 1.0)
    
    def _extract_features(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract features from the input state.
        
        Args:
            x: Input tensor
            mask: Optional attention mask
            
        Returns:
            Feature tensor
        """
        if self.config.use_transformer:
            # Apply transformer backbone
            x = self.embedding(x)
            
            for block in self.transformer_blocks:
                x = block(x, mask)
            
            # Global pooling
            x = torch.mean(x, dim=1)
            x = F.relu(self.pooling(x))
        else:
            # Apply MLP backbone
            x = self.backbone(x)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            mask: Optional action mask
            
        Returns:
            Tuple of (action logits, state value)
        """
        features = self._extract_features(x)
        
        # Policy head
        action_logits = self.policy_head(features)
        
        # Apply action mask if provided
        if mask is not None:
            action_logits = action_logits.masked_fill(mask == 0, -1e9)
        
        # Value head
        state_value = self.value_head(features)
        
        return action_logits, state_value


class PolicyNetwork(nn.Module):
    """
    Policy network for action selection.
    
    This network outputs a probability distribution over actions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: NetworkConfig
    ):
        """
        Initialize the policy network.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            config: Network configuration
        """
        super().__init__()
        
        self.network = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
            output_activation=None,  # No activation for logits
            use_batch_norm=config.use_batch_norm,
            use_layer_norm=config.use_layer_norm,
            dropout_rate=config.dropout_rate,
            init_type=config.init_type,
            gain=config.gain,
            use_residual=config.use_residual
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.distributions.Distribution:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            mask: Optional action mask
            
        Returns:
            Action distribution
        """
        logits = self.network(x)
        
        # Apply action mask if provided
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        
        return Categorical(logits=logits)
    
    def get_action(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        """
        Get an action from the policy.
        
        Args:
            x: Input tensor
            mask: Optional action mask
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, log probability, distribution)
        """
        dist = self.forward(x, mask)
        
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, dist


class ValueNetwork(nn.Module):
    """
    Value network for state evaluation.
    
    This network outputs a scalar value for each state.
    """
    
    def __init__(
        self,
        state_dim: int,
        config: NetworkConfig
    ):
        """
        Initialize the value network.
        
        Args:
            state_dim: State dimension
            config: Network configuration
        """
        super().__init__()
        
        self.network = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
            output_activation=None,  # No activation for value
            use_batch_norm=config.use_batch_norm,
            use_layer_norm=config.use_layer_norm,
            dropout_rate=config.dropout_rate,
            init_type=config.init_type,
            gain=config.gain,
            use_residual=config.use_residual
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            State value
        """
        return self.network(x).squeeze(-1)


class SplendorNetwork(nn.Module):
    """
    Specialized network for Splendor game state representation.
    
    This network processes Splendor game states and outputs action probabilities
    and state values. It handles the specific structure of Splendor game states
    and actions.
    """
    
    def __init__(
        self,
        config: NetworkConfig,
        action_dim: int = 300  # Default: upper bound on possible actions
    ):
        """
        Initialize the Splendor network.
        
        Args:
            config: Network configuration
            action_dim: Action dimension (upper bound on possible actions)
        """
        super().__init__()
        
        # State dimensions
        self.gem_dim = len(ALL_GEMS)  # Gem colors including gold
        self.card_tier_dim = len(CardTier)  # Card tiers
        self.player_dim = 20  # Player features (gems, bonuses, points, etc.)
        self.card_dim = 15  # Card features (cost, points, bonus, etc.)
        self.noble_dim = 10  # Noble features (requirements, points)
        
        # Number of elements
        self.max_players = 4
        self.max_visible_cards = 12  # 4 cards per tier
        self.max_nobles = 5
        
        # Calculate state dimension
        state_dim = (
            self.gem_dim +  # Gem pool
            self.card_tier_dim +  # Deck sizes
            (self.max_players * self.player_dim) +  # Players
            (self.max_visible_cards * self.card_dim) +  # Visible cards
            (self.max_nobles * self.noble_dim)  # Nobles
        )
        
        # Create shared backbone if specified
        if config.shared_backbone:
            self.backbone = BoardGameNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                config=config
            )
        else:
            # Separate policy and value networks
            self.policy_net = PolicyNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                config=config
            )
            
            self.value_net = ValueNetwork(
                state_dim=state_dim,
                config=config
            )
        
        self.shared_backbone = config.shared_backbone
        self.config = config
        self.action_dim = action_dim
    
    def encode_state(self, state: GameState) -> torch.Tensor:
        """
        Encode a Splendor game state as a tensor.
        
        Args:
            state: Splendor game state
            
        Returns:
            Encoded state tensor
        """
        # Initialize state vector with zeros
        state_vector = torch.zeros(1, self.gem_dim + self.card_tier_dim +
                                  (self.max_players * self.player_dim) +
                                  (self.max_visible_cards * self.card_dim) +
                                  (self.max_nobles * self.noble_dim))
        
        # Encode gem pool (normalized by max gems)
        offset = 0
        for i, color in enumerate(ALL_GEMS):
            state_vector[0, offset + i] = state.gem_pool.get(color, 0) / 10.0
        
        offset += self.gem_dim
        
        # Encode deck sizes (normalized by original deck size)
        for i, tier in enumerate(CardTier):
            deck_size = len(state.card_decks.get(tier, []))
            max_size = {CardTier.TIER_1: 40, CardTier.TIER_2: 30, CardTier.TIER_3: 20}.get(tier, 40)
            state_vector[0, offset + i] = deck_size / max_size
        
        offset += self.card_tier_dim
        
        # Encode players
        for i, player in enumerate(state.players):
            if i >= self.max_players:
                break
            
            player_offset = offset + (i * self.player_dim)
            
            # Encode player gems (normalized by max gems)
            for j, color in enumerate(ALL_GEMS):
                state_vector[0, player_offset + j] = player.gems.get(color, 0) / 10.0
            
            player_offset += self.gem_dim
            
            # Encode player bonuses (normalized by max reasonable bonuses)
            for j, color in enumerate(REGULAR_GEMS):
                state_vector[0, player_offset + j] = player.bonuses.get(color, 0) / 10.0
            
            player_offset += len(REGULAR_GEMS)
            
            # Encode player points (normalized by victory points)
            state_vector[0, player_offset] = player.points / 15.0
            
            player_offset += 1
            
            # Encode player card counts (normalized by reasonable max)
            state_vector[0, player_offset] = len(player.cards) / 20.0
            
            player_offset += 1
            
            # Encode player reserved card count (normalized by max)
            state_vector[0, player_offset] = len(player.reserved_cards) / 3.0
            
            player_offset += 1
            
            # Encode player noble count (normalized by max)
            state_vector[0, player_offset] = len(player.nobles) / 5.0
        
        offset += self.max_players * self.player_dim
        
        # Encode visible cards
        card_idx = 0
        for tier in CardTier:
            for card in state.card_tiers.get(tier, []):
                if card_idx >= self.max_visible_cards:
                    break
                
                card_offset = offset + (card_idx * self.card_dim)
                
                # Encode card tier (normalized)
                state_vector[0, card_offset] = (card.tier.value - 1) / 2.0
                
                card_offset += 1
                
                # Encode card points (normalized)
                state_vector[0, card_offset] = card.points / 5.0
                
                card_offset += 1
                
                # Encode card bonus (one-hot)
                bonus_idx = list(REGULAR_GEMS).index(card.bonus)
                state_vector[0, card_offset + bonus_idx] = 1.0
                
                card_offset += len(REGULAR_GEMS)
                
                # Encode card cost (normalized)
                for j, color in enumerate(REGULAR_GEMS):
                    state_vector[0, card_offset + j] = card.cost.get(color, 0) / 7.0
                
                card_idx += 1
        
        offset += self.max_visible_cards * self.card_dim
        
        # Encode nobles
        for i, noble in enumerate(state.nobles):
            if i >= self.max_nobles:
                break
            
            noble_offset = offset + (i * self.noble_dim)
            
            # Encode noble points (normalized)
            state_vector[0, noble_offset] = noble.points / 3.0
            
            noble_offset += 1
            
            # Encode noble requirements (normalized)
            for j, color in enumerate(REGULAR_GEMS):
                state_vector[0, noble_offset + j] = noble.requirements.get(color, 0) / 4.0
        
        return state_vector
    
    def create_action_mask(self, state: GameState) -> torch.Tensor:
        """
        Create a mask for valid actions.
        
        Args:
            state: Splendor game state
            
        Returns:
            Action mask tensor (1 for valid actions, 0 for invalid)
        """
        # Initialize mask with zeros
        mask = torch.zeros(1, self.action_dim)
        
        # Get valid actions
        valid_actions = state.get_valid_actions(state.current_player)
        
        # Encode each valid action
        for action in valid_actions:
            # Simple encoding: each action type gets a range of indices
            if action.action_type == ActionType.TAKE_GEMS:
                # Take gems actions: indices 0-99
                # Encode based on gems taken
                gems_encoding = sum(1 << i for i, color in enumerate(REGULAR_GEMS) if color in action.gems)
                idx = gems_encoding % 100
                mask[0, idx] = 1
            
            elif action.action_type == ActionType.PURCHASE_CARD:
                # Purchase card actions: indices 100-199
                # Encode based on card ID and whether it's reserved
                idx = 100 + (action.card_id % 100)
                mask[0, idx] = 1
            
            elif action.action_type == ActionType.RESERVE_CARD:
                # Reserve card actions: indices 200-299
                # Encode based on card ID or tier
                if action.card_id is not None:
                    idx = 200 + (action.card_id % 100)
                else:
                    idx = 290 + action.tier.value
                mask[0, idx] = 1
        
        return mask
    
    def decode_action(self, action_idx: int, state: GameState) -> Optional[Action]:
        """
        Decode an action index to an Action object.
        
        Args:
            action_idx: Action index
            state: Splendor game state
            
        Returns:
            Action object, or None if invalid
        """
        # Get valid actions
        valid_actions = state.get_valid_actions(state.current_player)
        
        # Map action index to action object
        if 0 <= action_idx < 100:
            # Take gems actions
            for action in valid_actions:
                if action.action_type == ActionType.TAKE_GEMS:
                    gems_encoding = sum(1 << i for i, color in enumerate(REGULAR_GEMS) if color in action.gems)
                    if gems_encoding % 100 == action_idx:
                        return action
        
        elif 100 <= action_idx < 200:
            # Purchase card actions
            card_id_mod = action_idx - 100
            for action in valid_actions:
                if action.action_type == ActionType.PURCHASE_CARD and action.card_id % 100 == card_id_mod:
                    return action
        
        elif 200 <= action_idx < 300:
            # Reserve card actions
            if action_idx >= 290:
                # Reserve from tier
                tier_value = action_idx - 290
                for action in valid_actions:
                    if (action.action_type == ActionType.RESERVE_CARD and 
                        action.card_id is None and 
                        action.tier.value == tier_value):
                        return action
            else:
                # Reserve specific card
                card_id_mod = action_idx - 200
                for action in valid_actions:
                    if (action.action_type == ActionType.RESERVE_CARD and 
                        action.card_id is not None and 
                        action.card_id % 100 == card_id_mod):
                        return action
        
        # If no matching action found, return the first valid action as fallback
        return valid_actions[0] if valid_actions else None
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            mask: Optional action mask
            
        Returns:
            Tuple of (action logits, state value)
        """
        if self.shared_backbone:
            return self.backbone(x, mask)
        else:
            # Get action distribution
            dist = self.policy_net(x, mask)
            
            # Get state value
            value = self.value_net(x)
            
            return dist.logits, value
    
    def get_action_and_value(
        self,
        state: GameState,
        deterministic: bool = False
    ) -> Tuple[Action, torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        """
        Get an action and value from the policy.
        
        Args:
            state: Splendor game state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, log probability, value, distribution)
        """
        # Encode state
        x = self.encode_state(state)
        
        # Create action mask
        mask = self.create_action_mask(state)
        
        # Forward pass
        action_logits, value = self.forward(x, mask)
        
        # Create distribution
        dist = Categorical(logits=action_logits)
        
        # Sample action
        if deterministic:
            action_idx = dist.probs.argmax(dim=-1)
        else:
            action_idx = dist.sample()
        
        # Get log probability
        log_prob = dist.log_prob(action_idx)
        
        # Decode action
        action = self.decode_action(action_idx.item(), state)
        
        return action, log_prob, value, dist


def create_splendor_network(config: NetworkConfig) -> SplendorNetwork:
    """
    Create a Splendor network with the given configuration.
    
    Args:
        config: Network configuration
        
    Returns:
        SplendorNetwork
    """
    return SplendorNetwork(config=config)
