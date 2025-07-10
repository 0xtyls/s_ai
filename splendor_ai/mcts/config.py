"""
Configuration for Monte Carlo Tree Search (MCTS).

This module defines the configuration parameters for the MCTS algorithm,
including search depth, exploration constants, and simulation strategies.
"""
from dataclasses import dataclass, field
from typing import Optional, Union, Literal, ClassVar
import math


@dataclass
class MCTSConfig:
    """
    Configuration parameters for Monte Carlo Tree Search.
    
    This class defines all tunable parameters for the MCTS algorithm,
    with validation and sensible defaults.
    """
    # Search parameters
    iterations: int = 1000
    """Number of MCTS iterations to perform per move decision"""
    
    exploration_weight: float = math.sqrt(2)
    """UCB1 exploration parameter (default is sqrt(2))"""
    
    max_depth: int = 100
    """Maximum depth for simulations"""
    
    time_limit: Optional[float] = None
    """Optional time limit in seconds (None = no limit)"""
    
    # Strategy parameters
    use_heuristics: bool = True
    """Whether to use domain-specific heuristics during selection and expansion"""
    
    simulation_policy: Literal["random", "heuristic"] = "random"
    """Policy for the simulation phase ('random' or 'heuristic')"""
    
    # Advanced parameters
    progressive_widening: bool = False
    """Whether to use progressive widening to handle large branching factors"""
    
    widening_factor: float = 0.5
    """Factor for progressive widening (if enabled)"""
    
    use_transposition_table: bool = True
    """Whether to use a transposition table to avoid redundant searches"""
    
    transposition_table_size: int = 100000
    """Maximum size of the transposition table"""
    
    # Parallelization
    num_workers: int = 1
    """Number of parallel workers for MCTS (1 = single-threaded)"""
    
    # Constants
    INFINITE_VALUE: ClassVar[float] = float('inf')
    """Value representing infinity in the algorithm"""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        
        if self.exploration_weight <= 0:
            raise ValueError("exploration_weight must be positive")
        
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        
        if self.time_limit is not None and self.time_limit <= 0:
            raise ValueError("time_limit must be positive or None")
        
        if self.simulation_policy not in ["random", "heuristic"]:
            raise ValueError("simulation_policy must be 'random' or 'heuristic'")
        
        if self.widening_factor <= 0 or self.widening_factor >= 1:
            raise ValueError("widening_factor must be between 0 and 1")
        
        if self.transposition_table_size <= 0:
            raise ValueError("transposition_table_size must be positive")
        
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
    
    @classmethod
    def default(cls) -> 'MCTSConfig':
        """
        Get the default configuration.
        
        Returns:
            Default MCTSConfig object
        """
        return cls()
    
    @classmethod
    def fast(cls) -> 'MCTSConfig':
        """
        Get a configuration optimized for speed (fewer iterations).
        
        Returns:
            Fast MCTSConfig object
        """
        return cls(
            iterations=100,
            use_heuristics=True,
            simulation_policy="heuristic",
            use_transposition_table=True
        )
    
    @classmethod
    def deep(cls) -> 'MCTSConfig':
        """
        Get a configuration optimized for deep search.
        
        Returns:
            Deep MCTSConfig object
        """
        return cls(
            iterations=5000,
            exploration_weight=1.2,  # Slightly less exploration
            max_depth=200,
            use_heuristics=True,
            simulation_policy="heuristic",
            use_transposition_table=True
        )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MCTSConfig':
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict: Dictionary of configuration parameters
            
        Returns:
            MCTSConfig object
        """
        # Filter out any keys that aren't valid parameters
        valid_params = {k: v for k, v in config_dict.items() 
                       if k in cls.__dataclass_fields__}
        return cls(**valid_params)
    
    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary of configuration parameters
        """
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if not isinstance(field.type, type(ClassVar))
        }
    
    def __str__(self) -> str:
        """
        Get a human-readable string representation.
        
        Returns:
            String representation
        """
        params = []
        for field_name, field in self.__dataclass_fields__.items():
            if not isinstance(field.type, type(ClassVar)):
                value = getattr(self, field_name)
                params.append(f"{field_name}={value}")
        
        return f"MCTSConfig({', '.join(params)})"
