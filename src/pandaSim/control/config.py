"""
Execution configuration for robot control.

This module provides configuration parameters specifically for robot control execution.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionConfig:
    """
    Configuration for robot control execution.
    
    Focuses only on controller-related parameters, avoiding UI/plotting configs.
    """
    
    # Control loop parameters
    frequency: float = 100.0  # Control frequency in Hz
    max_runtime: float = 15.0  # Maximum execution time in seconds
    
    # Controller convergence
    convergence_threshold: float = 0.001  # When to consider target reached
    
    # Safety timeouts
    safety_timeout: float = 30.0  # Absolute safety timeout in seconds
    
    # Simulation parameters
    sim_dt: Optional[float] = None  # Simulation timestep (auto-computed if None)
    
    def __post_init__(self):
        """Compute derived parameters."""
        if self.sim_dt is None:
            self.sim_dt = 1.0 / self.frequency
