"""
Data collection utilities for robot control execution.

This module provides clean data collection during robot control for analysis.
"""
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ExecutionResult:
    """
    Results from robot execution containing collected data.
    """
    # Trajectory data
    joint_velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    time_history: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Execution metadata
    converged: bool = False
    execution_time: float = 0.0
    n_steps: int = 0
    
    # Optional: end-effector data
    ee_poses: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for saving."""
        return {
            'joint_velocities': self.joint_velocities,
            'joint_positions': self.joint_positions,
            'time_history': self.time_history,
            'converged': self.converged,
            'execution_time': self.execution_time,
            'n_steps': self.n_steps,
            'ee_poses': self.ee_poses
        }
    
    def save(self, filename: str):
        """Save execution data to file."""
        np.savez(filename, **self.to_dict())


class DataCollector:
    """
    Efficient data collector for robot control execution.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all collected data."""
        self._qd_history: List[np.ndarray] = []
        self._q_history: List[np.ndarray] = []
        self._time_history: List[float] = []
        self._ee_poses: List[np.ndarray] = []
        self._start_time: Optional[float] = None
    
    def start_collection(self, start_time: float):
        """Mark the start of data collection."""
        self._start_time = start_time
    
    def collect_step(self, qd: np.ndarray, q: np.ndarray, time: float, ee_pose: Optional[np.ndarray] = None):
        """Collect data from one control step."""
        self._qd_history.append(qd.copy())
        self._q_history.append(q.copy())
        self._time_history.append(time)
        
        if ee_pose is not None:
            self._ee_poses.append(ee_pose.copy())
    
    def get_result(self, converged: bool) -> ExecutionResult:
        """Get final execution result."""
        if not self._qd_history:
            return ExecutionResult()
            
        result = ExecutionResult(
            joint_velocities=np.array(self._qd_history),
            joint_positions=np.array(self._q_history),
            time_history=np.array(self._time_history),
            converged=converged,
            execution_time=self._time_history[-1] - self._time_history[0] if self._time_history else 0.0,
            n_steps=len(self._qd_history)
        )
        
        if self._ee_poses:
            result.ee_poses = np.array(self._ee_poses)
            
        return result
