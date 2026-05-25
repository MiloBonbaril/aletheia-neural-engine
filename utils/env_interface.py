from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict
import numpy as np

class BaseEnvironment(ABC):
    """
    Abstract Base Class defining the unified environment interface.
    Any RL agent or training pipeline in this project will interact only with this interface,
    allowing seamless switching between Gymnasium, custom simulators, or Rust-based bridges.
    """

    @abstractmethod
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to an initial state.

        Args:
            seed (int, optional): The seed to initialize the environment's PRNG.

        Returns:
            state (np.ndarray): The initial observation state.
            info (dict): Auxiliary information.
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Takes an environmental action step.

        Args:
            action (np.ndarray): Action vector chosen by the agent.

        Returns:
            next_state (np.ndarray): Next observation.
            reward (float): Reward received.
            terminated (bool): True if the agent reached an terminal state.
            truncated (bool): True if the episode was truncated (e.g. time limit).
            info (dict): Auxiliary diagnostic details.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Cleans up and releases environment resources.
        """
        pass

    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the observation space.
        """
        pass

    @property
    @abstractmethod
    def action_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the action space.
        """
        pass

    @property
    @abstractmethod
    def action_low(self) -> np.ndarray:
        """
        Returns the lower bound of the action space dimensions.
        """
        pass

    @property
    @abstractmethod
    def action_high(self) -> np.ndarray:
        """
        Returns the upper bound of the action space dimensions.
        """
        pass
