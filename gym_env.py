import gymnasium as gym
import numpy as np
from typing import Tuple, Any, Dict
from env_interface import BaseEnvironment

class GymnasiumEnv(BaseEnvironment):
    """
    Concrete implementation of BaseEnvironment wrapping a standard Gymnasium (or OpenAI Gym) environment.
    """

    def __init__(self, env_id: str, render_mode: str = None):
        """
        Initializes the Gymnasium environment.

        Args:
            env_id (str): The ID of the environment (e.g., 'BipedalWalker-v3').
            render_mode (str, optional): The Gymnasium render mode ('human', 'rgb_array', etc.).
        """
        self.env_id = env_id
        self.render_mode = render_mode
        self.env = gym.make(env_id, render_mode=render_mode)

    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to an initial state.
        """
        return self.env.reset(seed=seed)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Steps the wrapped environment using the provided action.
        """
        return self.env.step(action)

    def close(self) -> None:
        """
        Closes and cleans up the wrapped environment.
        """
        self.env.close()

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the observation space.
        """
        return self.env.observation_space.shape

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the action space.
        """
        return self.env.action_space.shape

    @property
    def action_low(self) -> np.ndarray:
        """
        Returns the lower bounds of the action space.
        """
        return self.env.action_space.low

    @property
    def action_high(self) -> np.ndarray:
        """
        Returns the upper bounds of the action space.
        """
        return self.env.action_space.high
