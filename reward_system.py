import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
class Config:
    # Paper-specific constants
    VELOCITY_THRESHOLD = 0.5  # From paper - velocity threshold for shape change
    FLOW_THRESHOLD = 0.2  # From paper - flow threshold for shape stability

    # Reward parameters
    BASE_REWARD = 10  # Base reward for task completion
    BONUS_REWARD = 5  # Bonus reward for efficient shape change
    PENALTY = -2  # Penalty for each time-step the task is not completed
    DISCOUNT_FACTOR = 0.9  # Reward discount factor for future rewards

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Custom exceptions
class InvalidShapeError(Exception):
    pass

class RewardCalculationError(Exception):
    pass

# Data structures/models
class Shape:
    def __init__(self, velocity, flow):
        self.velocity = velocity
        self.flow = flow

    def is_stable(self):
        return self.velocity < Config.VELOCITY_THRESHOLD and self.flow < Config.FLOW_THRESHOLD

# Main class - RewardSystem
class RewardSystem:
    def __init__(self, config: Config):
        self.config = config
        self.total_reward = 0
        self.rewards = []
        self.shapes = []

    def calculate_reward(self, current_shape: Shape, target_shape: Shape) -> float:
        """
        Calculate the reward based on the current and target shapes.

        Parameters:
            current_shape (Shape): The current shape of the object.
            target_shape (Shape): The desired target shape.

        Returns:
            float: The calculated reward.
        """
        if not isinstance(current_shape, Shape) or not isinstance(target_shape, Shape):
            raise InvalidShapeError("Invalid shape input. Expected instances of Shape class.")

        # Paper's methodology: Reward based on velocity and flow thresholds
        if current_shape.is_stable():
            # Task completed - base reward
            reward = self.config.BASE_REWARD
        else:
            # Task not completed - small penalty
            reward = self.config.PENALTY

        # Bonus reward for efficient shape change
        if current_shape.velocity < target_shape.velocity and current_shape.flow < target_shape.flow:
            reward += self.config.BONUS_REWARD

        return reward

    def update_reward(self, current_shape: Shape, target_shape: Shape) -> None:
        """
        Update the total reward and rewards list based on the current and target shapes.

        Parameters:
            current_shape (Shape): The current shape of the object.
            target_shape (Shape): The desired target shape.

        Returns:
            None
        """
        reward = self.calculate_reward(current_shape, target_shape)
        self.total_reward += reward * self.config.DISCOUNT_FACTOR  # Apply discount factor
        self.rewards.append(reward)
        logger.debug(f"Reward for this step: {reward}")

    def get_total_reward(self) -> float:
        """
        Get the total accumulated reward.

        Returns:
            float: The total accumulated reward.
        """
        return self.total_reward

    def reset(self) -> None:
        """
        Reset the reward system for a new episode.

        Returns:
            None
        """
        self.total_reward = 0
        self.rewards = []
        self.shapes = []

    def record_shape(self, shape: Shape) -> None:
        """
        Record a shape for future reference.

        Parameters:
            shape (Shape): The shape to be recorded.

        Returns:
            None
        """
        self.shapes.append(shape)

# Helper functions
def process_shapes(shapes: List[Shape]) -> Tuple[Shape, Shape]:
    """
    Process a list of shapes to find the current and target shapes.

    Parameters:
        shapes (List[Shape]): A list of Shape objects.

    Returns:
        Tuple[Shape, Shape]: The current and target shapes.
    """
    if not all(isinstance(shape, Shape) for shape in shapes):
        raise InvalidShapeError("Invalid shape input. Expected a list of Shape objects.")

    current_shape = shapes[-1]  # Assume last shape is the current one
    target_shape = max(shapes, key=lambda x: x.velocity)  # Assume shape with max velocity is the target

    return current_shape, target_shape

# Integration interfaces
def calculate_reward_for_episode(shapes: List[Shape]) -> float:
    """
    Calculate the total reward for an episode consisting of a list of shapes.

    Parameters:
        shapes (List[Shape]): A list of Shape objects representing the episode.

    Returns:
        float: The total reward for the episode.
    """
    reward_system = RewardSystem(Config())
    for shape in shapes:
        reward_system.record_shape(shape)

    current_shape, target_shape = process_shapes(shapes)
    for _ in range(len(shapes)):  # Simulate time-steps
        reward_system.update_reward(current_shape, target_shape)

    return reward_system.get_total_reward()

# Example usage
if __name__ == "__main__":
    shapes = [
        Shape(velocity=0.2, flow=0.1),
        Shape(velocity=0.4, flow=0.3),
        Shape(velocity=0.1, flow=0.2),
        Shape(velocity=0.3, flow=0.1)
    ]

    total_reward = calculate_reward_for_episode(shapes)
    print(f"Total reward for the episode: {total_reward}")