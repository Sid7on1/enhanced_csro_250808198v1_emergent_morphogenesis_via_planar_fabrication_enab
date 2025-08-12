import logging
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class ExperienceReplayBuffer:
    """
    Experience replay buffer for storing and sampling experiences.
    """

    def __init__(self, capacity: int, batch_size: int):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory: Deque[Tuple[NDArray[np.float32], NDArray[np.float32], float, NDArray[np.float32], bool]] = deque(
            maxlen=capacity
        )

    def add(
        self, state: NDArray[np.float32], action: NDArray[np.float32], reward: float, next_state: NDArray[np.float32], done: bool
    ) -> None:
        """
        Add a new experience to the buffer.
        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state after taking the action.
        :param done: Whether the episode has terminated.
        """
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float33], NDArray[np.bool]]:
        """
        Sample a batch of experiences from the buffer.
        :return: Tuple of states, actions, rewards, next_states, and dones.
        """
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Return the current number of experiences in the buffer.
        """
        return len(self.memory)


class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    """
    Prioritized experience replay buffer that samples experiences based on their priority.
    """

    def __init__(self, capacity: int, batch_size: int, alpha: float):
        super().__init__(capacity, batch_size)
        self.alpha = alpha
        self.priorities: List[float] = []
        self.priority_sums: List[float] = []

    def add(
        self, state: NDArray[np.float32], action: NDArray[np.float32], reward: float, next_state: NDArray[np.float32], done: bool
    ) -> None:
        """
        Add a new experience to the buffer with a priority of 1.0.
        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state after taking the action.
        :param done: Whether the episode has terminated.
        """
        super().add(state, action, reward, next_state, done)
        self.priorities.append(1.0)
        self.priority_sums.append(1.0)

    def sample(self, beta: float = 0.6) -> Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.bool]]:
        """
        Sample a batch of experiences based on their priority.
        :param beta: Priority exponent for importance sampling weights.
        :return: Tuple of states, indices, actions, rewards, weights, and dones.
        """
        total_priority = sum(self.priority_sums)
        priorities = np.array(self.priority_sums)
        probabilities = priorities ** self.alpha / total_priority
        indices = np.random.choice(len(self.memory), size=self.batch_size, p=probabilities)

        states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in indices])
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool)

        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return states, indices, actions, rewards, weights, dones

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update the priorities of the experiences at the given indices.
        :param indices: List of indices of the experiences to update.
        :param priorities: List of new priorities.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.priority_sums[idx] = priority ** self.alpha

    def __len__(self) -> int:
        """
        Return the current number of experiences in the buffer.
        """
        return super().__len__()


class SequentialMemory:
    """
    Sequential memory for storing and retrieving experiences in sequence.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: List[Tuple[NDArray[np.float32], NDArray[np.float32], float, NDArray[np.float32], bool]] = []

    def add(
        self, state: NDArray[np.float32], action: NDArray[np.float32], reward: float, next_state: NDArray[np.float32], done: bool
    ) -> None:
        """
        Add a new experience to the memory.
        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state after taking the action.
        :param done: Whether the episode has terminated.
        """
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.bool]]:
        """
        Sample a batch of experiences from the memory.
        :param batch_size: Number of experiences to sample.
        :return: Tuple of states, actions, rewards, next_states, and dones.
        """
        batch = random.choices(self.memory, k=batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def clear(self) -> None:
        """
        Clear the memory.
        """
        self.memory.clear()

    def __len__(self) -> int:
        """
        Return the current number of experiences in the memory.
        """
        return len(self.memory)


class EpisodeMemory:
    """
    Episode memory for storing experiences in episodes.
    """

    def __init__(self):
        self.episodes: List[Tuple[List[NDArray[np.float32]], List[NDArray[np.float32]], List[float], List[bool]]] = []

    def add(
        self,
        state: NDArray[np.float32],
        action: NDArray[np.float32],
        reward: float,
        next_state: NDArray[np.float32],
        done: bool,
    ) -> None:
        """
        Add a new experience to the current episode.
        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state after taking the action.
        :param done: Whether the episode has terminated.
        """
        if not self.episodes or done:
            self.episodes.append(
                ([], [], [], [])
            )
        self.episodes[-1][0].append(state)
        self.episodes[-1][1].append(action)
        self.episodes[-1][2].append(reward)
        self.episodes[-1][3].append(done)

    def clear(self) -> None:
        """
        Clear the memory.
        """
        self.episodes.clear()

    def get_episodes(self) -> List[Tuple[List[NDArray[np.float32]], List[NDArray[np.float32]], List[float], List[bool]]]:
        """
        Return the list of episodes.
        """
        return self.episodes


class ExperienceDataset(Dataset):
    """
    Dataset for experience replay.
    """

    def __init__(self, buffer: ExperienceReplayBuffer):
        self.buffer = buffer

    def __len__(self) -> int:
        """
        Return the number of experiences in the dataset.
        """
        return len(self.buffer)

    def __getitem__(self, index: int) -> Tuple[NDArray[np.float32], NDArray[np.float32], float, NDArray[np.float32], bool]:
        """
        Get an experience at the given index.
        :param index: Index of the experience to retrieve.
        :return: Tuple of state, action, reward, next_state, and done.
        """
        state, action, reward, next_state, done = self.buffer.memory[index]
        return state, action, reward, next_state, done


def make_memory(
    memory_type: str, capacity: int, batch_size: int, alpha: Optional[float] = None
) -> "ExperienceReplayBuffer":
    """
    Factory function to create a memory object based on the specified type.
    :param memory_type: Type of memory to create ("experience", "sequential", "episode", "prioritized").
    :param capacity: Capacity of the memory.
    :param batch_size: Batch size for experience replay.
    :param alpha: Priority exponent for prioritized replay.
    :return: Memory object of the specified type.
    """
    if memory_type == "experience":
        return ExperienceReplayBuffer(capacity, batch_size)
    elif memory_type == "sequential":
        return SequentialMemory(capacity)
    elif memory_type == "episode":
        return EpisodeMemory()
    elif memory_type == "prioritized":
        if alpha is None:
            raise ValueError("Alpha value must be provided for prioritized replay.")
        return PrioritizedReplayBuffer(capacity, batch_size, alpha)
    else:
        raise ValueError(f"Invalid memory type: {memory_type}")


def create_data_loader(
    memory: ExperienceReplayBuffer, batch_size: int, shuffle: bool = True
) -> DataLoader:
    """
    Create a data loader for the given memory.
    :param memory: Experience replay memory.
    :param batch_size: Batch size for the data loader.
    :param shuffle: Whether to shuffle the data.
    :return: Data loader for the memory.
    """
    dataset = ExperienceDataset(memory)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Example usage
    memory = make_memory("experience", capacity=1000, batch_size=32)
    data_loader = create_data_loader(memory, batch_size=32)

    for i, (states, actions, rewards, next_states, dones) in enumerate(data_loader):
        print(f"Batch {i}:")
        print("States:", states.shape)
        print("Actions:", actions.shape)
        print("Rewards:", rewards.shape)
        print("Next States:", next_states.shape)
        print("Dones:", dones.shape)
        break