import numpy as np
from gymnasium import Space
from typing import Any, TypeVar
from pettingzoo.utils import BaseWrapper

T_cov = TypeVar("T_cov", covariant=True)


class VariableLengthTupleSpace(Space):
    def __init__(self, max_length, low, high):
        super().__init__()
        self.max_length = max_length  # Maximum sequence lengthTh
        self.low = low  # Minimum value in each dimension
        self.high = high  # Maximum value in each dimension

    def sample(self, mask: Any | None = None) -> T_cov:
        """Generate a random sample action within the space, with bias towards shorter actions."""
        # Create weights that are inversely proportional to the sequence length
        weights = np.exp(-np.arange(0, self.max_length - 2))
        # Normalize weights to create a probability distribution
        probabilities = weights / weights.sum()
        # Randomly choose a sequence length based on the biased probabilities
        length = np.random.choice(np.arange(2, self.max_length), p=probabilities)
        # Generate and return the sample action
        return [tuple(np.random.randint(self.low, self.high + 1, size=2)) for _ in range(length)]

    def contains(self, x):
        """Check if a given action is within the space."""
        if not isinstance(x, list) or not all(isinstance(t, tuple) and len(t) == 2 for t in x):
            return False
        if not (1 <= len(x) <= self.max_length):
            return False
        return all(self.low <= t[0] <= self.high and self.low <= t[1] <= self.high for t in x)


class HandleNoOpWrapper(BaseWrapper):
    def step(self, action):
        if action is None:
            # Use the public method to skip the turn
            self.env.skip_turn()
        else:
            super().step(action)
