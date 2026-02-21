import numpy as np
from gymnasium import Space
from typing import Any, List, Tuple
from pettingzoo.utils import BaseWrapper


class VariableLengthTupleSpace(Space):
    def __init__(self, max_length, low, high, allow_noop=True):
        super().__init__()
        self.max_length = max_length
        self.low = low  # Minimum value in each dimension
        self.high = high  # Maximum value in each dimension
        self.allow_noop = allow_noop

    def sample(self, mask: Any | None = None) -> List[Tuple[int, int]]:
        """Generate a random sample action within the space, with bias towards shorter actions."""
        if self.allow_noop and np.random.random() < 0.05:
            return []

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
        if len(x) == 0:
            return self.allow_noop
        if not (1 <= len(x) <= self.max_length):
            return False
        return all(self.low <= t[0] <= self.high and self.low <= t[1] <= self.high for t in x)


class HandleNoOpWrapper(BaseWrapper):
    def step(self, action):
        current_agent = self.agent_selection
        is_dead_agent = (
            current_agent is not None
            and (
                self.terminations.get(current_agent, False)
                or self.truncations.get(current_agent, False)
            )
        )

        if action is None and not is_dead_agent:
            # Normalize live-agent None no-op into an explicit in-space no-op action.
            return super().step([])

        # Dead agents must receive None so _was_dead_step can process removal correctly.
        return super().step(action)
