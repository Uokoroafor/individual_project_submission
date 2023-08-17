import random
from typing import List, Dict, Optional


class RandomDict(dict):
    """A dictionary that returns a new random value each time the key is called."""
    def __getitem__(self, key):
        return super().__getitem__(key)()


def pick_between_two_ranges(range1: List[float, float], range2: List[float, float]) -> float:
    """Picks a random value between two ranges.

    Args:
        range1 (List[float, float]): The first range.
        range2 (List[float, float]): The second range.

    Returns:
        float: A random value between the two ranges.
    """
    if random.random() < 0.5:
        return random.uniform(*range1)
    else:
        return random.uniform(*range2)
