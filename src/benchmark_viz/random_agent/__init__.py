# src/benchmark_viz/random_agent/__init__.py

from .constructor import init
from .methods.get_action import get_action

class RandomAgent:
    """Random baseline."""
    pass

RandomAgent.__init__ = init
RandomAgent.get_action = get_action
