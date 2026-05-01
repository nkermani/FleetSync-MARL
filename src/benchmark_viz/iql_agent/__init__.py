# src/benchmark_viz/iql_agent/__init__.py

import torch.nn as nn
from .constructor import init
from .methods.get_action import get_action
from .methods.update import update

class IQLAgent(nn.Module):
    """Independent Q-Learning - each agent learns separately."""
    pass

IQLAgent.__init__ = init
IQLAgent.get_action = get_action
IQLAgent.update = update
