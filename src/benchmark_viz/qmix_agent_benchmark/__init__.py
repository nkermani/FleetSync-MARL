# src/benchmark_viz/qmix_agent_benchmark/__init__.py

import torch.nn as nn
from .constructor import init
from .methods.forward import forward
from .methods.get_actions import get_actions
from .methods.update import update

class QMIXAgent(nn.Module):
    """QMIX - value decomposition with state conditioning."""
    pass

QMIXAgent.__init__ = init
QMIXAgent.forward = forward
QMIXAgent.get_actions = get_actions
QMIXAgent.update = update
