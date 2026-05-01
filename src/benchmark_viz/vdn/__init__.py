# src/benchmark_viz/vdn/__init__.py

import torch.nn as nn
from .constructor import init
from .methods.forward import forward
from .methods.get_actions import get_actions
from .methods.update import update

class VDN(nn.Module):
    """Value Decomposition Network."""
    pass

VDN.__init__ = init
VDN.forward = forward
VDN.get_actions = get_actions
VDN.update = update
