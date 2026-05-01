# src/model/qmix/qmix_network/__init__.py

import torch.nn as nn
from .constructor import init
from .methods.forward import forward
from .methods.get_actions import get_actions

class QMIXNetwork(nn.Module):
    pass

QMIXNetwork.__init__ = init
QMIXNetwork.forward = forward
QMIXNetwork.get_actions = get_actions
