# src/model/qmix/qmixer/__init__.py

import torch.nn as nn
from .constructor import init
from .methods.forward import forward

class QMixer(nn.Module):
    pass

QMixer.__init__ = init
QMixer.forward = forward
