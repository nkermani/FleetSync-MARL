# src/model/qmix/robot_actor/__init__.py

import torch.nn as nn
from .constructor import init
from .methods.forward import forward

class RobotActor(nn.Module):
    pass

RobotActor.__init__ = init
RobotActor.forward = forward
