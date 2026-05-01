# src/model/qmix/qmix_agent/__init__.py

from .constructor import init
from .methods.update import update
from .methods.update_target import update_target
from .methods.get_actions import get_actions

class QMIXAgent:
    pass

QMIXAgent.__init__ = init
QMIXAgent.update = update
QMIXAgent.update_target = update_target
QMIXAgent.get_actions = get_actions
