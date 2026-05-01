# src/train/trainer/__init__.py

from .constructor import init
from .methods._run_episode import _run_episode
from .methods.train import train
from .methods.evaluate import evaluate

class Trainer:
    pass

Trainer.__init__ = init
Trainer._run_episode = _run_episode
Trainer.train = train
Trainer.evaluate = evaluate
