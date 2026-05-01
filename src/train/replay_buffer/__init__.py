# src/train/replay_buffer/__init__.py

from .constructor import init
from .methods.push import push
from .methods.sample import sample
from .methods.__len__ import __len__

class ReplayBuffer:
    pass

ReplayBuffer.__init__ = init
ReplayBuffer.push = push
ReplayBuffer.sample = sample
ReplayBuffer.__len__ = __len__
