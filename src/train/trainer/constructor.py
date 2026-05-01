# src/train/trainer/constructor.py

def init(self, agent, target_update_freq=200):
    self.agent = agent
    self.target_update_freq = target_update_freq
    self.update_count = 0
    self.training_history = []
