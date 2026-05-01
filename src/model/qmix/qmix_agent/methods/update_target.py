# src/model/qmix/qmix_agent/methods/update_target.py

def update_target(self):
    self.target_model.load_state_dict(self.model.state_dict())
