import pytest
import torch
import numpy as np
from src.model.qmix import RobotActor, MixingNetwork, QMIXNetwork, SimpleQMIX
from src.train import ReplayBuffer


class TestRobotActor:
    def test_actor_initialization(self):
        actor = RobotActor(17, 5, 64)
        assert actor.obs_dim == 17
        assert actor.n_actions == 5

    def test_actor_forward(self):
        actor = RobotActor(17, 5, 64)
        obs = torch.randn(17)
        output = actor(obs)
        assert output.shape == (5,)

    def test_actor_get_action(self):
        actor = RobotActor(17, 5, 64)
        obs = torch.randn(17)
        action = actor.get_action(obs, epsilon=0.0)
        assert 0 <= action < 5


class TestMixingNetwork:
    def test_mixer_initialization(self):
        mixer = MixingNetwork(10, 3, 32)
        assert mixer.state_dim == 10
        assert mixer.num_agents == 3

    def test_mixer_forward(self):
        mixer = MixingNetwork(10, 3, 32)
        q_vals = torch.randn(3, 3)
        state = torch.randn(10)
        output = mixer(q_vals, state)
        assert output.shape == (3, 1)


class TestQMIXNetwork:
    def test_qmix_initialization(self):
        qmix = QMIXNetwork(num_agents=3, obs_dim=17, n_actions=5)
        assert qmix.num_agents == 3
        assert qmix.obs_dim == 17
        assert qmix.n_actions == 5

    def test_qmix_forward(self):
        qmix = QMIXNetwork(num_agents=3, obs_dim=17, n_actions=5)
        obs = torch.randn(3, 17)
        state = torch.randn(3 * 17)
        total_q, individual_qs = qmix(obs, state)
        assert total_q.shape[0] == 3

    def test_qmix_get_actions(self):
        qmix = QMIXNetwork(num_agents=3, obs_dim=17, n_actions=5)
        obs = torch.randn(3, 17)
        state = torch.randn(3 * 17)
        actions = qmix.get_actions(obs, state, epsilon=0.0)
        assert len(actions) == 3
        for a in actions:
            assert 0 <= a < 5


class TestSimpleQMIX:
    def test_simple_qmix_initialization(self):
        qmix = SimpleQMIX(num_agents=3, obs_dim=17, n_actions=5)
        assert qmix.num_agents == 3

    def test_simple_qmix_update(self):
        qmix = SimpleQMIX(num_agents=3, obs_dim=17, n_actions=5)

        batch = {
            "obs": np.random.randn(32, 3, 17),
            "state": np.random.randn(32, 3 * 17),
            "actions": np.random.randint(0, 5, (32, 3)),
            "reward": np.random.randn(32),
            "next_obs": np.random.randn(32, 3, 17),
            "next_state": np.random.randn(32, 3 * 17),
            "done": np.zeros(32),
        }

        loss = qmix.update(batch)
        assert isinstance(loss, float)


class TestReplayBuffer:
    def test_buffer_initialization(self):
        buffer = ReplayBuffer(1000, 3, 17, 51)
        assert buffer.capacity == 1000
        assert buffer.num_agents == 3

    def test_buffer_push_and_sample(self):
        buffer = ReplayBuffer(1000, 3, 17, 51)

        obs = np.random.randn(3, 17)
        state = np.random.randn(51)
        actions = [0, 1, 2]
        reward = 1.0
        next_obs = np.random.randn(3, 17)
        next_state = np.random.randn(51)
        done = False

        buffer.push(obs, state, actions, reward, next_obs, next_state, done)
        assert len(buffer) == 1

        sample = buffer.sample(1)
        assert "obs" in sample
        assert sample["obs"].shape == (1, 3, 17)