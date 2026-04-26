import pytest
import numpy as np
from src.env.robot import Robot, Task, Grid, TaskQueue
from src.env.environment import WarehouseEnv, SimpleWarehouseEnv


class TestRobot:
    def test_robot_initialization(self):
        robot = Robot(0, (5, 5), 20)
        assert robot.robot_id == 0
        assert robot.position[0] == 5
        assert robot.position[1] == 5
        assert not robot.has_load

    def test_robot_move(self):
        grid = Grid(20)
        robot = Robot(0, (5, 5), 20)

        robot.move(1, grid)
        assert robot.position[0] == 6

        robot.move(2, grid)
        assert robot.position[0] == 5

    def test_robot_reset(self):
        robot = Robot(0, (5, 5), 20)
        robot.position = np.array([10, 10])
        robot.reset((5, 5))
        assert robot.position[0] == 5
        assert robot.position[1] == 5


class TestGrid:
    def test_grid_initialization(self):
        grid = Grid(10)
        assert grid.size == 10
        assert len(grid.obstacles) == 0

    def test_add_obstacle(self):
        grid = Grid(10)
        grid.add_obstacle(5, 5)
        assert grid.is_obstacle(5, 5)

    def test_add_pickup_zone(self):
        grid = Grid(10)
        grid.add_pickup_zone(2, 2)
        assert grid.is_pickup(2, 2)

    def test_add_delivery_zone(self):
        grid = Grid(10)
        grid.add_delivery_zone(8, 8)
        assert grid.is_delivery(8, 8)


class TestTaskQueue:
    def test_task_queue_initialization(self):
        queue = TaskQueue(10)
        assert len(queue) == 0

    def test_add_task(self):
        queue = TaskQueue(10)
        task = Task(1, "pickup", (2, 2), (5, 5))
        assert queue.add_task(task)
        assert len(queue) == 1

    def test_get_pickup_tasks(self):
        queue = TaskQueue(10)
        task1 = Task(1, "pickup", (2, 2), (5, 5))
        task2 = Task(2, "delivery", (5, 5))
        queue.add_task(task1)
        queue.add_task(task2)

        pickups = queue.get_pickup_tasks()
        assert len(pickups) == 1


class TestWarehouseEnv:
    def test_env_initialization(self):
        env = WarehouseEnv(num_agents=3, grid_size=15, seed=42)
        assert env.num_agents == 3
        assert env.grid_size == 15

    def test_env_reset(self):
        env = WarehouseEnv(num_agents=3, grid_size=15, seed=42)
        obs, info = env.reset()
        assert obs.shape == (3, 17)
        assert "tasks_completed" in info

    def test_env_step(self):
        env = WarehouseEnv(num_agents=3, grid_size=15, seed=42)
        obs, info = env.reset()

        actions = [0, 1, 2]
        obs, reward, terminated, truncated, info = env.step(actions)

        assert obs.shape == (3, 17)
        assert isinstance(reward, (int, float))


class TestSimpleWarehouseEnv:
    def test_simple_env_reset(self):
        env = SimpleWarehouseEnv(num_agents=5, grid_size=20, seed=42)
        result = env.reset()
        assert "agents" in result
        assert result["agents"].shape[0] == 5

    def test_simple_env_step(self):
        env = SimpleWarehouseEnv(num_agents=5, grid_size=20, seed=42)
        env.reset()

        actions = [0, 1, 2, 3, 4]
        result = env.step(actions)

        assert len(result) == 6
        obs, global_state, reward, terminated, truncated, info = result
        assert obs["agents"].shape[0] == 5