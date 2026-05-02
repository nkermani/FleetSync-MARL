import numpy as np
from .robot import Robot, Task, Grid, TaskQueue


class WarehouseEnv:
    """Multi-agent warehouse environment for MARL training."""

    def __init__(self, num_agents=3, num_tasks=10, grid_size=15, max_steps=500, collision_penalty=-10, seed=None):
        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.collision_penalty = collision_penalty
        self.steps = 0
        self.tasks_completed = 0

        if seed is not None:
            np.random.seed(seed)

        self.grid = Grid(grid_size)
        self.robots = []
        self.task_queue = TaskQueue(num_tasks)

        self._setup_warehouse()

    def _setup_warehouse(self):
        """Initialize warehouse layout with zones and obstacles."""
        # Add pickup and delivery zones
        for i in range(2, self.grid_size - 2, 3):
            self.grid.add_pickup_zone(2, i)
            self.grid.add_delivery_zone(self.grid_size - 3, i)

        # Add some random obstacles
        for _ in range(self.grid_size // 2):
            x = np.random.randint(5, self.grid_size - 5)
            y = np.random.randint(5, self.grid_size - 5)
            self.grid.add_obstacle(x, y)

        # Initialize robots at random positions
        for i in range(self.num_agents):
            while True:
                pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                if not self.grid.is_obstacle(pos[0], pos[1]) and not any(
                    np.array_equal(pos, r.position) for r in self.robots
                ):
                    self.robots.append(Robot(i, pos, self.grid_size))
                    break

        # Create initial tasks
        for _ in range(self.num_tasks):
            pickup = self._random_zone_location(self.grid.pickup_zones)
            delivery = self._random_zone_location(self.grid.delivery_zones)
            if pickup is not None and delivery is not None:
                task = Task(None, "pickup", pickup, delivery)
                self.task_queue.add_task(task)

    def _random_zone_location(self, zone_set):
        """Get random location from a set of zone positions."""
        if not zone_set:
            return None
        return list(zone_set)[np.random.randint(0, len(zone_set))]

    def _get_global_state(self):
        """Get global state representation (concatenated observations of all agents)."""
        obs = self._get_observations()
        return obs.flatten()  # Shape: (num_agents * obs_dim,)

    def reset(self):
        """Reset environment to initial state."""
        self.steps = 0
        self.tasks_completed = 0
        self.robots = []
        self.task_queue = TaskQueue(self.num_tasks)
        self._setup_warehouse()

        obs = self._get_observations()
        info = {"tasks_completed": self.tasks_completed}
        return obs, info

    def step(self, actions):
        """Execute actions for all agents."""
        self.steps += 1
        rewards = []
        collisions = set()

        # Check for collisions
        positions = {}
        for i, robot in enumerate(self.robots):
            pos_tuple = tuple(robot.position)
            if pos_tuple in positions:
                collisions.add(i)
                collisions.add(positions[pos_tuple])
            else:
                positions[pos_tuple] = i

        # Move robots and collect rewards
        for i, robot in enumerate(self.robots):
            if i in collisions:
                rewards.append(self.collision_penalty)
                continue

            action = actions[i] if isinstance(actions, (list, np.ndarray)) else actions
            robot.move(action, self.grid)

            # Check for task pickup/delivery
            reward = -0.01  # Small step penalty

            task = self.task_queue.get_task_at(tuple(robot.position))
            if task and not robot.has_load:
                robot.pickup(task)
                reward += 10
            elif robot.has_load and self.grid.is_delivery(robot.position[0], robot.position[1]):
                robot.deliver()
                self.tasks_completed += 1
                reward += 20

            rewards.append(reward)

        obs = self._get_observations()
        state = self._get_global_state()
        total_reward = sum(rewards)
        terminated = self.steps >= self.max_steps or self.tasks_completed >= self.num_tasks
        truncated = False
        info = {"tasks_completed": self.tasks_completed}

        return obs, total_reward, terminated, truncated, info

    def _get_observations(self):
        """Get observations for all agents."""
        obs = []
        for robot in self.robots:
            agent_obs = self._get_agent_observation(robot)
            obs.append(agent_obs)
        return np.array(obs)

    def _get_agent_observation(self, robot):
        """Get local observation for a single agent (17-dimensional)."""
        obs = np.zeros(17)

        # Position (2)
        obs[0] = robot.position[0] / self.grid_size
        obs[1] = robot.position[1] / self.grid_size

        # Has load flag (1)
        obs[2] = 1 if robot.has_load else 0

        # Local grid neighborhood 5x5 -> 5 channels (10)
        x, y = robot.position
        idx = 3
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if not self.grid.in_bounds(nx, ny):
                    obs[idx] = 1  # Out of bounds
                elif self.grid.is_obstacle(nx, ny):
                    obs[idx] = 1
                elif any(np.array_equal([nx, ny], r.position) for r in self.robots if r.robot_id != robot.robot_id):
                    obs[idx] = 1
                elif self.grid.is_pickup(nx, ny):
                    obs[idx] = 0.5
                elif self.grid.is_delivery(nx, ny):
                    obs[idx] = 0.5
                idx += 1
                if idx >= 13:
                    break
            if idx >= 13:
                break

        # Distance to nearest pickup/delivery (2)
        if robot.has_load and robot.load_destination is not None:
            dist = np.linalg.norm(robot.position - robot.load_destination) / self.grid_size
            obs[13] = dist
            obs[14] = 1  # Has destination
        else:
            # Find nearest pickup
            pickups = self.task_queue.get_pickup_tasks()
            if pickups:
                nearest = min(pickups, key=lambda t: np.linalg.norm(robot.position - t.pickup_location))
                dist = np.linalg.norm(robot.position - nearest.pickup_location) / self.grid_size
                obs[13] = dist
                obs[14] = 0
            else:
                obs[13] = 1
                obs[14] = 0

        # Global progress (2)
        obs[15] = self.tasks_completed / max(1, self.num_tasks)
        obs[16] = self.steps / self.max_steps

        return obs

    def render(self, mode='human'):
        """Render the warehouse state."""
        print(f"Step: {self.steps}, Tasks Completed: {self.tasks_completed}")
        print("Robots:", [r.position.tolist() for r in self.robots])


class SimpleWarehouseEnv:
    """Simplified warehouse environment with gymnasium-style interface."""

    def __init__(self, num_agents=5, grid_size=20, seed=None):
        self.num_agents = num_agents
        self.grid_size = grid_size
        if seed is not None:
            np.random.seed(seed)

        self.grid = Grid(grid_size)
        self.robots = []
        self.steps = 0
        self.max_steps = 500
        self.tasks_completed = 0

    def reset(self):
        """Reset environment."""
        self.steps = 0
        self.tasks_completed = 0
        self.robots = []

        # Simple setup
        for i in range(self.num_agents):
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            self.robots.append(Robot(i, pos, self.grid_size))

        obs = self._get_observations()
        return {"agents": obs, "state": obs.flatten()}

    def step(self, actions):
        """Take a step in the environment."""
        self.steps += 1

        for i, robot in enumerate(self.robots):
            action = actions[i] if i < len(actions) else 4
            robot.move(action, self.grid)

        obs = self._get_observations()
        state = obs.flatten()
        reward = -0.01
        terminated = self.steps >= self.max_steps
        truncated = False

        return {"agents": obs, "state": state}, state, reward, terminated, truncated, {"tasks_completed": self.tasks_completed}

    def _get_observations(self):
        """Get simple observations for all agents."""
        obs = []
        for robot in self.robots:
            agent_obs = np.zeros(17)
            agent_obs[0] = robot.position[0] / self.grid_size
            agent_obs[1] = robot.position[1] / self.grid_size
            agent_obs[2] = 1 if robot.has_load else 0
            obs.append(agent_obs)
        return np.array(obs)
