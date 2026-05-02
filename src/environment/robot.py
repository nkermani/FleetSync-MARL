import numpy as np

class Robot:
    def __init__(self, robot_id, position, grid_size):
        self.robot_id = robot_id
        self.position = np.array(position, dtype=int)
        self.grid_size = grid_size
        self.has_load = False
        self.load_destination = None

    def move(self, action, grid):
        """Move robot based on action: 0=up, 1=right, 2=left, 3=down, 4=wait"""
        new_pos = self.position.copy()

        if action == 0:  # up
            new_pos[1] = max(0, self.position[1] - 1)
        elif action == 1:  # right
            new_pos[0] = min(self.grid_size - 1, self.position[0] + 1)
        elif action == 2:  # left
            new_pos[0] = max(0, self.position[0] - 1)
        elif action == 3:  # down
            new_pos[1] = min(self.grid_size - 1, self.position[1] + 1)
        elif action == 4:  # wait
            pass

        if not grid.is_obstacle(new_pos[0], new_pos[1]):
            self.position = new_pos

    def reset(self, position):
        self.position = np.array(position, dtype=int)
        self.has_load = False
        self.load_destination = None

    def pickup(self, task):
        self.has_load = True
        self.load_destination = task.delivery_location

    def deliver(self):
        self.has_load = False
        self.load_destination = None


class Task:
    def __init__(self, task_id, task_type, pickup_location, delivery_location=None):
        self.task_id = task_id
        self.task_type = task_type
        self.pickup_location = np.array(pickup_location, dtype=int)
        self.delivery_location = np.array(delivery_location, dtype=int) if delivery_location else None
        self.completed = False


class Grid:
    def __init__(self, size):
        self.size = size
        self.obstacles = set()
        self.pickup_zones = set()
        self.delivery_zones = set()

    def add_obstacle(self, x, y):
        self.obstacles.add((x, y))

    def add_pickup_zone(self, x, y):
        self.pickup_zones.add((x, y))

    def add_delivery_zone(self, x, y):
        self.delivery_zones.add((x, y))

    def is_obstacle(self, x, y):
        return (x, y) in self.obstacles

    def is_pickup(self, x, y):
        return (x, y) in self.pickup_zones

    def is_delivery(self, x, y):
        return (x, y) in self.delivery_zones

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size


class TaskQueue:
    def __init__(self, max_tasks):
        self.max_tasks = max_tasks
        self.tasks = []
        self.next_id = 0

    def add_task(self, task):
        if len(self.tasks) < self.max_tasks:
            task.task_id = self.next_id
            self.next_id += 1
            self.tasks.append(task)
            return True
        return False

    def get_pickup_tasks(self):
        return [t for t in self.tasks if t.task_type == "pickup" and not t.completed]

    def get_task_at(self, position):
        for task in self.tasks:
            if not task.completed and np.array_equal(task.pickup_location, position):
                return task
        return None

    def complete_task(self, task_id):
        for task in self.tasks:
            if task.task_id == task_id:
                task.completed = True
                return True
        return False

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)
