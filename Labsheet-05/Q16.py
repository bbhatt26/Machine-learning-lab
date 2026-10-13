# Q16: Design a simple Grid World environment
import numpy as np
class GridWorld:
    """A 4x4 grid: start=(0,0), goal=(3,3), one blocked cell=(1,1)."""
    def __init__(self):
        self.size = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.wall = (1, 1)
        self.state = self.start
        self.actions = [0, 1, 2, 3]  # Up, Down, Left, Right
    def reset(self):
        self.state = self.start
        return self.to_id(self.state)
    def to_id(self, pos):
        return pos[0] * self.size + pos[1]
    def step(self, action):
        r, c = self.state
        moves = {0: (r-1, c), 1: (r+1, c), 2: (r, c-1), 3: (r, c+1)}
        nr, nc = moves[action]
        candidate = (min(max(nr, 0), 3), min(max(nc, 0), 3))
        if candidate != self.wall:
            self.state = candidate
        done = self.state == self.goal
        return self.to_id(self.state), (10 if done else -1), done
env = GridWorld()
print("Start state ID:", env.reset())
print("Actions: 0=Up, 1=Down, 2=Left, 3=Right")
print("One right action:", env.step(3))
