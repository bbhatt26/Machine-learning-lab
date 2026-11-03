"""Custom Gymnasium environment for the maze.
Legend: '#' wall, 'S' start, 'G' goal, 'T' trap, ' ' free cell.
Actions: 0=Up 1=Right 2=Down 3=Left.
Rewards: -1 per step (encourages SHORTEST path), -5 hitting a wall, -10 stepping on a trap,
         +100 reaching the goal."""
from collections import deque
import numpy as np, gymnasium as gym
from gymnasium import spaces

DEFAULT_MAZE = [
    "############",
    "#S   #     #",
    "# ## # ### #",
    "# #  #   # #",
    "# # #### # #",
    "# #T     # #",
    "# ###### # #",
    "#      #T  #",
    "# #### ### #",
    "# #  #   # #",
    "# #T   #  G#",
    "############",
]
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ARROWS = ["^", ">", "v", "<"]

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, maze=None, max_steps=300, render_mode=None):
        self.grid = np.array([list(r) for r in (maze or DEFAULT_MAZE)])
        self.rows, self.cols = self.grid.shape
        self.start = tuple(map(int, np.argwhere(self.grid == "S")[0]))
        self.goal = tuple(map(int, np.argwhere(self.grid == "G")[0]))
        self.max_steps, self.render_mode = max_steps, render_mode
        self.observation_space = spaces.Discrete(self.rows * self.cols)
        self.action_space = spaces.Discrete(4)
        self.pos, self.steps = self.start, 0

    def to_state(self, pos): return pos[0] * self.cols + pos[1]
    def to_pos(self, s): return divmod(int(s), self.cols)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos, self.steps = self.start, 0
        return self.to_state(self.pos), {}

    def step(self, action):
        self.steps += 1
        r, c = self.pos[0] + MOVES[action][0], self.pos[1] + MOVES[action][1]
        if not (0 <= r < self.rows and 0 <= c < self.cols) or self.grid[r, c] == "#":
            reward, terminated = -5.0, False                 # bumped into a wall: stay in place
        else:
            self.pos = (r, c)
            cell = self.grid[r, c]
            reward = 100.0 if cell == "G" else -10.0 if cell == "T" else -1.0
            terminated = cell == "G"
        truncated = self.steps >= self.max_steps and not terminated
        return self.to_state(self.pos), reward, terminated, truncated, {}

    def render(self):
        g = self.grid.copy(); g[self.pos] = "A"
        return "\n".join("".join(r) for r in g)

    def shortest_path(self):
        """BFS shortest path from start to goal avoiding walls and traps (ground truth)."""
        q, seen = deque([(self.start, [self.start])]), {self.start}
        while q:
            p, path = q.popleft()
            if p == self.goal: return path
            for dr, dc in MOVES:
                n = (p[0] + dr, p[1] + dc)
                if 0 <= n[0] < self.rows and 0 <= n[1] < self.cols and n not in seen and self.grid[n] not in "#T":
                    seen.add(n); q.append((n, path + [n]))
        return None
