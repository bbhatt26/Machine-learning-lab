"""Project 2: Q-Learning based Maze Solver.
Trains the agent, plots learning curves, visualises the learned policy/path,
saves the Q-table and an animated GIF of the solution."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from maze_env import MazeEnv, ARROWS
from agent import train, greedy_path, moving_avg

OUT, MODELS = "outputs", "models"
PARAMS = dict(alpha=0.1, gamma=0.95, eps=1.0, eps_min=0.01, eps_decay=0.995)
EPISODES = 2000

env = MazeEnv()
print("Maze:\n" + env.render())
optimal = env.shortest_path(); opt_len = len(optimal) - 1
print(f"\nBFS shortest path length (ground truth): {opt_len} steps")

# ---- baseline: random agent (no learning)
rng = np.random.default_rng(1); wins = 0
for _ in range(200):
    env.reset(); done = False
    while not done:
        _, _, t, tr, _ = env.step(int(rng.integers(4))); done = t or tr
    wins += t
print(f"Random agent success rate: {wins / 200:.1%}")

# ---- train Q-learning
agent, hist = train(env, episodes=EPISODES, algo="q", seed=0, **PARAMS)
path, ok = greedy_path(env, agent.Q)
print(f"Q-learning greedy path: {len(path) - 1} steps, reached goal = {ok}")
print(f"Success rate (last 200 episodes, with exploration): {np.mean(hist['success'][-200:]):.1%}")

# ---- save the trained model (Q-table)
np.save(f"{MODELS}/q_table.npy", agent.Q)
json.dump({**PARAMS, "episodes": EPISODES, "learned_path_length": len(path) - 1, "optimal_path_length": opt_len,
           "reached_goal": bool(ok)}, open(f"{MODELS}/q_params.json", "w"), indent=2)

# ---- learning curves
fig, ax = plt.subplots(2, 2, figsize=(13, 8)); ep = np.arange(1, EPISODES + 1)
ax[0, 0].plot(ep, hist["reward"], alpha=.25); ax[0, 0].plot(ep, moving_avg(hist["reward"]), lw=2, c="C3")
ax[0, 0].set(title="Total reward per episode", xlabel="Episode", ylabel="Reward")
ax[0, 1].plot(ep, hist["steps"], alpha=.25); ax[0, 1].plot(ep, moving_avg(hist["steps"]), lw=2, c="C3")
ax[0, 1].set(title="Steps per episode", xlabel="Episode", ylabel="Steps"); ax[0, 1].set_yscale("log")
ax[1, 0].plot(ep, moving_avg(hist["success"], 100) * 100, c="C2"); ax[1, 0].set(title="Success rate (100-episode average)", xlabel="Episode", ylabel="% episodes reaching goal")
ax[1, 1].plot(ep, hist["epsilon"], label="epsilon (exploration)"); ax[1, 1].plot(ep, moving_avg(hist["td"]), label="mean |TD error|")
ax[1, 1].set(title="Exploration decay & TD error", xlabel="Episode"); ax[1, 1].legend()
for a in ax.ravel(): a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/01_learning_curves.png", dpi=110); plt.close()

# ---- maze + learned path + policy arrows
cmap = ListedColormap(["white", "#334155", "#fca5a5"])
img = np.zeros(env.grid.shape); img[env.grid == "#"] = 1; img[env.grid == "T"] = 2
def draw_maze(ax, title):
    ax.imshow(img, cmap=cmap); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    ax.text(env.start[1], env.start[0], "S", ha="center", va="center", fontweight="bold", color="green", fontsize=13)
    ax.text(env.goal[1], env.goal[0], "G", ha="center", va="center", fontweight="bold", color="red", fontsize=13)
fig, ax = plt.subplots(1, 3, figsize=(17, 5.8))
draw_maze(ax[0], f"Learned path ({len(path) - 1} steps; optimal = {opt_len})")
ax[0].plot([p[1] for p in path], [p[0] for p in path], "o-", c="#2563eb", lw=3, ms=5)
draw_maze(ax[1], "Learned policy (best action per cell)")
for r in range(env.rows):
    for c in range(env.cols):
        if env.grid[r, c] in " ST":                       # arrows only on walkable cells
            s = env.to_state((r, c)); ax[1].text(c, r, ARROWS[int(np.argmax(agent.Q[s]))], ha="center", va="center", fontsize=14, color="#1d4ed8")
V = agent.Q.max(1).reshape(env.grid.shape); V = np.ma.masked_where(env.grid == "#", V)
im = ax[2].imshow(V, cmap="viridis"); ax[2].set_title("State value  max Q(s, a)"); ax[2].set_xticks([]); ax[2].set_yticks([]); plt.colorbar(im, ax=ax[2], fraction=.046)
plt.tight_layout(); plt.savefig(f"{OUT}/02_maze_solution.png", dpi=110); plt.close()

# ---- before vs after training
fig, ax = plt.subplots(figsize=(5.5, 4)); ax.bar(["Random agent", "Q-learning agent"], [wins / 200 * 100, np.mean(hist["success"][-200:]) * 100], color=["#94a3b8", "#22c55e"])
ax.set(ylabel="Success rate (%)", title="Random vs trained agent"); plt.tight_layout(); plt.savefig(f"{OUT}/03_random_vs_qlearning.png", dpi=110); plt.close()

# ---- animation of the learned solution
fig, ax = plt.subplots(figsize=(5.5, 5.5)); draw_maze(ax, "Q-learning agent solving the maze")
dot, = ax.plot([], [], "o", c="#2563eb", ms=14); trail, = ax.plot([], [], "-", c="#93c5fd", lw=3)
def upd(i):
    xs, ys = [p[1] for p in path[:i + 1]], [p[0] for p in path[:i + 1]]
    dot.set_data([xs[-1]], [ys[-1]]); trail.set_data(xs, ys); return dot, trail
FuncAnimation(fig, upd, frames=len(path), interval=250).save(f"{OUT}/04_solution.gif", writer=PillowWriter(fps=4)); plt.close()

print(f"\nSaved: models/q_table.npy, plots + GIF in {OUT}/")
