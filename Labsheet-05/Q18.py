# Q18: Visualize the optimal path learned by the Grid World agent
import numpy as np
import matplotlib.pyplot as plt
from Q16 import GridWorld
env = GridWorld()
q = np.zeros((16, 4))
for episode in range(2500):
    state = env.reset()
    epsilon = max(0.05, 1.0 * (0.995 ** episode))
    for _ in range(100):
        action = np.random.randint(4) if np.random.random() < epsilon else int(np.argmax(q[state]))
        ns, reward, done = env.step(action)
        q[state, action] += 0.2 * (reward + (0 if done else 0.95*np.max(q[ns])) - q[state, action])
        state = ns
        if done: break
state = env.reset()
path = [env.state]
for _ in range(20):
    action = int(np.argmax(q[state]))
    state, _, done = env.step(action)
    path.append(env.state)
    if done: break
grid = np.zeros((4,4))
grid[1,1] = -1
grid[3,3] = 2
for r,c in path: grid[r,c] = 1
grid[3,3] = 2
plt.imshow(grid, cmap="coolwarm")
for r in range(4):
    for c in range(4):
        label = "S" if (r,c)==(0,0) else "G" if (r,c)==(3,3) else "X" if (r,c)==(1,1) else ""
        if (r,c) in path and label == "": label = "•"
        plt.text(c, r, label, ha="center", va="center", fontsize=14)
plt.title("Learned Grid World Path (S=start, G=goal, X=blocked)")
plt.xticks(range(4)); plt.yticks(range(4)); plt.tight_layout(); plt.show()
print("Path:", path)
