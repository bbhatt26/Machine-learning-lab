# Q17: Train a Q-Learning agent in custom Grid World
import numpy as np
from Q16 import GridWorld
env = GridWorld()
q = np.zeros((16, 4))
alpha, gamma, epsilon = 0.2, 0.95, 1.0
rewards = []
for episode in range(2000):
    state = env.reset()
    total, done = 0, False
    for _ in range(100):
        action = np.random.randint(4) if np.random.random() < epsilon else int(np.argmax(q[state]))
        next_state, reward, done = env.step(action)
        q[state, action] += alpha * (reward + (0 if done else gamma * np.max(q[next_state])) - q[state, action])
        state, total = next_state, total + reward
        if done:
            break
    rewards.append(total)
    epsilon = max(0.05, epsilon * 0.995)
print("Average reward over last 100 episodes:", np.mean(rewards[-100:]))
print("Learned Q-table:\n", np.round(q, 2))
