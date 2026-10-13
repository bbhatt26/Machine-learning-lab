# Q7: Initialize and update a Q-table during training
import gymnasium as gym
import numpy as np
env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros((env.observation_space.n, env.action_space.n))
learning_rate, gamma, epsilon = 0.8, 0.95, 0.2
state, _ = env.reset(seed=7)
action = env.action_space.sample() if np.random.random() < epsilon else int(np.argmax(q_table[state]))
next_state, reward, terminated, truncated, _ = env.step(action)
old_value = q_table[state, action]
next_max = 0 if terminated else np.max(q_table[next_state])
q_table[state, action] += learning_rate * (reward + gamma * next_max - old_value)
print("Q-table before update: all values initialized to 0")
print("State/action updated:", state, action)
print("Reward:", reward)
print("Updated Q-table:\n", q_table)
env.close()
