# Q10: Evaluate a trained Q-Learning agent
import gymnasium as gym
import numpy as np
from common import train_q_learning
env = gym.make("FrozenLake-v1", is_slippery=False)
q_table, _ = train_q_learning(env, episodes=4000)
test_rewards = []
for episode in range(100):
    state, _ = env.reset()
    total, done = 0, False
    while not done:
        action = int(np.argmax(q_table[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
    test_rewards.append(total)
print("Evaluation episodes:", len(test_rewards))
print("Success rate:", np.mean(test_rewards))
env.close()
