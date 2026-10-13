# Q35: Prepare a comparative report summarizing RL algorithm performance
# Run the experiments first. This script summarizes measured results.
import time
import numpy as np
import gymnasium as gym
from common import train_q_learning
from dqn_common import train_dqn

start = time.perf_counter()
env = gym.make("FrozenLake-v1", is_slippery=False)
_, q_rewards = train_q_learning(env, episodes=3000)
q_time = time.perf_counter() - start
env.close()

start = time.perf_counter()
_, dqn_rewards = train_dqn(episodes=150)
dqn_time = time.perf_counter() - start

print("\\nCOMPARATIVE REPORT: REINFORCEMENT LEARNING")
print("-" * 48)
print("Objective: Compare tabular Q-Learning with a Deep Q-Network.")
print("\\nAlgorithm / Procedure:")
print("1. Train Q-Learning on deterministic FrozenLake.")
print("2. Train DQN on CartPole using replay memory and a target network.")
print("\\nMeasured results:")
print(f"Q-Learning training time: {q_time:.2f} seconds")
print(f"Q-Learning last-100 success rate: {np.mean(q_rewards[-100:]):.3f}")
print(f"DQN training time: {dqn_time:.2f} seconds")
print(f"DQN last-20 mean episode reward: {np.mean(dqn_rewards[-20:]):.2f}")
print("\\nPerformance Analysis:")
print("Q-Learning stores action values in a table and suits small discrete state spaces.")
print("DQN approximates action values with a neural network and suits larger state spaces.")
print("The environments differ, so reward magnitudes should not be directly compared.")
print("\\nObservation: Results can vary with random seeds, versions and hardware.")
print("Conclusion: Choose the algorithm based on state-space size and task requirements.")
