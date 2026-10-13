# Q33: Compare training time and convergence of Q-Learning and DQN
import time
import gymnasium as gym
import numpy as np
from common import train_q_learning
from dqn_common import train_dqn
start = time.perf_counter()
env = gym.make("FrozenLake-v1", is_slippery=False)
_, qr = train_q_learning(env, episodes=3000)
q_time = time.perf_counter() - start
env.close()
start = time.perf_counter()
_, dr = train_dqn(episodes=150)
dqn_time = time.perf_counter() - start
print(f"Q-Learning time: {q_time:.2f} seconds")
print(f"DQN time: {dqn_time:.2f} seconds")
print("Q-Learning last-100 reward mean:", np.mean(qr[-100:]))
print("DQN last-20 reward mean:", np.mean(dr[-20:]))
print("Caution: these agents use different environments and episode counts.")
