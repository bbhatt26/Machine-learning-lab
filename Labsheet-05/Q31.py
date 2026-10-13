# Q31: Compare cumulative rewards from different RL algorithms
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from common import train_q_learning, moving_average
from dqn_common import train_dqn
env = gym.make("FrozenLake-v1", is_slippery=False)
_, q_rewards = train_q_learning(env, episodes=3000)
env.close()
_, dqn_rewards = train_dqn(episodes=150)
plt.plot(np.cumsum(q_rewards), label="Q-Learning cumulative reward (FrozenLake)")
plt.plot(np.cumsum(dqn_rewards), label="DQN cumulative reward (CartPole)")
plt.xlabel("Episode"); plt.ylabel("Cumulative reward")
plt.title("Cumulative Rewards (different environments; interpret cautiously)")
plt.legend(); plt.tight_layout(); plt.show()
