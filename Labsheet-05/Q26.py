# Q26: Compare Q-Learning and DQN performance
import gymnasium as gym
import numpy as np
from common import train_q_learning
from dqn_common import train_dqn
lake = gym.make("FrozenLake-v1", is_slippery=False)
_, q_rewards = train_q_learning(lake, episodes=3000)
lake.close()
dqn_model, dqn_rewards = train_dqn(episodes=150)
print("Q-Learning FrozenLake last-100 success rate:", np.mean(q_rewards[-100:]))
print("DQN CartPole last-20 mean reward:", np.mean(dqn_rewards[-20:]))
print("Note: environments and reward scales differ, so these values are not a direct apples-to-apples comparison.")
