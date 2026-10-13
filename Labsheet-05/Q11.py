# Q11: Plot cumulative/episode rewards during training
import gymnasium as gym
import matplotlib.pyplot as plt
from common import train_q_learning, moving_average
env = gym.make("FrozenLake-v1", is_slippery=False)
q_table, rewards = train_q_learning(env, episodes=3000)
plt.plot(rewards, alpha=0.35, label="Episode reward")
plt.plot(range(len(moving_average(rewards, 100))), moving_average(rewards, 100), label="100-episode average")
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("Q-Learning Rewards")
plt.legend(); plt.tight_layout(); plt.show()
env.close()
