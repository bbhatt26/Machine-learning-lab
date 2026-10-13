# Q32: Visualize the learning curve of an RL agent
import gymnasium as gym
import matplotlib.pyplot as plt
from common import train_q_learning, moving_average
env = gym.make("FrozenLake-v1", is_slippery=False)
_, rewards = train_q_learning(env, episodes=3000)
plt.plot(rewards, alpha=0.25, label="Raw reward")
plt.plot(moving_average(rewards, 100), label="100-episode moving average")
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("Q-Learning Learning Curve")
plt.legend(); plt.tight_layout(); plt.show()
env.close()
