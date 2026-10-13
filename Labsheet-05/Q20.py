# Q20: Analyze convergence behavior of Q-Learning
import gymnasium as gym
import matplotlib.pyplot as plt
from common import train_q_learning, moving_average
env = gym.make("FrozenLake-v1", is_slippery=False)
q, rewards = train_q_learning(env, episodes=5000)
plt.plot(moving_average(rewards, 100))
plt.xlabel("Episode"); plt.ylabel("Average reward (100 episodes)")
plt.title("Q-Learning Convergence")
plt.tight_layout(); plt.show()
print("Final 100-episode average:", sum(rewards[-100:])/100)
env.close()
