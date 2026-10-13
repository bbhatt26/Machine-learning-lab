# Q13: Study different discount factor (gamma) values
import gymnasium as gym
import matplotlib.pyplot as plt
from common import train_q_learning, moving_average
env = gym.make("FrozenLake-v1", is_slippery=False)
for gamma in [0.5, 0.8, 0.99]:
    _, rewards = train_q_learning(env, episodes=2500, gamma=gamma, epsilon_decay=0.998)
    plt.plot(moving_average(rewards, 100), label=f"gamma={gamma}")
plt.xlabel("Episode"); plt.ylabel("100-episode average reward")
plt.title("Effect of Discount Factor"); plt.legend(); plt.tight_layout(); plt.show()
env.close()
