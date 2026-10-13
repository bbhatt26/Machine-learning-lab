# Q14: Compare exploration and exploitation with different epsilon values
import gymnasium as gym
import matplotlib.pyplot as plt
from common import train_q_learning, moving_average
env = gym.make("FrozenLake-v1", is_slippery=False)
for epsilon_start in [0.2, 0.6, 1.0]:
    _, rewards = train_q_learning(env, episodes=2500, epsilon=epsilon_start, epsilon_decay=0.998)
    plt.plot(moving_average(rewards, 100), label=f"starting epsilon={epsilon_start}")
plt.xlabel("Episode"); plt.ylabel("100-episode average reward")
plt.title("Exploration vs Exploitation"); plt.legend(); plt.tight_layout(); plt.show()
env.close()
