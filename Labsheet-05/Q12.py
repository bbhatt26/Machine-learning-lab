# Q12: Study the effect of different learning rates
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from common import train_q_learning, moving_average
rates = [0.1, 0.5, 0.9]
env = gym.make("FrozenLake-v1", is_slippery=False)
for rate in rates:
    _, rewards = train_q_learning(env, episodes=2500, learning_rate=rate, epsilon_decay=0.998)
    avg = moving_average(rewards, 100)
    plt.plot(avg, label=f"alpha={rate}")
plt.xlabel("Episode"); plt.ylabel("100-episode average reward")
plt.title("Effect of Learning Rate"); plt.legend(); plt.tight_layout(); plt.show()
env.close()
