# Q34: Analyze impact of hyperparameters on learning performance
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from common import train_q_learning
env = gym.make("FrozenLake-v1", is_slippery=False)
settings = [
    {"learning_rate": 0.1, "gamma": 0.9},
    {"learning_rate": 0.5, "gamma": 0.9},
    {"learning_rate": 0.8, "gamma": 0.99},
]
for setting in settings:
    _, rewards = train_q_learning(env, episodes=2500,
                                  learning_rate=setting["learning_rate"],
                                  gamma=setting["gamma"],
                                  epsilon_decay=0.998)
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode="valid"),
             label=f"alpha={setting['learning_rate']}, gamma={setting['gamma']}")
plt.xlabel("Episode"); plt.ylabel("100-episode mean reward")
plt.title("Hyperparameter Comparison"); plt.legend(); plt.tight_layout(); plt.show()
env.close()
