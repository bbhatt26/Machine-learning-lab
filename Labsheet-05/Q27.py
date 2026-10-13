# Q27: Analyze effect of replay memory capacity on DQN
import matplotlib.pyplot as plt
import numpy as np
from dqn_common import train_dqn
for capacity in [1000, 10000]:
    _, rewards = train_dqn(episodes=150, replay_capacity=capacity)
    plt.plot(np.convolve(rewards, np.ones(10)/10, mode="valid"), label=f"memory={capacity}")
plt.xlabel("Episode"); plt.ylabel("10-episode mean reward")
plt.title("Replay Memory Capacity Effect"); plt.legend(); plt.tight_layout(); plt.show()
