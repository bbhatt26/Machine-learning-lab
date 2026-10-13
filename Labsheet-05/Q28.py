# Q28: Study the role of the target network in DQN
import matplotlib.pyplot as plt
import numpy as np
from dqn_common import train_dqn
for use_target in [True, False]:
    _, rewards = train_dqn(episodes=150, seed=42, use_target=use_target)
    avg = np.convolve(rewards, np.ones(10)/10, mode="valid")
    plt.plot(avg, label=f"target network={use_target}")
plt.xlabel("Episode"); plt.ylabel("10-episode mean reward")
plt.title("Effect of Target Network"); plt.legend(); plt.tight_layout(); plt.show()
