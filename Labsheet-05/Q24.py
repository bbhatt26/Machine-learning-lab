# Q24: Plot episode-wise reward during DQN training
import matplotlib.pyplot as plt
from dqn_common import train_dqn
model, rewards = train_dqn(episodes=250)
plt.plot(rewards, label="Episode reward")
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("DQN on CartPole")
plt.legend(); plt.tight_layout(); plt.show()
