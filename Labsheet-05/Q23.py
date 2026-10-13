# Q23: Train a DQN agent on CartPole
from dqn_common import train_dqn
model, rewards = train_dqn(episodes=250)
print("Episodes trained:", len(rewards))
print("Average reward over last 20 episodes:", sum(rewards[-20:])/min(20, len(rewards)))
