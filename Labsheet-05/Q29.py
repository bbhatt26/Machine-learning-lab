# Q29: Save a trained DQN model
import torch
from dqn_common import train_dqn
model, rewards = train_dqn(episodes=250)
torch.save(model.state_dict(), "cartpole_dqn.pth")
print("Model weights saved to cartpole_dqn.pth")
print("Last-20 average reward:", sum(rewards[-20:])/20)
