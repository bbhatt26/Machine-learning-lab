# Q22: Implement a basic Deep Q-Network using PyTorch
import torch
from dqn_common import DQN
model = DQN(input_size=4, output_size=2)
sample_state = torch.zeros(1, 4)
print(model)
print("Q-values for a sample state:", model(sample_state).detach().numpy())
