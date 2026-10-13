# Q30: Load saved DQN model and test it
import torch
import gymnasium as gym
from dqn_common import DQN
model = DQN(4, 2)
model.load_state_dict(torch.load("cartpole_dqn.pth", map_location="cpu", weights_only=True))
model.eval()
env = gym.make("CartPole-v1")
scores=[]
for ep in range(5):
    state, _ = env.reset(seed=200+ep)
    total, done = 0, False
    while not done:
        with torch.no_grad():
            action = int(model(torch.tensor(state, dtype=torch.float32)).argmax().item())
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward; done = terminated or truncated
    scores.append(total)
print("Test rewards:", scores)
print("Average test reward:", sum(scores)/len(scores))
env.close()
