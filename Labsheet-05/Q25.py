# Q25: Evaluate the trained DQN agent
import torch
import gymnasium as gym
from dqn_common import train_dqn
model, training_rewards = train_dqn(episodes=250)
env = gym.make("CartPole-v1")
scores=[]
for ep in range(10):
    state, _ = env.reset(seed=100+ep)
    total, done = 0, False
    while not done:
        with torch.no_grad():
            action = int(model(torch.tensor(state, dtype=torch.float32)).argmax().item())
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward; done = terminated or truncated
    scores.append(total)
print("Evaluation rewards:", scores)
print("Mean evaluation reward:", sum(scores)/len(scores))
env.close()
