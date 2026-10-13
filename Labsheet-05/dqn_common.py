"""Small educational DQN implementation for CartPole using PyTorch."""
import random
from collections import deque
import numpy as np
import torch
from torch import nn
import gymnasium as gym

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    def add(self, transition):
        self.memory.append(transition)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        return self.net(x)

def train_dqn(episodes=250, seed=42, replay_capacity=10000, use_target=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    state, _ = env.reset(seed=seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy = DQN(state_size, action_size)
    target = DQN(state_size, action_size)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    memory = ReplayMemory(replay_capacity)
    gamma, batch_size = 0.99, 64
    epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.985
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        total, done = 0, False
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(policy(torch.tensor(state, dtype=torch.float32)).argmax().item())
            ns, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.add((state, action, reward, ns, terminated))
            state = ns; total += reward
            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.int64).unsqueeze(1)
                rs = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
                terms = torch.tensor([b[4] for b in batch], dtype=torch.float32)
                current = policy(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_values = target(next_states).max(1).values if use_target else policy(next_states).max(1).values
                    expected = rs + gamma * next_values * (1 - terms)
                loss = nn.functional.smooth_l1_loss(current, expected)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 10)
                optimizer.step()
        rewards.append(total)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if use_target and (episode + 1) % 10 == 0:
            target.load_state_dict(policy.state_dict())
    env.close()
    return policy, rewards
