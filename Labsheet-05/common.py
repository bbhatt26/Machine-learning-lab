"""Shared helpers for Lab Sheet 05."""
import numpy as np

def train_q_learning(env, episodes=5000, learning_rate=0.8, gamma=0.95,
                     epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999):
    """Train tabular Q-learning; returns Q-table and episode rewards."""
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            target = reward
            if not terminated:
                target += gamma * np.max(q_table[next_state])
            q_table[state, action] += learning_rate * (
                target - q_table[state, action]
            )
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    return q_table, rewards

def moving_average(values, window=100):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")
