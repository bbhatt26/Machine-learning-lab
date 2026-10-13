# Q2: Create and execute a simple RL environment
import gymnasium as gym
env = gym.make("FrozenLake-v1")
state, info = env.reset(seed=42)
print("Initial state:", state)
print("Additional information:", info)
env.close()
