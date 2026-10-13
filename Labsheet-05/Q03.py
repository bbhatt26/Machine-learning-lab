# Q3: Explore observation space and action space
import gymnasium as gym
env = gym.make("FrozenLake-v1")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Number of states:", env.observation_space.n)
print("Number of actions:", env.action_space.n)
env.close()
