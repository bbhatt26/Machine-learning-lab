# Q1: Install and configure Gymnasium
# Install first: python -m pip install "gymnasium[toy-text]"
import gymnasium as gym
print("Gymnasium version:", gym.__version__)
env = gym.make("FrozenLake-v1")
print("Environment created successfully:", env)
env.close()
