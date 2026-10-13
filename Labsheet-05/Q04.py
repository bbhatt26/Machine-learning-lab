# Q4: Display state, action, reward and termination conditions
import gymnasium as gym
env = gym.make("FrozenLake-v1")
state, _ = env.reset(seed=42)
action = env.action_space.sample()
next_state, reward, terminated, truncated, info = env.step(action)
print("Current state:", state)
print("Action taken:", action, "(0=Left, 1=Down, 2=Right, 3=Up)")
print("Next state:", next_state)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
env.close()
