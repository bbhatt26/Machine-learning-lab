# Q5: Simulate random actions in FrozenLake
import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=False)
state, _ = env.reset(seed=42)
total_reward = 0
for step in range(1, 101):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    print(f"Step {step}: state={state}, action={action}, next={next_state}, reward={reward}")
    total_reward += reward
    state = next_state
    if terminated or truncated:
        break
print("Total reward:", total_reward)
print("Total steps:", step)
env.close()
