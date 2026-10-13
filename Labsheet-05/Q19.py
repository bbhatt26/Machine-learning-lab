# Q19: Compare FrozenLake and Grid World environments
import gymnasium as gym
import numpy as np
from common import train_q_learning
from Q16 import GridWorld
lake = gym.make("FrozenLake-v1", is_slippery=False)
q_lake, rewards_lake = train_q_learning(lake, episodes=3000)
print("FrozenLake success rate (last 100 training episodes):", np.mean(rewards_lake[-100:]))
lake.close()
env = GridWorld()
q = np.zeros((16,4)); rewards_grid=[]
for ep in range(2000):
    s=env.reset(); total=0; eps=max(.05, 1*(.995**ep))
    for _ in range(100):
        a=np.random.randint(4) if np.random.random()<eps else int(np.argmax(q[s]))
        ns,r,done=env.step(a)
        q[s,a]+=.2*(r+(0 if done else .95*np.max(q[ns]))-q[s,a])
        s=ns; total+=r
        if done: break
    rewards_grid.append(total)
print("Grid World average reward (last 100 episodes):", np.mean(rewards_grid[-100:]))
