# Q15: Implement an epsilon-greedy action selection policy
import numpy as np
q_values = np.array([0.1, 0.8, 0.3, 0.2])
epsilon = 0.2
if np.random.random() < epsilon:
    chosen_action = np.random.randint(len(q_values))  # Explore
    decision = "Exploration: random action"
else:
    chosen_action = int(np.argmax(q_values))  # Exploit
    decision = "Exploitation: best known action"
print("Q-values:", q_values)
print("Epsilon:", epsilon)
print(decision)
print("Chosen action:", chosen_action)
