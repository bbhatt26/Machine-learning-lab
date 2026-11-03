"""Tabular Q-Learning (and SARSA for comparison) agent."""
import numpy as np

class QAgent:
    def __init__(self, n_states, n_actions=4, alpha=0.1, gamma=0.95, eps=1.0, eps_min=0.01, eps_decay=0.995, seed=0):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma, self.eps, self.eps_min, self.eps_decay = alpha, gamma, eps, eps_min, eps_decay
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:                  # explore
            return int(self.rng.integers(self.n_actions))
        best = np.flatnonzero(self.Q[s] == self.Q[s].max())             # exploit (random tie-break)
        return int(self.rng.choice(best))

    def update(self, s, a, r, s2, terminated, a2=None, algo="q"):
        """Q-learning:  Q(s,a) += alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
           SARSA:       Q(s,a) += alpha * [ r + gamma * Q(s',a')        - Q(s,a) ]"""
        nxt = 0.0 if terminated else (self.Q[s2].max() if algo == "q" else self.Q[s2, a2])
        td_error = r + self.gamma * nxt - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        return td_error

    def decay(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


def train(env, episodes=2000, algo="q", seed=0, **kw):
    """Train an agent; returns (agent, history dict)."""
    agent = QAgent(env.observation_space.n, seed=seed, **kw)
    hist = dict(reward=[], steps=[], success=[], epsilon=[], td=[])
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep); a = agent.act(s)
        total, td_sum, done = 0.0, 0.0, False
        while not done:
            s2, r, term, trunc, _ = env.step(a)
            a2 = agent.act(s2)                                        # needed by SARSA, harmless for Q-learning
            td_sum += abs(agent.update(s, a, r, s2, term, a2, algo))
            s, a, total, done = s2, a2, total + r, term or trunc
        agent.decay()
        hist["reward"].append(total); hist["steps"].append(env.steps); hist["success"].append(int(term))
        hist["epsilon"].append(agent.eps); hist["td"].append(td_sum / env.steps)
    return agent, hist


def greedy_path(env, Q, max_steps=200):
    """Follow the learned greedy policy; returns (list of (row,col), reached_goal)."""
    s, _ = env.reset(); path = [env.to_pos(s)]
    for _ in range(max_steps):
        s, _, term, trunc, _ = env.step(int(np.argmax(Q[s]))); path.append(env.to_pos(s))
        if term: return path, True
        if trunc: break
    return path, False


def moving_avg(x, w=50):
    x = np.asarray(x, float); c = np.cumsum(np.insert(x, 0, 0))
    return np.concatenate([c[1:w] / np.arange(1, w), (c[w:] - c[:-w]) / w]) if len(x) >= w else x
