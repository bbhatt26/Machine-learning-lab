"""Tune alpha, gamma, epsilon-decay (3 seeds each) and compare Q-learning with SARSA."""
import itertools, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from maze_env import MazeEnv
from agent import train, greedy_path, moving_avg

EPISODES, SEEDS = 1000, [0, 1, 2]
env = MazeEnv(); opt = len(env.shortest_path()) - 1
base = dict(alpha=0.1, gamma=0.95, eps_decay=0.995)
sweeps = {"alpha": [0.01, 0.1, 0.5, 0.9], "gamma": [0.5, 0.8, 0.95, 0.99], "eps_decay": [0.95, 0.99, 0.995, 0.999]}

def run(algo, **kw):
    res, curves = [], []
    for sd in SEEDS:
        ag, h = train(env, episodes=EPISODES, algo=algo, seed=sd, **kw)
        p, ok = greedy_path(env, ag.Q)
        ma = moving_avg(h["reward"], 50); hit = np.flatnonzero(ma >= 75)      # 75 ~ near-optimal reward (~82)
        res.append(dict(final_reward=np.mean(h["reward"][-100:]), success=np.mean(h["success"][-100:]),
                        mean_reward_all=np.mean(h["reward"]), converge_ep=hit[0] + 1 if len(hit) else EPISODES,
                        path_len=(len(p) - 1) if ok else np.nan, solved=ok))
        curves.append(moving_avg(h["reward"], 50))
    d = pd.DataFrame(res)
    return dict(episodes_to_converge=d.converge_ep.mean(), mean_reward_all_episodes=d.mean_reward_all.mean(),
                avg_reward_last100=d.final_reward.mean(), success_last100=d.success.mean(),
                solved_runs=f"{int(d.solved.sum())}/{len(SEEDS)}", avg_path_len=d.path_len.mean(), optimal=opt), np.mean(curves, 0)

rows, fig_data = [], {}
for name, values in sweeps.items():
    fig_data[name] = {}
    for v in values:
        r, c = run("q", **{**base, name: v}); rows.append(dict(parameter=name, value=v, **r)); fig_data[name][v] = c
        print(f"{name}={v:<6} converge@{r['episodes_to_converge']:6.0f} ep | mean reward(all)={r['mean_reward_all_episodes']:7.1f} | final reward={r['avg_reward_last100']:6.1f} success={r['success_last100']:.0%} solved={r['solved_runs']} path={r['avg_path_len']}")

df = pd.DataFrame(rows); df.to_csv("outputs/hyperparameter_results.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for a, (name, d) in zip(ax, fig_data.items()):
    for v, c in d.items(): a.plot(c, label=f"{name}={v}")
    a.set(title=f"Effect of {name}", xlabel="Episode", ylabel="Reward (50-ep average)"); a.legend(fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/05_hyperparameter_tuning.png", dpi=110); plt.close()

# ---- Q-learning vs SARSA
cmp = {}
for algo in ["q", "sarsa"]:
    r, c = run(algo, **base); cmp[algo] = c; print(algo.upper(), {k: (round(float(v), 1) if not isinstance(v, str) else v) for k, v in r.items()})
plt.figure(figsize=(8, 4.5))
plt.plot(cmp["q"], label="Q-learning (off-policy)"); plt.plot(cmp["sarsa"], label="SARSA (on-policy)")
plt.xlabel("Episode"); plt.ylabel("Reward (50-ep average)"); plt.title("Q-learning vs SARSA"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/06_qlearning_vs_sarsa.png", dpi=110); plt.close()
best = df.sort_values("mean_reward_all_episodes", ascending=False).iloc[0]
print(f"\nBest single setting: {best.parameter}={best.value}")
