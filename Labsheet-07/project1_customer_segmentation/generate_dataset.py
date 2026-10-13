"""Generates a realistic synthetic customer dataset (mall-style) -> customers.csv"""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
# (n, age mean/sd, income mean/sd, spend mean/sd, freq mean/sd)
profiles = [
    (60, (25, 4), (30, 6), (78, 8), (30, 5)),    # young, low income, high spenders
    (60, (42, 6), (85, 10), (80, 8), (35, 5)),   # affluent, high spenders
    (60, (45, 7), (85, 10), (20, 8), (8, 3)),    # affluent, frugal
    (60, (50, 8), (30, 6), (22, 8), (6, 3)),     # older, low income, low spend
    (60, (33, 6), (55, 8), (50, 8), (18, 4)),    # average
]
rows = []
for n, a, i, s, f in profiles:
    rows.append(pd.DataFrame({
        "Age": rng.normal(*a, n).round().clip(18, 70),
        "Annual_Income_k": rng.normal(*i, n).round().clip(10, 140),
        "Spending_Score": rng.normal(*s, n).round().clip(1, 100),
        "Purchase_Frequency": rng.normal(*f, n).round().clip(1, 60),
    }))
df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)
df.insert(0, "CustomerID", range(1, len(df) + 1))
df.insert(2, "Gender", rng.choice(["Male", "Female"], len(df)))
# inject a few data-quality problems so the cleaning step is meaningful
df.loc[rng.choice(len(df), 6, replace=False), "Annual_Income_k"] = np.nan
df = pd.concat([df, df.sample(4, random_state=3)], ignore_index=True)
df.to_csv("customers.csv", index=False)
print(df.shape)
