"""Generate a synthetic retail sales dataset (with missing values and duplicates)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
N = 6000

products = {  # category: {sub_category: [(product, price)]}
    "Technology": {"Phones": [("Smartphone X", 18000), ("Smartphone Lite", 9000)],
                   "Laptops": [("Laptop Pro", 55000), ("Laptop Air", 38000)],
                   "Accessories": [("Wireless Mouse", 800), ("Headphones", 2500)]},
    "Furniture": {"Chairs": [("Ergo Chair", 7500), ("Office Chair", 4200)],
                  "Tables": [("Study Table", 6000), ("Conference Table", 15000)]},
    "Office Supplies": {"Paper": [("A4 Paper Ream", 350), ("Notebook Pack", 450)],
                        "Storage": [("File Cabinet", 5200), ("Storage Box", 700)]},
}
flat = [(c, s, p, pr) for c, d in products.items() for s, l in d.items() for p, pr in l]
regions = {"North": ["Delhi", "Uttarakhand", "Punjab"], "South": ["Karnataka", "Tamil Nadu", "Kerala"],
           "East": ["West Bengal", "Odisha", "Bihar"], "West": ["Maharashtra", "Gujarat", "Rajasthan"]}
reg_w = [0.28, 0.27, 0.17, 0.28]
segments = ["Consumer", "Corporate", "Home Office"]

dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, N), unit="D")
keep = rng.random(N) < np.where(dates.month >= 10, 1.0, 0.75)   # seasonality: more orders Oct-Dec
dates = dates[keep]; n = len(dates)

idx = rng.integers(0, len(flat), n)
region = rng.choice(list(regions), n, p=reg_w)
df = pd.DataFrame({
    "order_id": [f"ORD-{100000 + i}" for i in range(n)],
    "order_date": dates,
    "customer_id": [f"CUST-{x:04d}" for x in rng.integers(1, 800, n)],
    "segment": rng.choice(segments, n, p=[.5, .3, .2]),
    "region": region,
    "state": [rng.choice(regions[r]) for r in region],
    "category": [flat[i][0] for i in idx], "sub_category": [flat[i][1] for i in idx],
    "product": [flat[i][2] for i in idx], "unit_price": [flat[i][3] for i in idx],
    "quantity": rng.integers(1, 6, n),
    "discount": rng.choice([0, .05, .10, .20, .30], n, p=[.4, .2, .2, .12, .08]),
})
df["sales"] = (df.unit_price * df.quantity * (1 - df.discount)).round(2)
margin = df.category.map({"Technology": .18, "Furniture": .12, "Office Supplies": .25})
df["profit"] = (df.sales * (margin - df.discount * 0.9 + rng.normal(0, .03, n))).round(2)
df = df.sort_values("order_date").reset_index(drop=True)

# inject missing values + duplicates to practise cleaning
for col in ["discount", "quantity", "segment", "profit"]:
    df.loc[rng.choice(n, int(.02 * n), replace=False), col] = np.nan
df = pd.concat([df, df.sample(20, random_state=1)]).reset_index(drop=True)
df.to_csv("data/sales_data.csv", index=False)
print("Saved data/sales_data.csv", df.shape)
