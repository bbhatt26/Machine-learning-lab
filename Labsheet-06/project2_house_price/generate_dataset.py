"""Generates a synthetic house dataset -> house_prices.csv  (price in lakh INR)"""
import numpy as np, pandas as pd
rng = np.random.default_rng(5)
n = 800
loc_factor = {"Downtown": 1.45, "Suburb": 1.0, "Riverside": 1.25, "Outskirts": 0.7}
location = rng.choice(list(loc_factor), n, p=[.2, .4, .2, .2])
area = rng.normal(1500, 450, n).clip(500, 3500)
bedrooms = np.clip((area / 600 + rng.normal(0, .6, n)).round(), 1, 6).astype(int)
bathrooms = np.clip((bedrooms - rng.integers(0, 2, n)).clip(1), 1, 5).astype(int)
parking = rng.choice([0, 1, 2, 3], n, p=[.15, .4, .35, .1])
age = rng.integers(0, 40, n)
price = (area * 0.045 * np.array([loc_factor[l] for l in location]) + bedrooms * 4 + bathrooms * 3
         + parking * 5 - age * 0.6 + rng.normal(0, 6, n)).clip(15)
df = pd.DataFrame({"area_sqft": area.round(0), "bedrooms": bedrooms, "bathrooms": bathrooms,
                   "location": location, "parking_spaces": parking, "age_years": age, "price_lakhs": price.round(2)})
df.loc[rng.choice(n, 15, replace=False), "area_sqft"] = np.nan
df.loc[rng.choice(n, 10, replace=False), "location"] = np.nan
df = pd.concat([df, df.sample(6, random_state=1)], ignore_index=True)
df.to_csv("house_prices.csv", index=False); print(df.shape)
