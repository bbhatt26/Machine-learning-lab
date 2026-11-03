"""Generate a realistic synthetic COVID-19 dataset (with missing values)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)

countries = {  # name: (continent, population, severity, vaccination speed)
    "India": ("Asia", 1_400_000_000, 1.0, 1.0), "USA": ("North America", 330_000_000, 1.6, 0.9),
    "Brazil": ("South America", 213_000_000, 1.3, 0.8), "UK": ("Europe", 67_000_000, 1.2, 1.1),
    "France": ("Europe", 65_000_000, 1.2, 1.0), "Germany": ("Europe", 83_000_000, 1.0, 1.0),
    "Italy": ("Europe", 60_000_000, 1.1, 1.0), "Japan": ("Asia", 125_000_000, 0.5, 1.1),
    "South Africa": ("Africa", 60_000_000, 0.7, 0.5), "Australia": ("Oceania", 26_000_000, 0.4, 1.1),
}
dates = pd.date_range("2020-03-01", "2022-03-01")
t = np.arange(len(dates))
def wave(c, w, h): return h * np.exp(-0.5 * ((t - c) / w) ** 2)

rows = []
for name, (cont, pop, sev, vsp) in countries.items():
    shift = rng.integers(-25, 25, 3)
    base = wave(60 + shift[0], 35, 1) + wave(300 + shift[1], 45, 2) + wave(600 + shift[2], 30, 3.5)
    cases = pop * 1e-4 * sev * base * rng.normal(1, 0.12, len(t)).clip(0.3)
    cases = np.maximum(cases, 0)
    deaths = np.roll(cases, 10) * rng.uniform(0.008, 0.02) * np.linspace(1.3, 0.4, len(t))
    recov = np.roll(cases, 14) * rng.uniform(0.85, 0.95)
    vacc = np.where(t > 290, pop * 0.0035 * vsp * np.exp(-((t - 450) / 150) ** 2) * rng.normal(1, 0.1, len(t)), 0)
    for i, d in enumerate(dates):
        rows.append([d, name, cont, pop, int(cases[i]), int(deaths[i]), int(recov[i]), int(max(vacc[i], 0))])

df = pd.DataFrame(rows, columns=["date", "country", "continent", "population",
                                 "new_cases", "new_deaths", "new_recovered", "new_vaccinations"])
# inject ~3% missing values + a few bad (negative) entries to practise cleaning
for col in ["new_cases", "new_deaths", "new_recovered", "new_vaccinations"]:
    df.loc[rng.choice(len(df), int(0.03 * len(df)), replace=False), col] = np.nan
df.loc[rng.choice(len(df), 15, replace=False), "new_cases"] = -1
df.to_csv("data/covid_data.csv", index=False)
print("Saved data/covid_data.csv", df.shape)
