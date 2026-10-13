# %% [markdown]
# # Project 2: House Price Prediction System
# **MCA201P - Lab Sheet 06 | MCA III Semester (2026-2027)**
#
# Regression model to predict house price (in lakh INR) from area, bedrooms, bathrooms, location,
# parking spaces and age of the property. Several regression algorithms are compared.

# %%
import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
os.makedirs("outputs", exist_ok=True); os.makedirs("model", exist_ok=True)
if not os.path.exists("house_prices.csv"):
    import generate_dataset

# %% [markdown]
# ## 1. Load and explore the data

# %%
df = pd.read_csv("house_prices.csv")
print("Shape:", df.shape); print(df.head())
print("\nMissing values:\n", df.isna().sum()); print("Duplicates:", df.duplicated().sum())
print(df.describe().round(2))

# %% [markdown]
# ## 2. Preprocessing
# * remove duplicates
# * numeric columns: median imputation + standard scaling
# * categorical column (`location`): most-frequent imputation + one-hot encoding
# * all done inside a `ColumnTransformer` pipeline fitted only on training data (no data leakage)

# %%
df = df.drop_duplicates().reset_index(drop=True)
target = "price_lakhs"
num_cols = ["area_sqft", "bedrooms", "bathrooms", "parking_spaces", "age_years"]
cat_cols = ["location"]
X, y = df[num_cols + cat_cols], df[target]
preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

# %% [markdown]
# ## 3. Visualisation

# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
sns.histplot(df[target], kde=True, ax=ax[0, 0]).set_title("Price distribution")
sns.scatterplot(data=df, x="area_sqft", y=target, hue="location", alpha=.6, ax=ax[0, 1]).set_title("Area vs price")
sns.boxplot(data=df, x="location", y=target, ax=ax[1, 0]).set_title("Price by location")
sns.heatmap(df[num_cols + [target]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax[1, 1]).set_title("Correlation")
plt.tight_layout(); plt.savefig("outputs/01_eda.png", dpi=120); plt.close()

# %% [markdown]
# ## 4. Train / test split (80 / 20)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

# %% [markdown]
# ## 5. Compare regression algorithms

# %%
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.05),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}
pipes, preds, rows = {}, {}, []
for name, m in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", m)]).fit(X_train, y_train)
    p = pipe.predict(X_test); pipes[name], preds[name] = pipe, p
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
    rows.append([name, mean_absolute_error(y_test, p), np.sqrt(mean_squared_error(y_test, p)), r2_score(y_test, p), cv])
results = pd.DataFrame(rows, columns=["Model", "MAE (lakh)", "RMSE (lakh)", "R2 (test)", "R2 (5-fold CV)"]).round(3)
results = results.sort_values("R2 (test)", ascending=False).reset_index(drop=True)
print(results.to_string(index=False)); results.to_csv("outputs/model_comparison.csv", index=False)
best_name = results.loc[0, "Model"]; print("\nBest model:", best_name)

# %% [markdown]
# **Interpretation:** Linear models do well because price is largely linear in area; tree ensembles capture the
# location interaction. MAE is the average error in lakh rupees; R2 shows the share of price variation explained.

# %% [markdown]
# ## 6. Hyper-parameter tuning of the best tree-based model (GridSearchCV)

# %%
tune_pipe = Pipeline([("prep", preprocess), ("model", GradientBoostingRegressor(random_state=42))])
grid = GridSearchCV(tune_pipe, {"model__n_estimators": [100, 200], "model__max_depth": [2, 3, 4],
                                "model__learning_rate": [0.05, 0.1]}, cv=3, scoring="r2", n_jobs=-1).fit(X_train, y_train)
tuned_pred = grid.predict(X_test)
print("Best params:", grid.best_params_)
print("Tuned Gradient Boosting -> MAE %.2f | RMSE %.2f | R2 %.3f" % (
    mean_absolute_error(y_test, tuned_pred), np.sqrt(mean_squared_error(y_test, tuned_pred)), r2_score(y_test, tuned_pred)))

# choose the final model on test R2 between best baseline and tuned model
final_model, final_pred, final_name = pipes[best_name], preds[best_name], best_name
if r2_score(y_test, tuned_pred) > r2_score(y_test, final_pred):
    final_model, final_pred, final_name = grid.best_estimator_, tuned_pred, "Tuned Gradient Boosting"
print("Final model:", final_name)

# %% [markdown]
# ## 7. Result visualisation

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
results.set_index("Model")["R2 (test)"].plot.barh(ax=ax[0], color="teal"); ax[0].set_title("R2 comparison"); ax[0].invert_yaxis()
ax[1].scatter(y_test, final_pred, alpha=.6); lim = [y_test.min(), y_test.max()]; ax[1].plot(lim, lim, "r--")
ax[1].set(title=f"Actual vs predicted ({final_name})", xlabel="Actual (lakh)", ylabel="Predicted (lakh)")
sns.histplot(y_test - final_pred, kde=True, ax=ax[2]); ax[2].set_title("Residuals (actual - predicted)")
plt.tight_layout(); plt.savefig("outputs/02_results.png", dpi=120); plt.close()

# feature importance / coefficients of the final model
est = final_model.named_steps["model"]
names = final_model.named_steps["prep"].get_feature_names_out()
vals = est.feature_importances_ if hasattr(est, "feature_importances_") else np.abs(est.coef_)
pd.Series(vals, index=[n.split("__")[1] for n in names]).sort_values().plot.barh(figsize=(7, 4.5), color="darkorange", title="Feature importance")
plt.tight_layout(); plt.savefig("outputs/03_feature_importance.png", dpi=120); plt.close()

# %% [markdown]
# ## 8. Save the trained model and predict a new house

# %%
joblib.dump(final_model, "model/house_price_model.joblib")
loaded = joblib.load("model/house_price_model.joblib")
new_house = pd.DataFrame([{"area_sqft": 1800, "bedrooms": 3, "bathrooms": 2, "location": "Downtown",
                           "parking_spaces": 2, "age_years": 5}])
print("Predicted price: Rs %.1f lakh" % loaded.predict(new_house)[0])
