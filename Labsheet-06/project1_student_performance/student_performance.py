# %% [markdown]
# # Project 1: Student Performance Prediction System
# **MCA201P - Lab Sheet 06 | MCA III Semester (2026-2027)**
#
# Two supervised tasks on the same dataset:
# 1. **Regression** - predict the final score (0-100).
# 2. **Classification** - predict Pass / Fail (final score >= 40).
#
# Inputs: attendance, internal marks, assignments, study hours, previous semester result.

# %%
import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix, classification_report)
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
os.makedirs("outputs", exist_ok=True); os.makedirs("model", exist_ok=True)
if not os.path.exists("student_performance.csv"):
    import generate_dataset

# %% [markdown]
# ## 1. Load the dataset and explore

# %%
df = pd.read_csv("student_performance.csv")
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isna().sum())
print("Duplicates:", df.duplicated().sum())
print(df.describe().round(2))

# %% [markdown]
# ## 2. Data preprocessing
# * remove duplicate rows and the ID column (no predictive meaning)
# * missing values are filled with the **median** inside the model pipeline (so the test set never leaks into training)
# * features are standardised for distance/gradient based models

# %%
df = df.drop_duplicates().drop(columns="student_id").reset_index(drop=True)
features = ["attendance_pct", "internal_marks", "assignment_marks", "study_hours_per_day", "previous_sem_percentage"]
X, y = df[features], df["final_score"]
print("After cleaning:", df.shape)

# %% [markdown]
# ## 3. Exploratory visualisation

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(df["final_score"], kde=True, ax=ax[0]); ax[0].set_title("Distribution of final score")
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax[1]); ax[1].set_title("Correlation matrix")
plt.tight_layout(); plt.savefig("outputs/01_eda.png", dpi=120); plt.close()
sns.pairplot(df, y_vars="final_score", x_vars=features, height=2.6, kind="reg", plot_kws={"scatter_kws": {"alpha": .3}})
plt.savefig("outputs/02_feature_vs_score.png", dpi=110); plt.close()

# %% [markdown]
# ## 4. Train / test split (80 / 20)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

# %% [markdown]
# ## 5. Regression - compare algorithms

# %%
def make_pipe(model):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", model)])

reg_models = {
    "Linear Regression": make_pipe(LinearRegression()),
    "Decision Tree": make_pipe(DecisionTreeRegressor(max_depth=5, random_state=42)),
    "Random Forest": make_pipe(RandomForestRegressor(n_estimators=200, random_state=42)),
    "Gradient Boosting": make_pipe(GradientBoostingRegressor(random_state=42)),
}
rows, preds = [], {}
for name, pipe in reg_models.items():
    pipe.fit(X_train, y_train); p = pipe.predict(X_test); preds[name] = p
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
    rows.append([name, mean_absolute_error(y_test, p), np.sqrt(mean_squared_error(y_test, p)), r2_score(y_test, p), cv])
reg_results = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2 (test)", "R2 (5-fold CV)"]).round(3)
print(reg_results.to_string(index=False)); reg_results.to_csv("outputs/regression_results.csv", index=False)
best_reg_name = reg_results.sort_values("R2 (test)", ascending=False).iloc[0]["Model"]
print("\nBest regression model:", best_reg_name)

# %% [markdown]
# **Interpretation:** MAE is the average mark error; RMSE penalises large errors; R2 is the share of variance in
# final score explained by the model (closer to 1 is better). CV R2 close to test R2 means the model generalises.

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
reg_results.set_index("Model")[["R2 (test)"]].plot.bar(ax=ax[0], legend=False, rot=20, color="teal"); ax[0].set_title("R2 by model")
bp = preds[best_reg_name]
ax[1].scatter(y_test, bp, alpha=.6); ax[1].plot([0, 100], [0, 100], "r--")
ax[1].set(title=f"Actual vs predicted ({best_reg_name})", xlabel="Actual", ylabel="Predicted")
plt.tight_layout(); plt.savefig("outputs/03_regression_results.png", dpi=120); plt.close()

best_reg = reg_models[best_reg_name]
inner = best_reg.named_steps["model"]
imp = inner.feature_importances_ if hasattr(inner, "feature_importances_") else np.abs(inner.coef_)
pd.Series(imp, index=features).sort_values().plot.barh(color="darkorange", figsize=(6, 3.5), title="Feature importance")
plt.tight_layout(); plt.savefig("outputs/04_feature_importance.png", dpi=120); plt.close()

# %% [markdown]
# ## 6. Classification - Pass / Fail
# A student passes if the final score is at least 40.

# %%
df["result"] = (df["final_score"] >= 40).astype(int)
print(df["result"].value_counts().rename({1: "Pass", 0: "Fail"}))
yc = df["result"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, yc, test_size=0.2, random_state=42, stratify=yc)
clf_models = {
    "Logistic Regression": make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced")),
    "KNN": make_pipe(KNeighborsClassifier(n_neighbors=7)),
    "Random Forest": make_pipe(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
}
rows = []
for name, pipe in clf_models.items():
    pipe.fit(Xc_train, yc_train); p = pipe.predict(Xc_test)
    rows.append([name, accuracy_score(yc_test, p), precision_score(yc_test, p), recall_score(yc_test, p), f1_score(yc_test, p)])
clf_results = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1"]).round(3)
print(clf_results.to_string(index=False)); clf_results.to_csv("outputs/classification_results.csv", index=False)
best_clf_name = clf_results.sort_values("F1", ascending=False).iloc[0]["Model"]
best_clf = clf_models[best_clf_name]
print("\nBest classifier:", best_clf_name)
print(classification_report(yc_test, best_clf.predict(Xc_test), target_names=["Fail", "Pass"]))

plt.figure(figsize=(4.5, 4))
sns.heatmap(confusion_matrix(yc_test, best_clf.predict(Xc_test)), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.title(f"Confusion matrix - {best_clf_name}"); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig("outputs/05_confusion_matrix.png", dpi=120); plt.close()

# %% [markdown]
# **Interpretation:** Precision = of students predicted to pass, how many really passed. Recall = of students who
# really passed, how many we found. For an early-warning system, recall of the *Fail* class is also important.

# %% [markdown]
# ## 7. Save the trained models and make a prediction

# %%
joblib.dump(best_reg, "model/student_score_regressor.joblib")
joblib.dump(best_clf, "model/student_result_classifier.joblib")
loaded_reg = joblib.load("model/student_score_regressor.joblib")
loaded_clf = joblib.load("model/student_result_classifier.joblib")
new_student = pd.DataFrame([{"attendance_pct": 85, "internal_marks": 38, "assignment_marks": 30,
                             "study_hours_per_day": 4, "previous_sem_percentage": 72}])
print("Predicted final score:", round(float(loaded_reg.predict(new_student)[0]), 1))
print("Predicted result     :", "Pass" if loaded_clf.predict(new_student)[0] == 1 else "Fail")
