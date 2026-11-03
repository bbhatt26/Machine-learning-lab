from pathlib import Path
import json, joblib, pandas as pd, matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

ROOT=Path(__file__).resolve().parent
MODEL_DIR=ROOT/"models"; MODEL_DIR.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/"loan_applicants.csv")
X=df.drop(columns=["Loan_Status"]); y=df["Loan_Status"].astype(int)
num=X.select_dtypes(include="number").columns.tolist()
cat=X.select_dtypes(exclude="number").columns.tolist()
pre=ColumnTransformer([
 ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
 ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                  ("encode",OneHotEncoder(handle_unknown="ignore"))]),cat)
])
models={
 "Logistic Regression":LogisticRegression(max_iter=1500,class_weight="balanced"),
 "Decision Tree":DecisionTreeClassifier(max_depth=6,random_state=42,class_weight="balanced"),
 "Random Forest":RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced"),
 "Gradient Boosting":GradientBoostingClassifier(random_state=42)
}
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
rows=[]; best=None; best_score=-1; best_name=""
for name,clf in models.items():
 pipe=Pipeline([("preprocessing",pre),("classifier",clf)])
 pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
 row={"model":name,"accuracy":accuracy_score(yte,pred),
      "precision":precision_score(yte,pred,zero_division=0),
      "recall":recall_score(yte,pred,zero_division=0),
      "f1_score":f1_score(yte,pred,zero_division=0)}
 rows.append(row)
 if row["f1_score"]>best_score: best,best_score,best_name=pipe,row["f1_score"],name
pd.DataFrame(rows).sort_values("f1_score",ascending=False).to_csv(MODEL_DIR/"model_comparison.csv",index=False)
joblib.dump(best,MODEL_DIR/"loan_approval_model.joblib")
pred=best.predict(Xte)
with open(MODEL_DIR/"classification_report.json","w") as f:
 json.dump(classification_report(yte,pred,output_dict=True,zero_division=0),f,indent=2)
ConfusionMatrixDisplay(confusion_matrix(yte,pred),display_labels=["Not Approved","Approved"]).plot(cmap="Blues")
plt.title(f"Loan Approval – {best_name}"); plt.tight_layout()
plt.savefig(MODEL_DIR/"confusion_matrix.png",dpi=150); plt.close()
print(pd.DataFrame(rows).sort_values("f1_score",ascending=False).to_string(index=False))
print("Best model:",best_name,"| Saved to:",MODEL_DIR)
