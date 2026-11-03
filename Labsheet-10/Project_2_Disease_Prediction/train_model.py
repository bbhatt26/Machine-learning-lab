from pathlib import Path
import json, joblib, pandas as pd, matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
 classification_report,confusion_matrix,ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

ROOT=Path(__file__).resolve().parent
MODEL_DIR=ROOT/"models"; MODEL_DIR.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/"patient_symptoms.csv")
X=df.drop(columns=["disease"]); y=df["disease"].astype(str)
numeric=["age","temperature_c","systolic_bp","pulse_rate"]
symptoms=[c for c in X.columns if c not in numeric]
pre=ColumnTransformer([
 ("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
 ("symptoms",SimpleImputer(strategy="most_frequent"),symptoms)
])
models={
 "Logistic Regression":LogisticRegression(max_iter=2000,class_weight="balanced"),
 "Decision Tree":DecisionTreeClassifier(max_depth=8,random_state=42,class_weight="balanced"),
 "Random Forest":RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced"),
 "Gradient Boosting":GradientBoostingClassifier(random_state=42)
}
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
rows=[]; best=None; best_score=-1; best_name=""
for name,clf in models.items():
 pipe=Pipeline([("preprocessing",pre),("classifier",clf)])
 pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
 row={"model":name,"accuracy":accuracy_score(yte,pred),
      "precision_weighted":precision_score(yte,pred,average="weighted",zero_division=0),
      "recall_weighted":recall_score(yte,pred,average="weighted",zero_division=0),
      "f1_weighted":f1_score(yte,pred,average="weighted",zero_division=0)}
 rows.append(row)
 if row["f1_weighted"]>best_score: best,best_score,best_name=pipe,row["f1_weighted"],name
pd.DataFrame(rows).sort_values("f1_weighted",ascending=False).to_csv(MODEL_DIR/"model_comparison.csv",index=False)
joblib.dump(best,MODEL_DIR/"disease_prediction_model.joblib")
pred=best.predict(Xte)
with open(MODEL_DIR/"classification_report.json","w") as f:
 json.dump(classification_report(yte,pred,output_dict=True,zero_division=0),f,indent=2)
labels=sorted(y.unique())
fig,ax=plt.subplots(figsize=(9,7))
ConfusionMatrixDisplay(confusion_matrix(yte,pred,labels=labels),display_labels=labels).plot(ax=ax,cmap="Blues",xticks_rotation=45)
plt.title(f"Disease Classification – {best_name}"); plt.tight_layout()
plt.savefig(MODEL_DIR/"confusion_matrix.png",dpi=150); plt.close()
print(pd.DataFrame(rows).sort_values("f1_weighted",ascending=False).to_string(index=False))
print("Best model:",best_name,"| Saved to:",MODEL_DIR)
