from pathlib import Path
import joblib, pandas as pd, streamlit as st
ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"models"/"loan_approval_model.joblib"
st.set_page_config(page_title="Loan Approval Predictor",page_icon="🏦")
st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details for an educational model prediction.")
st.warning("Classroom demonstration only — not a real lending or credit decision.")
if not MODEL.exists():
 st.error("Model not found. Run `python train_model.py` first."); st.stop()
model=joblib.load(MODEL)
with st.form("loan_form"):
 c1,c2=st.columns(2)
 with c1:
  income=st.number_input("Applicant income (per month)",min_value=0,value=45000,step=1000)
  amount=st.number_input("Loan amount",min_value=0,value=150000,step=5000)
  term=st.selectbox("Loan term (months)",[12,24,36,48,60],index=2)
  credit=st.selectbox("Credit history",[1,0],format_func=lambda x:"Good (1)" if x==1 else "Poor/Unavailable (0)")
  education=st.selectbox("Education",["Graduate","Not Graduate"])
 with c2:
  coincome=st.number_input("Co-applicant income (per month)",min_value=0,value=0,step=1000)
  employed=st.selectbox("Self employed?",["No","Yes"])
  married=st.selectbox("Marital status",["Yes","No"])
  dependents=st.selectbox("Dependents",["0","1","2","3+"])
 submitted=st.form_submit_button("Predict loan status")
if submitted:
 sample=pd.DataFrame([{"ApplicantIncome":income,"CoapplicantIncome":coincome,"LoanAmount":amount,
  "Loan_Amount_Term":term,"Credit_History":credit,"Education":education,
  "Self_Employed":employed,"Married":married,"Dependents":dependents}])
 result=int(model.predict(sample)[0])
 st.success("Model prediction: Likely to be approved" if result==1 else "Model prediction: Likely not to be approved")
 if hasattr(model,"predict_proba"):
  st.write(f"Model-estimated approval score: **{model.predict_proba(sample)[0][1]*100:.1f}%**")
 st.caption("Synthetic educational data; never use this output for actual lending.")
