from pathlib import Path
import joblib, pandas as pd, streamlit as st
ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"models"/"disease_prediction_model.joblib"
SYMPTOMS=["fever","cough","fatigue","headache","sore_throat","body_pain","nausea","rash","breathlessness","runny_nose"]
st.set_page_config(page_title="Disease Prediction Demo",page_icon="🩺")
st.title("🩺 Disease Prediction System")
st.write("Select symptoms and enter basic attributes for an educational model output.")
st.error("Educational demonstration only — not a medical diagnosis or treatment recommendation. Consult a qualified healthcare professional.")
if not MODEL.exists():
 st.error("Model not found. Run `python train_model.py` first."); st.stop()
model=joblib.load(MODEL)
with st.form("disease_form"):
 c1,c2=st.columns(2)
 with c1:
  age=st.number_input("Age",min_value=1,max_value=120,value=25)
  temp=st.number_input("Body temperature (°C)",min_value=34.0,max_value=43.0,value=37.0,step=.1)
 with c2:
  bp=st.number_input("Systolic blood pressure (mmHg)",min_value=50,max_value=250,value=120)
  pulse=st.number_input("Pulse rate (beats/min)",min_value=30,max_value=220,value=80)
 st.subheader("Symptoms")
 values={}
 symptom_cols=st.columns(2)
 for i,s in enumerate(SYMPTOMS):
  with symptom_cols[i%2]: values[s]=int(st.checkbox(s.replace("_"," ").title()))
 submitted=st.form_submit_button("Predict (educational demo)")
if submitted:
 sample=pd.DataFrame([{"age":age,"temperature_c":temp,"systolic_bp":bp,"pulse_rate":pulse,**values}])
 prediction=model.predict(sample)[0]
 st.subheader(f"Model output: {prediction}")
 if hasattr(model,"predict_proba"):
  probs=model.predict_proba(sample)[0]
  classes=model.named_steps["classifier"].classes_
  ranked=sorted(zip(classes,probs),key=lambda x:x[1],reverse=True)[:3]
  st.write("Top model scores (not medical probabilities):")
  for label,score in ranked: st.write(f"- {label}: {score*100:.1f}%")
 st.warning("This output is not a diagnosis. Symptoms can have many causes; seek medical advice when needed.")
