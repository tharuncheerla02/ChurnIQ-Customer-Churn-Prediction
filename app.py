import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="ChurnIQ | Retention Intelligence", page_icon="📡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#f5f7fb;}
.block-container {padding-top:2rem; max-width:1400px;}
.hero {background:linear-gradient(115deg,#101c3d,#253e75);padding:28px 32px;border-radius:20px;color:white;margin-bottom:22px;}
.hero h1 {font-family:Manrope,sans-serif;font-size:2rem;margin:0 0 6px 0;color:white;}
.hero p {color:#cbd5ef;margin:0;font-size:1rem;}
.kpi {background:white;border:1px solid #e7ebf3;border-radius:16px;padding:18px 20px;}
.kpi-label {color:#667085;font-size:.82rem;font-weight:600;}
.kpi-value {font-family:Manrope,sans-serif;font-size:1.8rem;font-weight:800;color:#182544;margin-top:4px;}
.kpi-note {color:#667085;font-size:.76rem;margin-top:3px;}
.section {font-family:Manrope,sans-serif;font-size:1.15rem;font-weight:800;color:#182544;margin:12px 0;}
div[data-testid="stMetric"] {background:white;border:1px solid #e7ebf3;padding:14px 16px;border-radius:14px;}
.stButton>button {border-radius:10px;font-weight:700;min-height:2.8rem;}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "churn_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

model = load_model()

st.markdown("""
<div class="hero">
  <h1>📡 ChurnIQ</h1>
  <p>Customer retention intelligence · Telco churn prediction workspace</p>
</div>
""", unsafe_allow_html=True)

pages = st.sidebar.radio("WORKSPACE", ["Overview", "Predict churn", "About the model"])
st.sidebar.markdown("---")
st.sidebar.caption("TELCO CUSTOMER ANALYTICS")
st.sidebar.caption("Model: Random Forest · 200 trees")
if model is None:
    st.sidebar.warning("Model file not found. Train it using train_model.py.")

if pages == "Overview":
    st.markdown('<div class="section">Model snapshot</div>', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    for col, label, value, note in [
        (a,"TEST ACCURACY","78.50%","Held-out test set"),
        (b,"CHURN RECALL","66.31%","Churn class detection"),
        (c,"CHURN F1 SCORE","62.08%","Precision / recall balance"),
        (d,"TEST CUSTOMERS","1,409","20% stratified split"),
    ]:
        col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section">Experiment results</div>', unsafe_allow_html=True)
    metrics = pd.DataFrame({
        "Model":["Logistic Regression","Decision Tree","Random Forest"],
        "Accuracy":[.738112,.735273,.784954],
        "Precision":[.504303,.500867,.583529],
        "Recall":[.783422,.772727,.663102],
        "F1 Score":[.613613,.607781,.620776]
    })
    left,right=st.columns([1.4,1])
    with left:
        fig=px.bar(metrics.melt(id_vars="Model",var_name="Metric",value_name="Score"),
                   x="Model",y="Score",color="Metric",barmode="group",
                   color_discrete_sequence=["#4968ad","#8ba5d9","#e5a35b","#55a995"])
        fig.update_layout(height=360,plot_bgcolor="white",paper_bgcolor="white",
                          yaxis_range=[0,1],legend_title="",margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown("**Churn distribution in source dataset**")
        dist=pd.DataFrame({"Status":["Stayed","Churned"],"Customers":[5174,1869]})
        fig2=px.pie(dist,names="Status",values="Customers",hole=.66,
                    color="Status",color_discrete_map={"Stayed":"#4968ad","Churned":"#e5a35b"})
        fig2.update_layout(height=300,showlegend=True,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2,use_container_width=True)
        st.caption("7,043 customers · 26.54% churn rate")
    st.info("The evaluation figures above are taken from the supplied notebook's saved outputs. They are not recalculated in this dashboard.")

elif pages == "Predict churn":
    st.markdown('<div class="section">Customer profile</div>', unsafe_allow_html=True)
    if model is None:
        st.error("The trained model file `churn_model.pkl` is missing.")
        st.code("python train_model.py --data WA_Fn-UseC_-Telco-Customer-Churn.csv", language="bash")
        st.stop()

    with st.form("customer_form"):
        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown("**Demographics**")
            gender=st.selectbox("Gender",["Female","Male"])
            senior=st.selectbox("Senior citizen",["No","Yes"])
            partner=st.selectbox("Partner",["Yes","No"])
            dependents=st.selectbox("Dependents",["Yes","No"])
            tenure=st.slider("Tenure (months)",0,72,12)
        with c2:
            st.markdown("**Services**")
            phone=st.selectbox("Phone service",["Yes","No"])
            lines=st.selectbox("Multiple lines",["No","Yes","No phone service"])
            internet=st.selectbox("Internet service",["DSL","Fiber optic","No"])
            security=st.selectbox("Online security",["No","Yes","No internet service"])
            backup=st.selectbox("Online backup",["No","Yes","No internet service"])
            protection=st.selectbox("Device protection",["No","Yes","No internet service"])
            support=st.selectbox("Tech support",["No","Yes","No internet service"])
            tv=st.selectbox("Streaming TV",["No","Yes","No internet service"])
            movies=st.selectbox("Streaming movies",["No","Yes","No internet service"])
        with c3:
            st.markdown("**Account & billing**")
            contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])
            paperless=st.selectbox("Paperless billing",["Yes","No"])
            payment=st.selectbox("Payment method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            monthly=st.number_input("Monthly charges ($)",18.25,200.0,70.0,step=1.0)
            total=st.number_input("Total charges ($)",0.0,10000.0,840.0,step=10.0)
        submitted=st.form_submit_button("Assess churn risk",type="primary",use_container_width=True)

    if submitted:
        row=pd.DataFrame([{
            "gender":gender,"SeniorCitizen":1 if senior=="Yes" else 0,"Partner":partner,
            "Dependents":dependents,"tenure":tenure,"PhoneService":phone,"MultipleLines":lines,
            "InternetService":internet,"OnlineSecurity":security,"OnlineBackup":backup,
            "DeviceProtection":protection,"TechSupport":support,"StreamingTV":tv,
            "StreamingMovies":movies,"Contract":contract,"PaperlessBilling":paperless,
            "PaymentMethod":payment,"MonthlyCharges":monthly,"TotalCharges":total
        }])
        probability=float(model.predict_proba(row)[0][1])
        prediction=int(model.predict(row)[0])
        st.markdown("---")
        x,y=st.columns([1,2])
        with x:
            st.metric("Predicted churn probability",f"{probability:.1%}")
            st.progress(probability)
        with y:
            if prediction:
                st.error("⚠️ **Higher churn risk** — this profile is classified as likely to churn.")
                st.write("Consider a proactive retention conversation, reviewing contract fit, and checking service issues.")
            else:
                st.success("✓ **Lower churn risk** — this profile is classified as likely to stay.")
                st.write("Continue monitoring engagement and service experience.")
        st.caption("This is a model estimate, not a certainty. Use it as decision support, not as the sole basis for customer treatment.")

else:
    st.markdown('<div class="section">Model methodology</div>', unsafe_allow_html=True)
    st.write("This project follows the workflow in the supplied Customer_churn_prediction notebook.")
    st.markdown("""
    1. **Dataset:** Telco customer churn data — 7,043 rows and 21 columns.
    2. **Cleaning:** Remove customerID, convert TotalCharges to numeric, and encode Churn as 0/1.
    3. **Split:** 80/20 train-test split with stratification and random_state=42.
    4. **Preprocessing:** Median imputation and standard scaling for numeric fields; most-frequent imputation and one-hot encoding for categorical fields.
    5. **Models:** Logistic Regression, Decision Tree, and Random Forest with class balancing.
    6. **Selection:** Random Forest selected by highest F1 score (0.6208).
    """)
    st.markdown("**Test-set classification report**")
    report=pd.DataFrame({
        "Class":["No Churn","Churn"],
        "Precision":[.87,.58],"Recall":[.83,.66],"F1 score":[.85,.62],"Support":[1035,374]
    })
    st.dataframe(report,hide_index=True,use_container_width=True)
    st.warning("The notebook's confusion-matrix plot title says Random Forest, and its final saved model is explicitly the Random Forest pipeline.")
