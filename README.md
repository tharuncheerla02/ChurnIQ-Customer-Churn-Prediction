# ChurnIQ — Telco Customer Churn Prediction

A Streamlit dashboard and prediction interface built from `Customer_churn_prediction.ipynb`.

## What's included
- Overview dashboard with notebook test metrics and model comparison.
- Customer profile form that predicts churn probability.
- Methodology and test classification report.
- Reproducible training script for the Random Forest pipeline.

## Dataset
The notebook references `WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 rows, 21 columns). Obtain the same Telco Customer Churn CSV used by your notebook and place it in this folder. The CSV was not included in the uploaded files, so the model artifact cannot be generated until you provide it.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python train_model.py --data WA_Fn-UseC_-Telco-Customer-Churn.csv
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `train_model.py`, `requirements.txt`, and the CSV (only if your dataset license permits redistribution).
2. Generate `churn_model.pkl` locally using the training command above, then upload it to the repository. Do not commit customer-identifiable data.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **New app**, select the repository, branch, and `app.py`, then deploy.
5. Keep the model artifact and dependencies in the repository, or configure a secure download step for a private artifact.

## Notebook evaluation reference
Random Forest: accuracy 0.784954, precision 0.583529, recall 0.663102, F1 0.620776.
The notebook split uses 80/20 stratified sampling with random_state=42.
