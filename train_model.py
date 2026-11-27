import argparse
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default="WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--output",default="churn_model.pkl")
    args=parser.parse_args()
    df=pd.read_csv(args.data)
    df=df.drop(columns=["customerID"],errors="ignore")
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
    y=df.pop("Churn").map({"Yes":1,"No":0})
    X=df
    numeric=X.select_dtypes(include=["number"]).columns.tolist()
    categorical=X.select_dtypes(exclude=["number"]).columns.tolist()
    pre=ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),numeric),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(handle_unknown="ignore"))]),categorical)
    ])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    model=Pipeline([("preprocessor",pre),("classifier",RandomForestClassifier(
        n_estimators=200,max_depth=12,class_weight="balanced",random_state=42,n_jobs=-1
    ))])
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    print("Accuracy:",round(accuracy_score(y_test,pred),4))
    print("Precision:",round(precision_score(y_test,pred),4))
    print("Recall:",round(recall_score(y_test,pred),4))
    print("F1:",round(f1_score(y_test,pred),4))
    joblib.dump(model,args.output)
    print("Saved model to",args.output)

if __name__=="__main__":
    main()
