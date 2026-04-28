import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .config import NUMERICAL_FEATURES,CATEGORICAL_FEATURES,PASSTHROUGH_COLS,PREPROCESSOR_PATH

class FairFlowEncoder:
    def __init__(self):
        self.scaler=StandardScaler()
        self.label_encoders={}
        self.fitted=False
    
    def fit(self,X_train:pd.DataFrame):
        # Scale numerical features
        num_cols=[c for c in NUMERICAL_FEATURES if c in X_train.columns]
        self.scaler.fit(X_train[num_cols])

        # Encode categorical features
        cat_cols=[c for c in CATEGORICAL_FEATURES if c in X_train.columns]
        for col in cat_cols:
            le=LabelEncoder()
            le.fit(X_train[col].astype(str))
            self.label_encoders[col] = le
        self.fitted=True

        # Encode Sex: male=1, female=0
        sex_le = LabelEncoder()
        X_train["Sex_encoded"] = sex_le.fit_transform(X_train["Sex"])

        # Encode Age_group: convert category to int
        X_train["Age_group_encoded"] = X_train["Age_group"].cat.codes

        return self
    def transform(self,X:pd.DataFrame)->pd.DataFrame:
        assert self.fitted, "Call fit() before transform"
        X=X.copy()
        num_cols=[c for c in NUMERICAL_FEATURES if c in X.columns]
        X[num_cols]=self.scaler.transform(X[num_cols])

        for col, le in self.label_encoders.items():
            if col in X.columns:
                X[col]=le.transform(X[col].astype(str))
            
        return X
    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)
    def save(self,path=PREPROCESSOR_PATH):
        with open(path,"wb") as f:
            pickle.dump(self,f)
        print(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls,path=PREPROCESSOR_PATH):
        with open(path,"rb") as f:
            return pickle.load(f)
    

