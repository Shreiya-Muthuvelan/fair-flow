# preprocessing/loader.py
import pandas as pd
from .config import RAW_DATA_PATH, TARGET_COL, CATEGORICAL_FEATURES

def load_raw_data(path=RAW_DATA_PATH):
    df=pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df

def split_X_y(df):
    y=df[TARGET_COL].map({"good":1,"bad":0})
    X=df.drop(columns=[TARGET_COL])
    return X,y

def explore_data(df):
    print(f"Shape of the dataset:{df.shape}\n")
    print("First 5 rows of the dataset : \n")
    print(df.head())
    print("Types of attributes : \n")
    print(df.dtypes)
    print("Exploring Categorical Attributes : ")
    for i in CATEGORICAL_FEATURES:
        print(i)
        print(df[i].value_counts())
    print("Exploring Attributes with Null Values")
    print(df.isna().sum())
    
