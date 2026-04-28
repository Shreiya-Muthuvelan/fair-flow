import pandas as pd
from sklearn.model_selection import train_test_split
from .config import SPLIT_SIZES,RANDOM_STATE,SPLITS_DIR
from pathlib import Path

def split_and_save(X:pd.Dataframe,y:pd.Series):
    Path(SPLITS_DIR).mkdir(parents=True,exist_ok=True)

    X_train,X_temp,y_train,y_temp=train_test_split(X,y,
                                                   test_size=(SPLIT_SIZES["val"]+ SPLIT_SIZES["test"]),
                                                   stratify=y,
                                                   random_state=RANDOM_STATE)
    
    X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,
                                               test_size=0.5,stratify=y_temp,random_state=RANDOM_STATE)
    
    splits={"X_train":X_train,"y_train":y_train,
            "X_val":X_val,"y_val":y_val,
            "X_test":X_test,"y_test":y_test
            }
    for name in ["X_train", "X_val", "X_test"]:
        df = splits[name]
        path=f"{SPLITS_DIR}{name}.parquet"
        df.to_parquet(path,index=False)
        print(f"Saved {name}: {len(df)} rows -> {path}")
    return splits

