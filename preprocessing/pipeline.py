from .loader import load_raw_data, split_X_y
from .cleaner import clean
from .encoder import FairFlowEncoder
from .splitter import split_and_save
import pandas as pd


def run_preprocessing_pipeline():
    print("Step 1 : Load")
    df=load_raw_data()
    X,y=split_X_y(df)

    print("Step 2 : Clean")
    X=clean(X)

    print("Step 3: Split")
    splits=split_and_save(X,y)

    print("Step 4 : Encode")
    encoder=FairFlowEncoder()
    encoder.fit(splits["X_train"])
    splits["X_train"]=encoder.transform(splits["X_train"])
    splits["X_val"]=encoder.transform(splits["X_val"])
    splits["X_test"]=encoder.transform(splits["X_test"])
    encoder.save()

    print("Step 5: Save the encoded splits")
    from pathlib import Path
    from preprocessing.config import SPLITS_DIR

    for name in ["X_train","X_val","X_test","y_train","y_val","y_test"]:
        if isinstance(splits[name], pd.Series):
            splits[name].to_frame().to_parquet(f"{SPLITS_DIR}{name}.parquet", index=False)
        else:
            splits[name].to_parquet(f"{SPLITS_DIR}{name}.parquet", index=False)
    
    return splits,encoder

