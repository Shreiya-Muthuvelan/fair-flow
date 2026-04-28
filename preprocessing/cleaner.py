import pandas as pd

from .config import NUMERICAL_FEATURES, SENSITIVE_ATTRS

def clean(X: pd.DataFrame) -> pd.DataFrame:
    X=X.copy()
    X=X.drop(columns=['Unnamed: 0'])

    print("Combining 'quite rich' and 'rich' columns in Savings Account attribute")
    X['Saving accounts']=X['Saving accounts'].replace(['quite rich','rich'],'rich')
    print("Imputing NaN values")
    X['Saving accounts']=X['Saving accounts'].fillna('no_account')
    X['Checking account']=X['Checking account'].fillna('no_account')

    X['Saving accounts'].value_counts()
    X['Checking account'].value_counts()

    print("Feature Engineering")
    print("Creating a Feature Age_group")
    X['Age_group']=pd.cut(X['Age'],bins=[0,25,35,45,100],labels=['18-25','26-35','36-45','46+'])
    print("Creating Debt_ratio( Debt_ratio=Credit Amount/Duration )")
    X['Debt_ratio']=X['Credit amount'] / (X['Duration']+1)
    return X

    

