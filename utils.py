import pandas as pd

def load_assignments(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_outcomes(path: str) -> pd.DataFrame:
    return pd.read_csv(path)