import pandas as pd
from colors import *

def load_dataset(dataset_label: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(dataset_label)
        print(type(df))
    except Exception as e:
        print(f"{BHRED}Fail to read file '{RED}{dataset_label}{BHRED}'.{RESET}")
        raise e
    return df

def train_test_split(df: pd.DataFrame, test_size=0.25) -> tuple:
    df = df.sample(frac=1)
    train_size = int(df.shape[0] * (1 - test_size))

    train = df[0:train_size]
    test = df[train_size:0]
    return train, test
