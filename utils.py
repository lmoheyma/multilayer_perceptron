import pandas as pd
import numpy as np
from colors import *

def load_dataset(dataset_label: str) -> pd.core.frame.DataFrame:
    try:
        df = pd.read_csv(dataset_label)
        print(type(df))
    except Exception as e:
        print(f"{BHRED}Fail to read file '{RED}{dataset_label}{BHRED}'.{RESET}")
        raise e
    return df

def train_test_split(df: pd.core.frame.DataFrame, test_size=0.25) -> tuple:
    df = df.sample(frac=1)
    df.dropna(inplace=True)
    train_size = int(df.shape[0] * (1 - test_size))

    X_train = df[0:train_size]
    X_test = df[train_size:0]
    y_train = X_train['Diagnosis']
    y_test = X_test['Diagnosis']
    X_train.drop(['Diagnosis', 'Index'], axis = 1, inplace = True)
    X_test.drop(['Diagnosis', 'Index'], axis = 1, inplace = True)

    return X_train, X_test, y_train, y_test

def min_max_scaling(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return (X - min_val) / (max_val - min_val)

def print_info(message: str) -> None:
    print(f'{CYANB}{BWHITE}[INFO]{RESET}{BWHITE} {message}{RESET}')

def list_of_ints(arg):
    return list(map(int, arg.split()))

if __name__ == '__main__':
    df = load_dataset('datasets/data.csv')
    print(df.columns)
