import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colors import *

def load_dataset(dataset_label: str) -> pd.core.frame.DataFrame:
    try:
        df = pd.read_csv(dataset_label)
    except Exception as e:
        print(f"{BHRED}Fail to read file '{RED}{dataset_label}{BHRED}'.{RESET}")
        raise e
    return df

def train_test_split(df: pd.core.frame.DataFrame, test_size=0.25) -> tuple:
    df = df.sample(frac=1)
    train_size = int(df.shape[0] * (1 - test_size))
    X_train = df[0:train_size]
    X_test = df[train_size:]
    y_train = X_train['Diagnosis']
    y_test = X_test['Diagnosis']
    X_train = X_train.drop(['Diagnosis', 'Index'], axis=1)
    X_test = X_test.drop(['Diagnosis', 'Index'], axis=1)

    print(f'x_train shape : {X_train.shape}\nx_valid shape : {X_test.shape}')
    return X_train, X_test, y_train, y_test

def min_max_scaling(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return (X - min_val) / (max_val - min_val)

def data_preprocessing(X):
    features = X.select_dtypes('float64')
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    features_to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
    for feature in features_to_drop:
        X.drop([feature], axis=1, inplace=True)
    features = X.select_dtypes('float64')
    features_col = features.columns
    X[features_col] = X[features_col].fillna(X[features_col].mean())
    X[features_col] = min_max_scaling(X[features_col])
    return X

def print_info(message: str) -> None:
    print(f'{CYANB}{BWHITE}[INFO]{RESET}{BWHITE} {message}{RESET}')

def plot_print_info(func):
    def wrapper(model, ax):
        func(model, ax)
        print_info(f'{func.__doc__.strip()}...')
    return wrapper

@plot_print_info
def display_loss_plot(model, ax):
    """
    Display Loss Function plot
    """
    ax[0].plot(model.losses, label="training loss")
    ax[0].plot(model.losses_test, label="validation loss")
    ax[0].legend()
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid()

@plot_print_info
def display_accuracy_score_plot(model, ax):
    """
    Display Accuracy Score plot
    """
    ax[1].plot(model.accuracy, label="training acc")
    ax[1].plot(model.accuracy_test, label="validation acc")
    ax[1].legend()
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid()

def list_of_ints(arg):
    return list(map(int, arg.split()))

if __name__ == '__main__':
    df = load_dataset('datasets/data.csv')
    print(df.columns)
