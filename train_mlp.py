import numpy as np
import pandas as pd
from argparse import ArgumentParser
from utils import * 
import pickle

pd.set_option('future.no_silent_downcasting', True)

class MultilayerPerceptron:
    def __init__(self, X_train, X_test, y_train, y_test, args):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.hidden_layers = args.layer
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.loss_function = args.loss
        self.batch_size = args.batch_size
    
    def create_network(self):
        pass

    def sigmoid_function(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def softmax(self, x: float) -> float:
        pass

    def binary_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: # Loss function
        return -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))

    def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_pred == y_true)

    def feed_forward(self, batch):
        pass

    def back_propagation(self):
        pass

    def fit(self):
        pass

    def save_weights(self, filename="weights.pkl") -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.models, f)
        print_info(f'Weights saved in {UGREEN}{filename}')

    def load_weights(self, filename="weights.pkl") -> None:
        with open(filename, "rb") as f:
            self.models = pickle.load(f)

def main():
    parser = ArgumentParser(
        description='Multilayer Perceptron')
    parser.add_argument('-dataset', type=str, default='datasets/data.csv',
        help='Path to a train dataset file to train the model')
    parser.add_argument('-layer', type=tuple, default='(24, 24, 24)',
        help='Numbers of perceptrons for each layer')
    parser.add_argument('-epochs', type=int, default=700,
        help='Total number of iterations of all the training data '
        'in one cycle for training the model')
    parser.add_argument('-learning-rate', type=int, default=0.1,
        help='Hyperparameter that controls how much to change '
        'the model when the model weights are updated.')
    parser.add_argument('-loss', type=str, default='binaryCrossEntropy',
        help='Loss function to use during training')
    parser.add_argument('-batch-size', type=int, default=8,
        help='Size of each minibatchs for SGD')

    
    args = parser.parse_args()
    df = load_dataset(args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(df)

    model = MultilayerPerceptron(X_train, X_test, y_train, y_test, args)
    model.fit()


if __name__ == '__main__':
    main()
