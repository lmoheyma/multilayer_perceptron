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
        input_size = self.X_train.shape[1]
        hidden_size1 = 64
        hidden_size2 = 32
        output_size = 1
        self.layer_sizes = np.array([input_size, hidden_size1, hidden_size2, output_size])
        self.epochs = args.epochs
        self.learning_rate = 0.1
        self.loss_function = args.loss
        self.batch_size = args.batch_size
    
    def init_layers(self):
        self.hidden_layers = [np.zeros((self.batch_size, layer_size)) for layer_size in self.layer_sizes]

    def init_weights(self):
        self.weights = []
        for i in range(self.layer_sizes.shape[0] - 1):
            weight_shape = (self.layer_sizes[i], self.layer_sizes[i+1])
            self.weights.append(np.random.uniform(-0.1, 0.1, size=weight_shape))

    def sigmoid_function(self, x) -> float:
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, sigmoid):
        return sigmoid * (1 - sigmoid)

    def softmax(self, x) -> float:
        return self.sigmoid_function(x)

    def binary_cross_entropy(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_pred == y_true)

    def feed_forward(self, batch):
        h_l = batch
        self.hidden_layers[0] = h_l
        for i, weight in enumerate(self.weights):
            h_l = self.sigmoid_function(np.dot(h_l, weight))
            self.hidden_layers[i+1] = h_l  
        return h_l

    def back_propagation(self, output, batch_y):
        delta_t = (output - batch_y) * self.sigmoid_prime(output)
        grad = np.mean(np.dot(self.hidden_layers[-2].T, delta_t), axis=1, keepdims=True)
        self.weights[-1] -= self.learning_rate * grad
        for i in range(2, len(self.weights) + 1):
            delta_t = self.sigmoid_prime(self.hidden_layers[-i]) * np.dot(delta_t, self.weights[-i+1].T)
            grad = np.mean(np.dot(self.hidden_layers[-i-1].T, delta_t), axis=1, keepdims=True)
            self.weights[-i] -= self.learning_rate * grad

    def fit(self):
        n_samples = self.X_train.shape[0]
        self.y_train = np.where(self.y_train == 'B', 1, 0) # B or M
    
        self.init_weights()
        
        for epoch in range(self.epochs):
            self.init_layers()
            shuffle = np.random.permutation(n_samples)
            X_batches = np.array_split(self.X_train, n_samples / self.batch_size)
            Y_batches = np.array_split(self.y_train, n_samples / self.batch_size)
            
            train_loss = 0
            for batch_x, batch_y in zip(X_batches, Y_batches):
                batch_x = batch_x.to_numpy() if hasattr(batch_x, 'to_numpy') else batch_x
                batch_y = batch_y.to_numpy() if hasattr(batch_y, 'to_numpy') else batch_y
                batch_y = batch_y.reshape(-1, 1)
                
                pred = self.feed_forward(batch_x)
                train_loss += self.binary_cross_entropy(pred, batch_y)
                self.back_propagation(pred, batch_y)

            train_loss = (train_loss / len(X_batches))
            print(f"Epoch {epoch+1}: loss = {train_loss.round(3)}")
        
        return self.weights

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
    parser.add_argument('-layer', type=list_of_ints, default='24 24 24',
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
