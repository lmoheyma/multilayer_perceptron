import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from colors import UGREEN, RESET
from utils import print_info, data_preprocessing, \
    display_accuracy_score_plot, display_loss_plot, \
    load_dataset, train_test_split, list_of_ints
import pickle
from weights_init import WeightInitializer
from activation_function import ActivationFunction
pd.set_option('future.no_silent_downcasting', True)


class MultilayerPerceptron:
    def __init__(self, X_train, X_test, y_train, y_test, args,
                 weight_initializer=WeightInitializer,
                 activation=ActivationFunction,
                 output_activation=ActivationFunction):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.args = args
        self.losses = []
        self.losses_test = []
        self.accuracy = []
        self.accuracy_test = []
        self.weight_init = weight_initializer
        self.activation_function = activation
        self.output_activation = output_activation

    def init_hyperparameters(self):
        self.epochs = self.args.epochs
        self.learning_rate = self.args.learning_rate
        self.batch_size = self.args.batch_size

    def init_layers(self, size):
        self.hidden_layers = [np.zeros((size, ls)) for ls in self.layer_sizes]

    def init_weights(self):
        self.layer_sizes = np.array([self.X_train.shape[1]] +
                                    self.args.layer + [1])
        self.weights = []
        self.biases = []
        for i in range(self.layer_sizes.shape[0] - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i+1]
            self.weights.append(self.weight_init.initialize(fan_in, fan_out))
            bias_shape = (1, self.layer_sizes[i+1])
            self.biases.append(np.zeros(bias_shape))

    def binary_cross_entropy(self, y_pred, y_true):
        epsilon = 1e-15
        return -np.mean(y_true * np.log(y_pred + epsilon) + (1-y_true) *
                        np.log(1-y_pred + epsilon))

    def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = (y_pred >= 0.5).astype(int)
        return np.mean(y_pred == y_true)

    def feed_forward(self, batch):
        h_l = batch
        self.hidden_layers[0] = h_l
        for i, weight in enumerate(self.weights):
            h_l = np.dot(h_l, weight)
            if i < len(self.weights) - 1:
                h_l = self.activation_function.base_function(h_l)
            else:
                h_l = self.output_activation.base_function(h_l)
            self.hidden_layers[i + 1] = h_l
        return h_l

    def back_propagation(self, output, batch_y):
        delta = output - batch_y
        nb_layers = len(self.weights)
        deltas = [0] * nb_layers
        deltas[-1] = delta
        for i in range(nb_layers - 2, -1, -1):
            delta = self.activation_function.prime_function(
                self.hidden_layers[i+1]) * \
                np.dot(deltas[i+1], self.weights[i+1].T)
            deltas[i] = delta
        for i in range(nb_layers):
            grad_w = np.dot(self.hidden_layers[i].T, deltas[i]) / \
                batch_y.shape[0]
            grad_b = np.mean(deltas[i], axis=0, keepdims=True)
            self.weights[i] -= self.learning_rate * grad_w
            self.biases[i] -= self.learning_rate * grad_b

    def fit(self):
        self.init_hyperparameters()
        n_samples = self.X_train.shape[0]

        self.y_train = np.where(self.y_train == 'B', 1, 0)
        self.y_test = np.where(self.y_test == 'B', 1, 0)

        X_train = self.X_train.to_numpy()
        y_train = self.y_train
        X_test = self.X_test.to_numpy()
        y_test = self.y_test

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        self.init_weights()

        for epoch in range(self.epochs):
            perm = np.random.permutation(n_samples)
            X_train = X_train[perm]
            y_train = y_train[perm]
            X_batches = np.array_split(X_train, n_samples / self.batch_size)
            y_batches = np.array_split(y_train, n_samples / self.batch_size)
            epoch_loss = 0
            epoch_acc = 0

            for batch_x, batch_y in zip(X_batches, y_batches):
                self.init_layers(len(batch_x))
                output = self.feed_forward(batch_x)
                epoch_loss += self.binary_cross_entropy(output, batch_y)
                epoch_acc += self.accuracy_score(batch_y, output)
                self.back_propagation(output, batch_y)

            epoch_loss /= len(X_batches)
            epoch_acc /= len(X_batches)
            self.losses.append(epoch_loss)
            self.accuracy.append(epoch_acc)

            self.init_layers(X_test.shape[0])
            val_pred = self.feed_forward(X_test)
            val_loss = self.binary_cross_entropy(val_pred, y_test)
            val_acc = self.accuracy_score(y_test, val_pred)
            self.losses_test.append(val_loss)
            self.accuracy_test.append(val_acc)

            print(f"Epoch {epoch+1}/{self.epochs} - train_loss: \
{epoch_loss:.4f} - val_loss: {val_loss:.4f} - train_acc: {epoch_acc:.4f} \
- val_acc: {val_acc:.4f}")

    def save_weights(self, activation_fn,
                     filename="weights_and_topology.pkl"):
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'topology': self.layer_sizes,
            'layer_activation': activation_fn['layer'],
            'output_activation': activation_fn['output']
        }
        with open(filename, "wb") as f:
            pickle.dump(model_data, f)
        print_info(f'Weights and topology saved in {UGREEN}{filename}')

    def load_weights(self, filename="weights_and_topology.pkl") -> None:
        with open(filename, "rb") as f:
            model_data = pickle.load(f)
        print(f"Model loaded from {UGREEN}{filename}{RESET}")
        return model_data.values()


def main():
    parser = ArgumentParser(
        description='Multilayer Perceptron')
    parser.add_argument('-dataset', type=str, default='../datasets/data.csv',
                        help='Path to a train dataset file to train the model')
    parser.add_argument('-layer', type=list_of_ints, default='26 26',
                        help='Numbers of perceptrons for each layer')
    parser.add_argument('-epochs', type=int, default=300,
                        help='Total number of iterations of all the training '
                        'data in one cycle for training the model')
    parser.add_argument('-learning-rate', type=float, default=0.1,
                        help='Hyperparameter that controls how much to change '
                        'the model when the weights are updated.')
    parser.add_argument('-batch-size', type=int, default=8,
                        help='Size of each minibatchs for SGD')
    parser.add_argument('-weights-init', type=str,
                        choices=['he', 'xavier', 'uniform'],
                        default="uniform",
                        help="Weights initialization methods : "
                        "'he', 'xavier' or 'uniform'")
    parser.add_argument('-activation', type=str,
                        choices=['sigmoid', 'relu', 'tanh'],
                        default='sigmoid',
                        help="Activation function: "
                        "'sigmoid', 'tanh' or 'relu'")
    parser.add_argument('-output-activation', type=str,
                        choices=['sigmoid', 'softmax'],
                        default='sigmoid',
                        help="Activation function for output layer: "
                        "'sigmoid' or 'softmax'")

    args = parser.parse_args()
    try:
        df = load_dataset(args.dataset)
    except IOError:
        exit()

    df = data_preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_split(df)

    initializer = WeightInitializer(method=args.weights_init)
    activation_function = ActivationFunction(function=args.activation)
    output_activation = ActivationFunction(function=args.output_activation)

    model = MultilayerPerceptron(X_train, X_test, y_train, y_test, args,
                                 weight_initializer=initializer,
                                 activation=activation_function,
                                 output_activation=output_activation)
    model.fit()
    model.save_weights({'layer': args.activation,
                        'output': args.output_activation})

    _, ax = plt.subplots(1, 2, figsize=(15, 5))
    display_loss_plot(model, ax)
    display_accuracy_score_plot(model, ax)
    plt.show()


if __name__ == '__main__':
    main()
