import argparse
from utils import *
from weights_init import WeightInitializer
from activation_function import ActivationFunction
from train_mlp import MultilayerPerceptron

def main():
    parser = argparse.ArgumentParser(description="Use a dataset to predict the output of our logistic regression model.")
    parser.add_argument("-db", "--dataset",type=str, default='datasets/data.csv',
        help="Path to the dataset you want to predict.")
    parser.add_argument("-w", "--weights", type=str, default='weights.pkl',
        help="Path to the file containing the weights to be used for the prediction.")
    parser.add_argument('-target', type=str, default='Diagnosis',
        help='Name of the target feature to predict')

    args = parser.parse_args()
    try:
        df = load_dataset(args.dataset)
    except IOError:
        exit()
    df = data_preprocessing(df)

    activation_function = ActivationFunction()
    model = MultilayerPerceptron(X_train=None, X_test=None,
                                 y_train=None, y_test=None,
                                 args=args, activation=activation_function)

    try:
        model.weights, model.biases, model.layer_sizes, model.activation_function.function = model.load_weights(args.weights)
        y = np.where(df[args.target] == 'B', 1, 0)
    except KeyError as e:
        print(f'{BHRED}Invalid target: {RED}{e}{BHRED}.{RESET}')
        exit()
    except Exception:
        print(f"{BHRED}Fail to read file '{RED}{args.weights}{BHRED}'.{RESET}")
        exit(1)
    X = df.select_dtypes('float64').to_numpy()

    model.init_layers(len(X))
    pred = model.feed_forward(X)
    pred = [1 if x > 0.5 else 0 for x in pred]
    print(f'Accuracy score: {np.mean(pred == y)}')

if __name__ == '__main__':
    main()