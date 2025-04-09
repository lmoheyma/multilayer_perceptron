import argparse
from utils import *
from weights_init import WeightInitializer
from activation_function import ActivationFunction
from train_mlp import MultilayerPerceptron

def main():
    parser = argparse.ArgumentParser(description="Use a dataset to predict the output of our logistic regression model.")
    parser.add_argument(
        "-db", "--dataset",
        type=str,
        help="Path to the dataset you want to predict.",
        default='datasets/data.csv')
    parser.add_argument(
        "-w", "--weights",
        type=str,
        help="Path to the file containing the weights to be used for the prediction.",
        default='weights.pkl')

    args = parser.parse_args()
    df = load_dataset(args.dataset)

    df = data_preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_split(df)

    model = MultilayerPerceptron(X_train, X_test, y_train, y_test, args)
    try:
        model.weights, model.biases, model.layer_sizes = model.load_weights(args.weights)
    except Exception as e:
        print(f"{BHRED}Fail to read file '{RED}{args.weights}{BHRED}'.{RESET}")
        exit(1)
    print(df)
    y = df['Diagnosis']
    X = df.select_dtypes('float64')

if __name__ == '__main__':
    main()