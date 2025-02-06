import numpy as np
import pandas as pd
from argparse import ArgumentParser
from utils import * 

pd.set_option('future.no_silent_downcasting', True)

class MultilayerPerceptron:
    def __init__(self, layers):
        pass
    
    def create_network(self):
        pass

    def fit(self):
        pass



def main():
    parser = ArgumentParser(
        description='Multilayer Perceptron')
    parser.add_argument('-dataset', type=str, default='datasets/data.csv',
        help='Path to a train dataset file to train the model')
    
    args = parser.parse_args()
    df = load_dataset(args.dataset)
    train_test_split(df)

    model = MultilayerPerceptron()


if __name__ == '__main__':
    main()
