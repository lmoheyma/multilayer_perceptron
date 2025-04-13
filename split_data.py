from utils import train_test_split, load_dataset
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Use a dataset to display accuracy score a model")
    parser.add_argument('-dataset', type=str,
                        default='datasets/data.csv',
                        help="Path to the dataset you want to predict.")
    parser.add_argument("-test-size", type=float,
                        default=0.25,
                        help="Percentage for the test dataset")
    args = parser.parse_args()
    try:
        df = load_dataset(args.dataset)
    except IOError:
        exit()
    train_test_split(df, args.test_size)
