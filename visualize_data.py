import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import load_dataset
from colors import *

class DataVisualizer:
    def __init__(self, df):
        self.df = df
        self.features = df.select_dtypes('float64')

    def histogram(self):
        nb_col = 5
        nb_row = 2
        plt.style.use('ggplot')
        _, ax = plt.subplots(nb_row, nb_col, tight_layout=True, figsize=(15, 8))
        for i, col in enumerate(self.features.iloc[:, :10]): # 10 first features
            plot = ax[int(i / nb_col), i % nb_col]
            for key, grp in self.df.groupby(['Diagnosis']):
                plot.hist(grp[col], alpha=0.5, label = key)
            plot.set_title(col, fontsize=10)
        handles, labels = ax[0, 0].get_legend_handles_labels()
        plt.legend(handles, labels, loc='best', borderpad=1.5, prop={'size': 12})
        self.save_plot('histogram')

    def correlation_heatmap(self):
        plt.figure(figsize=(18, 8))
        heatmap = sns.heatmap(self.features.corr(), vmin=-1, vmax=1, annot=True)
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
        self.save_plot('correlation_heatmap')

    def save_plot(self, filename: str) -> None:
        try:
            if not os.path.isdir('plots'):
                os.mkdir('plots')
            plt.savefig(f'plots/{filename}.png')
        except Exception as e:
            print(f"{BHRED}Fail to save file '{RED}{filename}{BHRED}'.{RESET}")
            exit(1)


def main():
    df = load_dataset('datasets/data.csv')
    # df.drop('')
    visualizer = DataVisualizer(df)
    # visualizer.histogram()
    visualizer.correlation_heatmap()
    plt.show()

if __name__ == '__main__':
    main()
