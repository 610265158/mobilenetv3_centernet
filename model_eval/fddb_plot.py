import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

roc = pd.read_csv('/home/lz/WiderFace-Evaluation/fddb/evaluation/tempDiscROC.txt', sep=' ', header=None)
roc.columns = ['tpr', 'fp', 'threshold']


def plot_roc():
    _, axis = plt.subplots(nrows=1, ncols=1, figsize=(7, 4), dpi=120)
    axis.plot(roc.fp, roc.tpr, c='r', linewidth=2.0);
    axis.set_title('Discrete Score ROC')
    axis.set_xlim([0, 2000.0])
    axis.set_ylim([0.6, 1.0])
    axis.set_xlabel('False Positives')
    axis.set_ylabel('True Positive Rate');
    plt.show()
plot_roc()