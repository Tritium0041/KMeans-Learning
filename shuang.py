from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import warnings
import seaborn as sns
from pathlib import Path
import pandas as pd

SEED = 114514
sns.set()
warnings.filterwarnings('ignore')


# 无监督，聚类 f(x) = 2 * x + 2

# x: 1, 2, 3
# f(x): 4, 6, 8
# f_2(x): 0, 1, 0
# 有监督

def load_dataset(data_file):
    if not Path(data_file).exists():
        raise Exception("cannot find the dataset.")
    dataset = pd.read_csv(data_file, header=None)
    dataset = dataset.sample(frac=1, random_state=SEED)
    labels = dataset.pop(0)
    return dataset, labels


def main():
    X, y = load_dataset("dataset/wine.data")
    kmeans = KMeans(n_clusters=3, random_state=SEED)
    kmeans.fit(X)
    score = kmeans.score(X)
    print('score of data:', score)
    X_trans = PCA(n_components=2).fit_transform(X)
    print("X_trains shape:", X_trans.shape)
    plt.title('ground truth')
    plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y, edgecolors='b')
    plt.show()
    plt.title('kmeans pred without PCA')
    plt.scatter(X_trans[:, 0], X_trans[:, 1], c=kmeans.fit_predict(X), edgecolors='b')
    plt.show()
    plt.title('kmeans pred with PCA')
    plt.scatter(X_trans[:, 0], X_trans[:, 1], c=kmeans.fit_predict(X_trans), edgecolors='b')
    plt.show()


if __name__ == '__main__':
    main()
