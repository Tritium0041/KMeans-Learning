import pandas as pd
from typing import Union
import numpy as np
import random
from pathlib import Path

# 1. 随机找 k 个聚类中心点
# 2. 为每个点找出最近的聚类中心点，并归为此类
# 3. 更新聚类中心点
# 4. 重复 2,3 直到中心点不再发生变化(或者到指定轮数)
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings

sns.set()
warnings.filterwarnings('ignore')


class KMeans:

    def __init__(self, k=2, seed=114514, max_iter=50):
        self.classes = None
        self.centers = None
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.n_iter = 0

    def fit(self, X: Union[pd.DataFrame, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        centers = np.array(random.sample(list(X), k=self.k))
        dist_matrix = np.zeros(shape=(len(X), self.k))
        nearest = np.zeros(len(X))
        for _ in range(self.max_iter):
            for idx_point, point in enumerate(X):
                for idx_center, center in enumerate(centers):
                    dist_matrix[idx_point][idx_center] = np.sum(np.square(point - center))
            nearest_tmp = np.argmin(dist_matrix, axis=1)
            if np.sum(nearest_tmp == nearest) == len(nearest):
                break
            else:
                nearest = nearest_tmp
            self.n_iter += 1
            for idx in range(self.k):
                group = X[np.where(nearest == idx)]
                centers[idx] = np.sum(group, axis=0) / len(group)

        self.centers = centers
        self.classes = np.argmin(dist_matrix, axis=1)


if __name__ == '__main__':
    data_file = "./dataset/wine.data"
    if not Path(data_file).exists():
        raise Exception("cannot find the dataset.")
    dataset = pd.read_csv(data_file, header=None)
    dataset = dataset.sample(frac=1, random_state=114514)
    _ = dataset.pop(0)
    kmeans = KMeans(k=2)
    kmeans.fit(dataset)
    print(kmeans.classes)
    print(kmeans.n_iter)
    dataset_trans = PCA(n_components=2).fit_transform(dataset)
    plt.scatter(dataset_trans[:, 0], dataset_trans[:, 1], c=kmeans.classes, edgecolors='b')
    plt.show()
