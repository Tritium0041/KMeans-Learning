# 1. 随机找 k 个聚类中心点
# 2. 为每个点找出最近的聚类中心点，并归为此类
# 3. 更新聚类中心点
# 4. 重复 2,3 直到中心点不再发生变化(或者到指定轮数)

import numpy as np
import jax.numpy as jnp
from jax import vmap
import pandas as pd
import seaborn as sns
from typing import Union
from matplotlib import pyplot as plt
import random
from pathlib import Path

from sklearn.decomposition import PCA
import warnings

sns.set()
SEED = 1130
random.seed(SEED)
warnings.filterwarnings('ignore')


class KMeansClassification:

    def __init__(self, n_clusters=3, random_state=SEED, max_iter=50):
        self.classes = None
        self.k_center = None
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.iter_num = 0

    def dist(self, data):
        dist_square = lambda centers, point: jnp.sum(jnp.square(point - centers), axis=1)
        return vmap(dist_square, in_axes=(None, 0), out_axes=0)(self.k_center, data)

    def fit(self, data: Union[pd.DataFrame, np.ndarray]):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        self.k_center = np.array(random.sample(list(data), k=self.n_clusters))
        k_classes = np.zeros(len(data))
        self.iter_num = 0
        for _ in range(self.max_iter):
            k_classes_next = np.argmin(np.array(self.dist(data)), axis=1)
            if np.sum(k_classes == k_classes_next) == len(k_classes):
                break
            else:
                k_classes = k_classes_next
            for i in range(self.n_clusters):
                where = data[np.where(k_classes == i)]
                self.k_center[i] = (np.sum(where, axis=0) / len(where))
            self.iter_num += 1
        self.classes = k_classes

    def score(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        assert self.k_center is not None
        return np.sum(np.sqrt(self.dist(data))) / len(data) / self.n_clusters


if __name__ == '__main__':
    data_file = "./dataset/wine.data"
    if not Path(data_file).exists():
        raise Exception("cannot find the dataset.")
    dataset = pd.read_csv(data_file, header=None)
    dataset = dataset.sample(frac=1, random_state=SEED)
    _ = dataset.pop(0)
    scores = []
    ks = list(range(2, 10))
    for k in ks:
        kmeans = KMeansClassification(n_clusters=k)
        kmeans.fit(dataset)
        scores.append(kmeans.score(dataset))
    plt.plot(ks, scores)
    # dataset_trans = PCA(n_components=2).fit_transform(dataset)
    # plt.scatter(dataset_trans[:, 0], dataset_trans[:, 1], c=kmeans.classes, edgecolors='b')
    plt.show()
