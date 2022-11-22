#new一个吧
# 1. 随机找 k 个聚类中心点
# 2. 为每个点找出最近的聚类中心点，并归为此类
# 3. 更新聚类中心点
# 4. 重复 2,3 直到中心点不再发生变化(或者到指定轮数)
import pandas as pd
from typing import Union
import numpy as np
import random
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings

sns.set()
warnings.filterwarnings('ignore')
#这一段是ignore警告信息，seaborn进行更好的可视化结果
class KMeans:
    #先写构造函数
    def __init__(self,seed = 123, k = 3, max_iter = 100):
        #seed:随机种子，k：聚类的个数，max_iter：最大迭代数
        self.classes = None
        self.centers = None
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.n_iter = 0 #这段抄了 就是赋值

    def fit(self,X: Union[pd.DataFrame, np.ndarray]):
        #传入一个X，可以是pandas的dataframe或者numpy的ndarray
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            #把datarame转成ndarray进行下一步操作
        centers = np.array(random.sample(list(X), k=self.k))#从点阵里随机取k个样本，设置为中心点
        dist_matrix = np.zeros(shape=(len(X), self.k))#新建一个矩阵 用来存储点到中心点的距离
        nearest = np.zeros(len(X))#存储
        for _ in range(self.max_iter): #这个_好像之后都没有使用，貌似只是用来遍历的
            for idx_point, point in enumerate(X):#取X中的点 for的编号
                for idx_center, center in enumerate(centers):#这个点对于每一个取的中心点
                    dist_matrix[idx_point][idx_center] = np.sum(np.square(point - center))#求距离的平方并存入矩阵
            nearest_tmp = np.argmin(dist_matrix, axis=1)#存一下目前的距离中心最近点
            if np.sum(nearest_tmp == nearest) == len(nearest):
                break #结果相同则迭代结束 跳出循环
            else:
                nearest = nearest_tmp #将其设为最近点
            self.n_iter += 1
            for idx in range(self.k):
                group = X[np.where(nearest == idx)] #取出最近点的放入新阵列
                centers[idx] = np.sum(group, axis=0) / len(group) #取一个均值点作为新中心点
        self.centers = centers
        self.classes = np.argmin(dist_matrix, axis=1)





def main():
    data_file = "./dataset/finaldogs2.csv"
    if not Path(data_file).exists():
        raise Exception("cannot find the dataset.")
    dataset = pd.read_csv(data_file, header=None)
    dataset = dataset.sample(frac=1, random_state=312412312) #精数据集随机打乱
    #_ = dataset.pop(0) #这一行注释了也能运行啊 暂时不知作用
    kmeans = KMeans(k=5)
    kmeans.fit(dataset)
    print(kmeans.classes)
    print(kmeans.n_iter)
    dataset_trans = PCA(n_components=2).fit_transform(dataset)#用PCA给数据降维
    plt.scatter(dataset_trans[:, 0], dataset_trans[:, 1], c=kmeans.classes, edgecolors='b')
    #生成一个scatter散点图:x轴y轴是数据的两维，设置点的颜色，设置边框颜色
    plt.show()


if __name__ =="__main__":
    main()

