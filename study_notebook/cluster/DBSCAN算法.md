DBSCAN算法


```python
# -*- coding: gbk -*-

import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.datasets import make_moons
from sklearn.datasets.samples_generator import make_blobs
import random


class DBSCAN():
    def __init__(self, epsilon, MinPts):
        self.epsilon = epsilon
        self.MinPts = MinPts

    def dist(self, x1, x2):
        return np.linalg.norm(x1 - x2)
    
    def getCoreObjectSet(self, X):
        N = X.shape[0]
        Dist = np.eye(N) * 9999999
        CoreObjectIndex = []
        for i in range(N):
            for j in range(N):
                if i > j:
                    Dist[i][j] = self.dist(X[i], X[j])
        for i in range(N):
            for j in range(N):
                if i < j:
                    Dist[i][j] = Dist[j][i]
        for i in range(N):
            # 获取对象周围小于epsilon的点的个数
            dist = Dist[i]
            num = dist[dist < self.epsilon].shape[0]
            if num >= self.MinPts:
                CoreObjectIndex.append(i)
        return np.array(CoreObjectIndex), Dist
    
    def element_delete(self, a, b):
        if isinstance(b, np.ndarray) == False:
            b = np.array([b])
        for i in range(b.shape[0]):
            index = np.where(a == b[i])
            a = np.delete(a, index[0])
        return a
    
    def fit(self, X):
        N = X.shape[0]
        CoreObjectIndex, Dist = self.getCoreObjectSet(X)
        self.k = 0
        self.C = []
        UnvisitedObjectIndex = np.arange(N)
    
        while(CoreObjectIndex.shape[0] != 0):
            old_UnvisitedObjectIndex = copy.deepcopy(
                UnvisitedObjectIndex)  # 记录当前未访问的样本id
            OriginIndex = np.random.choice(
                CoreObjectIndex.shape[0], 1, replace=False)  # 随机选取一个核心对象
            Queue = np.array([-1, CoreObjectIndex[OriginIndex]])  # 初始化队列
    
            CoreObjectIndex = self.element_delete(
                CoreObjectIndex, CoreObjectIndex[OriginIndex])  # 将核心对象id从id集合中除去
            while(Queue.shape[0] != 1):
                # 取出队列中首个样本id
                index = Queue[0]
                if index == -1:
                    Queue = np.delete(Queue, 0)
                    Queue = np.append(Queue, -1)
                    continue
    
                Queue = self.element_delete(Queue, index)
                index = int(index)
                DistWithOthers = Dist[index]
                OthersIndex = np.where(DistWithOthers < self.epsilon)[0]
                num = OthersIndex.shape[0]
                if num >= self.MinPts:
                    delta = list(set(OthersIndex).intersection(
                        set(UnvisitedObjectIndex)))  # 取核心对象内的样本和未访问样本集合的交集
                    delta = np.array(delta)
                    Queue = np.append(Queue, delta)
                    UnvisitedObjectIndex = self.element_delete(
                        UnvisitedObjectIndex, delta)
    
            self.k += 1
            self.C.append(
                self.element_delete(old_UnvisitedObjectIndex, UnvisitedObjectIndex))
            CoreObjectIndex = self.element_delete(
                CoreObjectIndex, self.C[self.k - 1])
        print("共有{} 个簇".format(self.k))
    
        Y = np.zeros(X.shape[0])
        for i in range(self.k):
            Y[self.C[i]] = i + 3
    
        return Y
    
    def plt_show(self, X, Y, pre_Y, name=0):
        if X.shape[1] == 2:
            fig = plt.figure(name)
            plt.subplot(211)
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
            plt.subplot(212)
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=pre_Y)
            plt.colorbar()
        else:
            print('error arg')


if __name__ == '__main__':

    center = [[1, 1], [-1, -1], [1, -1]]
    cluster_std = 0.35
    X1, Y1 = make_blobs(n_samples=300, centers=center,
                        n_features=2, cluster_std=cluster_std, random_state=1)
    
    dbscan1 = DBSCAN(epsilon=0.4, MinPts=5)
    pre_Y1 = dbscan1.fit(X1)
    dbscan1.plt_show(X1, Y1, pre_Y1, name=1)
    
    center = [[1, 1], [-1, -1], [2, -2]]
    cluster_std = [0.35, 0.1, 0.8]
    X2, Y2 = make_blobs(n_samples=300, centers=center,
                        n_features=2, cluster_std=cluster_std, random_state=1)
    
    dbscan2 = DBSCAN(epsilon=0.4, MinPts=5)
    pre_Y2 = dbscan2.fit(X2)
    dbscan2.plt_show(X2, Y2, pre_Y2, name=2)
    
    X3, Y3 = make_moons(n_samples=1000, noise=0.1)
    dbscan3 = DBSCAN(epsilon=0.1, MinPts=5)
    pre_Y3 = dbscan3.fit(X3)
    dbscan3.plt_show(X3, Y3, pre_Y3, name=3)
    
    plt.show()
```

