# -*- coding: gbk -*-

import numpy as np
import copy
from sklearn.datasets import make_moons
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt


class MoG():
    def __init__(self, k=3, max_iter=20, e=0.0001):
        self.k = k
        self.max_iter = max_iter
        self.e = e
        self.ll = 0

    def dist(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def get_miu(self, X):
        index = np.random.choice(X.shape[0], 1, replace=False)
        mius = []
        mius.append(X[index])

        for _ in range(self.k - 1):
            max_dist_index = 0
            max_distance = 0
            for j in range(X.shape[0]):
                min_dist_with_miu = 999999

                for miu in mius:
                    dist_with_miu = self.dist(miu, X[j])
                    if min_dist_with_miu > dist_with_miu:
                        min_dist_with_miu = dist_with_miu

                if max_distance < min_dist_with_miu:
                    max_distance = min_dist_with_miu
                    max_dist_index = j
            mius.append(X[max_dist_index])

        mius_array = np.array([])
        for i in range(self.k):
            if i == 0:
                mius_array = mius[i]
            else:
                mius[i] = mius[i].reshape(mius[0].shape)
                mius_array = np.append(mius_array, mius[i], axis=0)

        return mius_array

    def p(self, x, label):

        miu = self.mius_array[label].reshape(1, -1)

        covdet = np.linalg.det(self.Sigma[label])  # 计算|cov|
        covinv = np.linalg.inv(self.Sigma[label])  # 计算cov的逆

        if covdet < 1e-5:              # 以防行列式为0
            covdet = np.linalg.det(
                self.Sigma[label] + np.eye(x.shape[0]) * 0.001)
            covinv = np.linalg.inv(
                self.Sigma[label] + np.eye(x.shape[0]) * 0.001)
        a = np.float_power(
            2 * np.pi, x.shape[0] / 2) * np.float_power(covdet, 0.5)
        b = -0.5 * (x - miu)@covinv@(x - miu).T
        return 1 / a * np.exp(b)

    def pM(self, x, label):
        pm = 0
        for i in range(self.k):
            pm += self.Alpha[i] * self.p(x, i)
        return self.Alpha[label] * self.p(x, label) / pm

    def update_miu(self, X, label):

        a = 0
        b = 0
        for i in range(X.shape[0]):
            a += self.Gamma[i][label] * X[i]
            b += self.Gamma[i][label]
        if b == 0:
            b = 1e-10
        return a / b

    def update_sigma(self, X, label):

        a = 0
        b = 0
        for i in range(X.shape[0]):
            X[i] = X[i].reshape(1, -1)
            miu = self.mius_array[label].reshape(1, -1)
            a += self.Gamma[i][label] * \
                (X[i] - miu).T@(X[i] - miu)
            b += self.Gamma[i][label]
        if b == 0:
            b = 1e-10
        sigma = a / b
        return sigma

    def update_alpha(self, X, label):

        a = 0
        for i in range(X.shape[0]):
            a += self.Gamma[i][label]
        return a / X.shape[0]

    def LL(self, X):
        ll = 0
        for i in range(X.shape[0]):
            before_ln = 0
            for j in range(self.k):
                before_ln += self.Alpha[j] * self.Gamma[i][j]
            ll += np.log(before_ln)
        return ll

    def fit(self, X):
        self.Alpha = np.array([1 / self.k] * self.k)  # 初始化alpha
        self.mius_array = self.get_miu(X)  # 初始化miu
        self.Sigma = np.array(
            [np.eye(X.shape[1], dtype=float) * 0.1] * self.k)  # 初始化sigma
        self.Gamma = np.zeros([X.shape[0], self.k])

        Y = np.zeros([X.shape[0]])
        iter = 0
        while(iter < self.max_iter):
            old_ll = self.ll
            for i in range(X.shape[0]):
                for j in range(self.k):
                    self.Gamma[i][j] = self.pM(X[i], j)
            for i in range(self.k):
                self.mius_array[i] = self.update_miu(X, i)
                self.Sigma[i] = self.update_sigma(X, i)
                self.Alpha[i] = self.update_alpha(X, i)
            self.ll = self.LL(X)
            print(self.ll)
            if abs(self.ll - old_ll) < 0.01:
                print('迭代{}次退出'.format(iter))
                break
            iter += 1

        if iter == self.max_iter:
            print("迭代超过{}次，退出迭代".format(self.max_iter))
        for i in range(X.shape[0]):
            tmp_y = -1
            tmp_gamma = -1
            for j in range(self.k):
                if tmp_gamma < self.Gamma[i][j]:
                    tmp_gamma = self.Gamma[i][j]
                    tmp_y = j
            Y[i] = tmp_y
        return Y


if __name__ == '__main__':

    fig = plt.figure(1)

    plt.subplot(221)
    center = [[1, 1], [-1, -1], [1, -1]]
    cluster_std = 0.35
    X1, Y1 = make_blobs(n_samples=1000, centers=center,
                        n_features=2, cluster_std=cluster_std, random_state=1)
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)

    plt.subplot(222)
    km1 = MoG(k=3, max_iter=20)
    km_Y1 = km1.fit(X1)
    mius = km1.mius_array
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=km_Y1)
    plt.scatter(mius[:, 0], mius[:, 1], marker='^', c='r')

    plt.subplot(223)
    X2, Y2 = make_moons(n_samples=1000, noise=0.1)
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2)

    plt.subplot(224)
    km2 = MoG(k=2, max_iter=20)
    km_Y2 = km2.fit(X2)
    mius = km2.mius_array
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=km_Y2)
    plt.scatter(mius[:, 0], mius[:, 1], marker='^', c='r')

    plt.show()
