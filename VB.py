import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, gamma
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VariationalGaussianMixture(object):

    def __init__(self, n_component=10, alpha0=1.):

        # 混合ガウス分布の混合要素数
        self.n_component = n_component

        # 混合係数piの事前分布のパラメータ
        self.alpha0 = alpha0

    def init_params(self, X):
        self.sample_size, self.ndim = X.shape

        # 事前分布のパラメータを設定
        self.alpha0 = np.ones(self.n_component) * self.alpha0
        self.m0 = np.zeros(self.ndim)
        self.W0 = np.eye(self.ndim)
        self.nu0 = self.ndim
        self.beta0 = 1.

        self.component_size = self.sample_size / self.n_component + np.zeros(self.n_component)

        # 確率分布のパラメータを初期化
        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        indices = np.random.choice(self.sample_size, self.n_component, replace=False)
        self.m = X[indices].T
        self.W = np.tile(self.W0, (self.n_component, 1, 1)).T
        self.nu = self.nu0 + self.component_size

    # 確率分布のパラメータを返す
    def get_params(self):
        return self.alpha, self.beta, self.m, self.W, self.nu

    # 変分ベイズ法
    def fit(self, X, iter_max=100):
        self.init_params(X)

        self.param_list = []

        # 確率分布が収束するまで更新する
        for i in range(iter_max):
            params = np.hstack([array.flatten() for array in self.get_params()])

            # 確率分布q(z)の更新
            r = self.e_like_step(X)
            # 確率分布q(pi, mu, Lambda)の更新
            self.m_like_step(X, r)

            # parameterをえる
            #print(self.get_params())
            self.param_list.extend([self.get_params()])


            # 収束していれば終了
            if np.allclose(params, np.hstack([array.ravel() for array in self.get_params()])):
                r_ = pd.DataFrame(r)
                # csv に保存
                r_.to_csv("z_VB.csv")
                break
        else:
            print("parameters may not have converged")

    # 確率分布q(z)の更新
    def e_like_step(self, X):
        d = X[:, :, None] - self.m
        gauss = np.exp(
            -0.5 * self.ndim / self.beta
            - 0.5 * self.nu * np.sum(
                np.einsum('ijk,njk->nik', self.W, d) * d,
                axis=1)
        )
        pi = np.exp(digamma(self.alpha) - digamma(self.alpha.sum()))
        Lambda = np.exp(digamma(self.nu - np.arange(self.ndim)[:, None]).sum(axis=0) + self.ndim * np.log(2) + np.linalg.slogdet(self.W.T)[1])

        # 負担率の計算
        r = pi * np.sqrt(Lambda) * gauss

        # 負担率を正規化
        r /= np.sum(r, axis=-1, keepdims=True)

        # 0割が生じた場合に備えて
        r[np.isnan(r)] = 1. / self.n_component
        return r

    # 確率分布q(pi, mu, Lambda)の更新
    def m_like_step(self, X, r):


        self.component_size = r.sum(axis=0)


        Xm = X.T.dot(r) / self.component_size

        d = X[:, :, None] - Xm

        S = np.einsum('nik,njk->ijk', d, r[:, None, :] * d) / self.component_size

        # q(pi)のパラメータを更新
        self.alpha = self.alpha0 + self.component_size


        self.beta = self.beta0 + self.component_size


        self.m = (self.beta0 * self.m0[:, None] + self.component_size * Xm) / self.beta

        d = Xm - self.m0[:, None]

        self.W = np.linalg.inv(
            np.linalg.inv(self.W0)
            + (self.component_size * S).T
            + (self.beta0 * self.component_size * np.einsum('ik,jk->ijk', d, d) / (self.beta0 + self.component_size)).T).T

        self.nu = self.nu0 + self.component_size

    def classify(self, X):
        # rのargmax
        return np.argmax(self.e_like_step(X), 1)



def main():
    X = np.array(pd.read_csv("x.csv", header=None))

    model = VariationalGaussianMixture(n_component=4, alpha0=0.01)
    model.fit(X, iter_max=100)

    param = pd.DataFrame(model.param_list)
    param.to_csv("params_VB.dat")

    labels = model.classify(X)
    labels = labels.reshape(10000, )

    data = X
    fig=plt.figure()
    ax=Axes3D(fig)
    ax.scatter3D(data[:,0][None,:],data[:,1][None,:],data[:,2][None,:],c=labels)
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
    plt.savefig("VB.png")
    plt.show()


if __name__ == '__main__':
    main()
