import numpy as np
import pandas as pd

#各クラスタのクラス
class EM_cluster():
    def __init__(self, mu, sigma, pi):
        #initialize
        #mu, sigma is matrices
        self.mu = mu
        self.sigma = sigma
        self.pi = pi


    #現在のパラメータからxに対する事後確率を計算する
    def prob(self, x):
        sigma_inv = np.linalg.inv(self.sigma)
        form = np.dot((x - self.mu), sigma_inv)
        gauss = np.exp((-0.5) * np.dot(form, (x - self.mu).T))
        a = np.sqrt(np.linalg.det(self.sigma)*(2*np.pi)**self.sigma.ndim)
        p = gauss / a
        return p


class EM_GMM():
    def __init__(self, n_clusters, X):
        self.n_clusters = n_clusters
        self.data = X
        self.sample_size, self.dim = X.shape

        self.lh = -100000

        cluster_list = []
        for i in range(n_clusters):
            mu = X[i]
            sigma = np.matrix(np.eye(self.dim))
            cluster = EM_cluster(mu, sigma, 1/n_clusters)
            cluster_list.append(cluster)
        self.cluster_list = cluster_list

    def get_params(self):
        params = []
        for cluster in self.cluster_list:
            add = [cluster.mu[0][0]]
            add.extend(list(np.array(cluster.sigma.flatten())[0]))
            add.append(cluster.pi[0][0])

            params.extend(add)
        print(params)
        return params

    def E_step(self):
    #caluculate responsibility
    #X is data. np.array
        print("E_step...")

        def gamma_n(data, cluster_list):
            """
                argment : a data and clusters
                return : responsibility of clusters for a data.
            """
            #分母
            mom = 0
            for cluster in cluster_list:
                prob = cluster.prob(data)
                mom += prob * cluster.pi
            #クラスタ毎のgamma をリストで保持
            gamma_list = []
            for cluster in cluster_list:
                y = (cluster.pi * cluster.prob(data)) / mom
                gamma_list.append(y)
            return np.array(gamma_list)


        gammas_ = []
        for x_n in self.data:
            gamma_list = gamma_n(x_n, self.cluster_list)
            gammas_.append(gamma_list)

        self.gamma = gammas_
        return gammas_

    def M_step(self):
        """ update parameters """
        print("M-step...")

        """ N_k をクラスタごとに計算する """
        def cal_N(gamma_list, k):
            #N = np.zeros(self.n_clusters)
            N_k = 0
            for gamma in gamma_list:
                N_k = N_k + gamma[k]
            return N_k

        #N = cal_N(self.gamma)

        k = 0
        for cluster in self.cluster_list:
            gamma_for_cluster = list(map(lambda x: x[k], self.gamma))
            N_k = cal_N(self.gamma, k)


            """ update mu """
            n = 0
            new_mu = 0
            for x_n in self.data:
                gamma_nk = gamma_for_cluster[n]
                new_mu = new_mu + gamma_nk * x_n
                n += 1
            cluster.mu = new_mu / N_k

            """ update sigma """
            n = 0
            new_sigma = np.matrix(np.eye(self.dim))

            for x_n in self.data:
                gamma_nk = gamma_for_cluster[n]
                dot = np.dot((x_n - cluster.mu).T, (x_n - cluster.mu))
                new_sigma = new_sigma + dot * gamma_nk
                n += 1
            cluster.sigma = new_sigma / N_k

            """ update pi """
            cluster.pi = N_k / len(self.data)


            k += 1

        return 0

    # 対数尤度を返す、変化量が十分小さければFalseを返す
    def get_loglikelihood(self):
        loglikelihood = 0
        for x_n in self.data:
            a = 0
            for cluster in self.cluster_list:
                prob = cluster.prob(x_n)
                a += cluster.pi * prob
            loglikelihood += np.log(a)
        print("log likelihood : " + str(loglikelihood))

        if(loglikelihood - self.lh < 0.000001):
            print("finish.")
            #return True
            return False

        self.lh = loglikelihood
        return loglikelihood



if(__name__ == "__main__"):
    X = pd.read_csv("x.csv", header=None)
    X = np.array(X)

    EM = EM_GMM(4, X)

    params = []

    for i in range(1000):
        print(i+1, "iterations -----------------------------")
        EM.E_step()
        EM.M_step()

        params.append(EM.get_params())

        if(EM.get_loglikelihood()):
            pass
        else:
            gamma_ = np.array(EM.gamma).reshape(10000, 4)
            gamma_ = pd.DataFrame(gamma_)
            gamma_.to_csv("z.csv")
            break

    params = pd.DataFrame(params)
    params.to_csv("params.dat")

    labels = np.argmax(EM.gamma, axis=1)
    labels = labels.reshape(10000, )
    print(labels)
    data = X
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig=plt.figure()
    ax=Axes3D(fig)



    ax.scatter3D(data[:,0][None,:],data[:,1][None,:],data[:,2][None,:],c=labels)

    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)

    plt.savefig("EM.png")
    plt.show()
