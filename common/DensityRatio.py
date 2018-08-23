# using:utf-8

import numpy  as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import rbf_kernel

class kernelDensityRatio:
    """
    kernel density-ratio estimator using Gaussian kernel
    gamma: bandwidth of Gaussian kernel
    lam: regularizaiton parameter
    """
    def __init__(self, gamma=None, lam=None):
        self.gamma = gamma             # カーネル幅
        self.lam = lam                 # 正則化パラメータ
        
    def fit(self, de, nu):             # 密度比推定
        if self.gamma is None:
            ma = nu.shape[0] + de.shape[0]
            idx = np.random.choice(ma,round(ma/2))
            self.gamma = (1/np.median(distance.pdist(np.r_[nu,de][idx,:])))**2
        if self.lam is None:
            self.lam = (min(nu.shape[0], de.shape[0]))**(-0.9)
        gamma = self.gamma; lam = self.lam
        n = de.shape[0]
        # グラム行列の計算
        Kdd = rbf_kernel(de, gamma=gamma)
        Kdn = rbf_kernel(de, nu, gamma=gamma)
        # 係数の推定
        Amat = Kdd + n*lam*np.identity(n)
        bvec = -np.mean(Kdn,1)/lam
        self.alpha = np.linalg.solve(Amat, bvec)
        self.de, self.nu = de, nu
        return self
    
    def predict(self, x):               # 予測点 x での密度比の値
        Wde =  np.dot(rbf_kernel(x, self.de, gamma=self.gamma), self.alpha)
        Wnu = np.mean(rbf_kernel(x, self.nu, gamma=self.gamma),1)/self.lam
        return np.maximum(Wde + Wnu,0)
