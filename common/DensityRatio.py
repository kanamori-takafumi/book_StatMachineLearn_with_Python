#-*- using:utf-8 -*-
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
        self.gamma = gamma
        self.lam = lam
        
    def fit(self, de, nu):
        gamma = self.gamma
        lam = self.lam
        if gamma is None:
            ma = nu.shape[0] + de.shape[0]
            idx = np.random.choice(ma,round(ma/2))
            gamma = (1/np.median(distance.pdist(np.r_[nu,de][idx,:])))**2
        if lam is None:
            lam = (min(nu.shape[0], de.shape[0]))**(-0.9)
        n = de.shape[0]
        Kdd = rbf_kernel(de, gamma=gamma)
        Kdn = rbf_kernel(de, nu, gamma=gamma)
        Amat = Kdd + n*lam*np.identity(n)
        bvec = -np.mean(Kdn,1)/lam
        self.alpha = np.linalg.solve(Amat, bvec)
        self.gamma, self.lam = gamma, lam
        self.de, self.nu = de, nu
        return self
    
    def predict(self, x):
        Wde =  np.dot(rbf_kernel(x, self.de, gamma=self.gamma), self.alpha)
        Wnu = np.mean(rbf_kernel(x, self.nu, gamma=self.gamma),1)/self.lam
        return np.maximum(Wde + Wnu,0)
