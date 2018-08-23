# using:utf-8

import numpy as np

class EMmixBernoulli:
    """
    EM algorithm for Mixture of Bernoulli models
    
    Parameters:
    ---
    x: data matrix
    K: number of components
        
    Returns: 
    ---
    p: likelihood of each component
    q: mixing probability
    BIC: Bayesian information criterion
    gmm: auxiliary parameter in EM algorithm
    """
    
    def __init__(self, K=None, maxitr=None, tol=None):
        self.K = K
        self.maxitr = maxitr
        self.tol = tol

    def fit(self, x):
        """
        EMアルゴリズムでパラメータ推定
        """
        K = self.K; maxitr = self.maxitr; tol = self.tol
        if K is None:
            K=5
        if maxitr is None:
            maxitr = 1000
        if tol is None:
            tol = 1e-5
        n,d = x.shape                    # データ数nと次元d
        eps = np.finfo(float).eps        
                                         # コンポーネント初期設定
        mu = np.mean(x)
        p = np.random.beta(mu, 1-mu, size=K*d).reshape(K,d)
        q = np.repeat(1/K,K)             # 混合確率
        ul = np.inf
        converge_ = False
        for itr in np.arange(maxitr):    # EMアルゴリズム
                                         # 多次元ベルヌーイ分布の確率を計算
            mp = (np.exp(np.dot(np.log(p), x.T) + np.dot(np.log(1-p), 1-x.T)).T * q).T
                                         # gmm, q, p 更新．pmin, pmax で発散を防ぐ．
            gmm = np.clip(mp/np.sum(mp,0),eps,1-eps)
            q = np.clip(np.sum(gmm,1)/n,eps,1-eps)
            p = np.clip((np.dot(gmm,x).T/(n*q)).T,eps,1-eps)
                                         # 負の対数尤度の上界
            lp = np.dot(np.log(p),x.T) + np.dot(np.log(1-p),1-x.T)
            uln = -np.sum(gmm*((lp.T + np.log(q)).T-np.log(gmm)))
            if np.abs(ul-uln)<tol:       # 停止条件
                converge_ = True
                break
            ul = uln
        BIC = ul+0.5*(d*K+(K-1))*np.log(n) # BIC
        self.p = p; self.q = q; self.BIC = BIC; self.gmm = gmm
        self.converge_ = converge_; self.itr = itr
        return self

    def predict_proba(self, newx):
        """
        各クラスタへの newx の所属確率
        """
        p = self.p; q = self.q
                                         # 同時確率
        jp = np.exp(np.dot(np.log(p), newx.T) + np.dot(np.log(1-p), 1-newx.T)).T * q
        mp = np.sum(jp,1)                # 周辺確率
        cp = (jp.T/mp).T                 # 条件付き確率
        return cp
    
    def predict(self, newx):
        """
        newx のクラスタリング
        """
        cp = self.predict_proba(newx)    # 条件付き確率の計算
        cl = np.argmax(cp,axis=1)
        return cl

    def set_K(self, K):
        """
        コンポーネント設定
        """
        self.K = K
        return self
