# coding: utf-8

import numpy as np
def EMmixBernoulli(x, K=5, maxitr=1000, tol=1e-5):
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
    # 次元dとデータ数n
    n,d = x.shape
    eps = np.finfo(float).eps
    # コンポーネント初期設定
    mu = np.mean(x)
    p = np.random.beta(mu, 1-mu, size=K*d).reshape(K,d)
    # 混合確率
    q = np.repeat(1/K,K)
    ul = np.inf
    for itr in np.arange(maxitr): # EMアルゴリズム
        # 多次元ベルヌーイ分布の確率を計算
        mp = (np.exp(np.dot(np.log(p), x.T) + np.dot(np.log(1-p), 1-x.T)).T * q).T
        # γ, q, p 更新．pmin, pmax で発散を防ぐ．
        gmm = np.clip(mp/np.sum(mp,0),eps,1-eps)
        q = np.clip(np.sum(gmm,1)/n,eps,1-eps)
        p = np.clip((np.dot(gmm,x).T/(n*q)).T,eps,1-eps)
        # 負の対数尤度の上界
        lp = np.dot(np.log(p),x.T) + np.dot(np.log(1-p),1-x.T)
        uln = -np.sum(gmm*((lp.T + np.log(q)).T-np.log(gmm)))
        if np.abs(ul -uln)<tol: # 停止条件
            break
        ul = uln
    BIC = ul+0.5*(d*K+(K-1))*np.log(n) # BIC
    return([p,q,BIC,gmm])

