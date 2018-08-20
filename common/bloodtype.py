#-*- using:utf-8 -*-
import numpy as np
from scipy.optimize import fmin

def nlikelihood(theta,n):
    """
    negative log-likelihood: 
    theta = np.array([thetaA, thetaB]); n = np.array([nA, nB, nAB, nO])
    """
    a = theta[0]; b = theta[1]; o = 1-a-b
    p = np.array([a**2+2*a*o, b**2+2*b*o, 2*a*b, o**2])
    return(-np.sum(n * np.log(p)))

def mle(n):
    """
    calculate the mle of the bloodtype model
    n = np.array([nA, nB, nAB, nO])
    initial point:[1/3,1/3]
    """
    sol = fmin(nlikelihood, [1/3,1/3], args=(n,))
    return(np.array([sol[0], sol[1], 1-np.sum(sol)]))
