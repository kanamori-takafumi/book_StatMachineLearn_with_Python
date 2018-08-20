#-*- using:utf-8 -*-
# Python codes of spirals, twoDnormals in mlbench library of R
import numpy as np
from scipy.special import gamma
import sys

def onespiral(n, cycles=1, sd=0):
    w = np.linspace(0,cycles,n)
    x = np.zeros((n,2))
    x[:,0] = (2*w+1)*np.cos(2*np.pi*w)/3
    x[:,1] = (2*w+1)*np.sin(2*np.pi*w)/3
    if sd > 0:
        e = np.random.normal(scale=sd,size=n)
        xs = np.cos(2*np.pi*w) - np.pi*(2*w+1)*np.sin(2*np.pi*w)
        ys = np.sin(2*np.pi*w) + np.pi*(2*w+1)*np.cos(2*np.pi*w)
        nrm = np.sqrt(xs**2 + ys**2)
        x[:,0] = x[:,0] + e*ys/nrm
        x[:,1] = x[:,1] - e*xs/nrm
    return(x)

def spirals(n, cycles=1, sd=0, label=[0,1]):
    x = np.zeros((n,2))
    c2 = np.random.choice(n,size=round(n/2),replace=False)
    c1 = np.delete(np.arange(n),c2)
    cl = np.repeat(label[0],n)
    cl[c2] = label[1]
    x[c1,:] =  onespiral(c1.size, cycles=cycles, sd=sd)
    x[c2,:] = -onespiral(c2.size, cycles=cycles, sd=sd)
    return([x,cl])

def twoDnormals(n, cl=2, sd=1, r=None):
    if r is None:
        r = np.sqrt(cl)
    e = np.random.choice(cl,size=n)
    m = r*np.c_[np.cos(np.pi/4 + e*2*np.pi/cl), np.sin(np.pi/4 + e*2*np.pi/cl)]
    x = np.random.normal(scale=sd,size=2*n).reshape(n,2) + m
    return([x, e])

def circle(n, d=2):
    if not isinstance(d,int) or (d<2):
        print("d must be an integer >=2")
        sys.exit()
    x = np.random.uniform(low=-1,high=1,size=n*d).reshape(n,d)
    z = np.repeat(1,n)
    r = (2**(d-1) * gamma(1+d/2)/(np.pi**(d/2)))**(1/d)
    z[np.sum(x**2,1) > r**2] = 2
    return([x,z])
