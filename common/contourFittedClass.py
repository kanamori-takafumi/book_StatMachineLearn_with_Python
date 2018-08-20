#-*- using:utf-8 -*-
import numpy  as np
import matplotlib.pyplot as plt
def cplot(clf,X,Y,h=0.02,file=None):
    """
    contour plot of decision boundary.
    clf: fitted object of svm.SVC etc.
    X: data matrix, Y: observed labels {0,1}
    h = .02: step size of the mesh
    """
    fig = plt.figure()
    plt.scatter(X[Y==0,0], X[Y==0,1], c='green', marker='+', s=80)
    plt.scatter(X[Y==1,0], X[Y==1,1], c='red',   marker='.', s=80)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    if file is None:
        plt.show()
    else:
        plt.savefig(file)

