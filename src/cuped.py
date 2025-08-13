import numpy as np

def cuped_adjust(y, theta):
    cov = np.cov(y, theta, ddof=1)
    var = np.var(theta, ddof=1)
    if var == 0:
        return y, 0.0
    c = cov / var  # compute constant c
    y_adj = y - c * theta 
    return y_adj, c