import numpy as np
from scipy.linalg import cho_factor, cho_solve

def direct_solve(A, b, l=None, y=None):
    M = np.dot(A.T, A)
    rhs = np.dot(b, A)
    if l is not None:
        M = M + l*np.eye(M.shape[0])
        if y is not None:
            rhs = rhs + l*y
    # scipy doesnt seem to batch
    # c, low = cho_factor(M)
    # x = cho_solve((c, low), rhs)
    L = np.linalg.cholesky(M,upper=False)
    Linv_rhs = np.dot(rhs, np.linalg.inv(L.T))
    x = np.dot(Linv_rhs, np.linalg.inv(L))
    return x

def direct_solve_reverse(A, b, l, y=None):
    M = np.dot(A, A.T) + l*np.eye(A.shape[0])
    rhs = np.dot(b, A) 
    if y is not None:
        rhs += l*y
    x = rhs/l
    Arhs = np.dot(rhs, A.T)
    # scipy doesnt seem to batch
    #c, low = cho_factor(M)
    #lhs = cho_solve((c, low), Arhs)
    L = np.linalg.cholesky(M,upper=False)
    Linv_Arhs = np.dot(Arhs, np.linalg.inv(L.T))
    lhs = np.dot(Linv_Arhs, np.linalg.inv(L))
    x -= 1.0/l*np.dot(lhs, A)
    return x





