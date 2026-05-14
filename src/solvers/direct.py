import numpy as np
from scipy.linalg import cho_factor, cho_solve

def direct_solve(A, b, l=None, y=None):
    M = np.dot(A.T, A)
    rhs = np.dot(A.T, b)
    if l is not None:
        M = M + l*np.eye(M.shape[0])
        print(f"{M=}")
        if y is not None:
            rhs = rhs + l*y
    c, low = cho_factor(M)
    x = cho_solve((c, low), rhs)
    return x

def direct_solve_reverse(A, b, l, y=None):
    M = np.dot(A, A.T) + l*np.eye(A.shape[0])
    rhs = np.dot(A.T, b) 
    if y is not None:
        rhs += l*y
    x = rhs/l
    Arhs = np.dot(A, rhs)
    c, low = cho_factor(M)
    lhs = cho_solve((c, low), Arhs)
    x -= 1.0/l*np.dot(A.T, lhs)
    return x





