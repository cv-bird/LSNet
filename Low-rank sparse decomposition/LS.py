import numpy as np
from numpy.linalg import svd, norm

def shrinkage(X, tau):
    """Element-wise soft-thresholding operator"""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def svd_shrinkage(X, tau):
    """Singular value soft-thresholding"""
    U, S, Vt = svd(X, full_matrices=False)
    S_shrink = shrinkage(np.diag(S), tau)
    return U @ S_shrink @ Vt

def robust_pca(X, lam=None, mu=None, tol=1e-6, max_iter=500):
    """
    using ADMM
    Parameters
    ----------
    X : np.ndarray
        Input data matrix (MxN), can contain np.nan for missing values.
    lam : float, optional
        Regularization parameter (default 1/sqrt(max(M,N)))
    mu : float, optional
        Augmented Lagrangian parameter (default 10*lam)
    tol : float, optional
        Tolerance for convergence (default 1e-6)
    max_iter : int, optional
        Maximum number of iterations (default 500)
    Returns
    -------
    L : np.ndarray
        Low-rank component
    S : np.ndarray
        Sparse component
    """
    X = X.copy()
    M, N = X.shape
    unobserved = np.isnan(X)
    X[unobserved] = 0
    normX = norm(X, 'fro')

    if lam is None:
        lam = 1 / np.sqrt(max(M, N))
    if mu is None:
        mu = 10 * lam

    L = np.zeros((M, N))
    S = np.zeros((M, N))
    Y = np.zeros((M, N))

    for iter in range(1, max_iter+1):
        # update L and S
        L = svd_shrinkage(X - S + (1/mu)*Y, 1/mu)
        S = shrinkage(X - L + (1/mu)*Y, lam/mu)
        
        # update Lagrange multiplier
        Z = X - L - S
        Z[unobserved] = 0
        Y = Y + mu*Z

        err = norm(Z, 'fro') / normX
        if iter == 1 or iter % 10 == 0 or err < tol:
            print(f"iter: {iter:04d}\terr: {err:.6f}\trank(L): {np.linalg.matrix_rank(L)}\tcard(S): {np.count_nonzero(S[~unobserved])}")
        
        if err < tol:
            break

    return L, S
