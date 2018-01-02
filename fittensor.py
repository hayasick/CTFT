import numpy as np
import sys
import sktensor as st
from scipy.linalg import eigh

def residual(X, ranks, method='hooi', p=None):
    X = st.dtensor(X)
    if method == 'hooi':
        C, Us = st.tucker_hooi(X, ranks, init='nvecs')
    elif method == 'randomized':
        C, Us = randomized_hooi(X, ranks)
    elif method == 'mach':
        C, Us = mach(X, ranks, p)
    return np.sum((X - C.ttm(Us)) ** 2)
    
def approx_residual(X, ranks, k):
    order = len(X.shape)   # order of X

    n = np.max(X.shape)

    if n <= k:
        return residual(X, ranks)

    inds = []
    for i in range(order):
        _n = X.shape[i]
        s = k
        indn = np.random.choice(_n, s, replace=True)
        inds.append(indn)
    ind_str = ','.join(['inds[%d]' % i for i in range(order)])
    
    ind_ix = eval('np.ix_(%s)' % ind_str)
    miniX = X[ind_ix]

    scale = np.prod(X.shape) / np.prod(miniX.shape)
    return scale * residual(miniX, ranks)


def mach(X, ranks, p):
    """
    Implementation of MACH prposed in
    C. E. Tsourakakis. Mach: Fast randomized tensor decompositions. In ICDM, pages 689–700, 2010.
    """

    prod_ns = np.prod(X.shape)
    indn = np.random.choice(prod_ns, int(prod_ns * p), replace=False)
    multinds = np.unravel_index(indn, X.shape)
    X_sp = st.sptensor(multinds, (1 / p) * X[multinds], shape=X.shape)

    ### for sparse eigen decomposition (scipy linalg problem)
    _ranks = np.array(ranks)
    _shape = np.array(X.shape)
    _ind = _ranks >= _shape
    _ranks[_ind] = _shape[_ind] - 1
    return st.tucker_hooi(X_sp, _ranks.tolist(), init='nvecs')
    

def randomized_hooi(X, ranks):
    """
    Implementation of RandTucker2i prposed in
    G. Zhou, A. Cichocki, and S. Xie. Decomposition of big tensors with low multilinear rank. arXiv preprint
287 arXiv:1412.1885, 2014.
    """

    def randomized_nvecs(X, Omega, n, rank):
        Xn = X.unfold(n).dot(Omega)
        Y = Xn.dot(Xn.T)
        N = Y.shape[0]
        _, U = eigh(Y, eigvals=(N - rank, N - 1))
        U = np.array(U[:, ::-1])
        return U
    
    U = []
    Omega = []
    for i in range(len(ranks)):
        U.append(np.random.randn(X.shape[i], ranks[i]))
        Omega.append(np.random.randn(int(np.prod(ranks) / ranks[i]), ranks[i]))

    for _ in range(2):
        for n in range(len(ranks)):
            Utilde = X.ttm(U, n, transp=True, without=True)
            U[n] = randomized_nvecs(Utilde, Omega[n], n, ranks[n])

    core = Utilde.ttm(U, n, transp=True)
    return core, U

    

if __name__ == "__main__":

    def gen_lowrank_X(ns, ranks, sigma=0):
        C = np.random.randn(np.prod(ranks)).reshape(ranks)
        Us = list()
        for i in range(len(ns)):
            (n, r) = (ns[i], ranks[i])
            Us.append(np.random.randn(n * r).reshape((n, r)))

        A = st.ttm(st.dtensor(C), Us)
        A /= np.sqrt(np.mean(A ** 2))
        A += np.random.randn(np.prod(ns)).reshape(ns) * sigma
        return A
    
    np.random.seed(1)
    [n, rank, k] = [200, 5, 10]
    order = 3

    ranks = [rank] * order
    ns = [n] * order
    X = gen_lowrank_X(ns, ranks, 0.01)
    
    print('hooi:    \t', residual(X, ranks))
    print('sampling:\t', approx_residual(X, ranks, k))
    print('randomized:\t', residual(X, ranks, method='randomized'))
