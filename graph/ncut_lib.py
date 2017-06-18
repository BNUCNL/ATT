#! /usr/bin/env python
# coding=utf-8

import numpy as np
from scipy import sparse, rand, linalg
from scipy.sparse.linalg import eigsh

def generate_weight_matrix(N):
    """
    A function to randomized generate randomized weight matrix

    W: nonnegative and symmetric matrix, N*N
    """
    m = np.random.rand(N,N)
    m = np.triu(m)
    m += m.T - 2*np.diag(m.diagonal())
    return m    


class SVDError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def ncut(w, neig_values):
    """
    Normalized cut spectral clustering
    w is simiarity matrix
    The number of eigenvectors corresponds to the maximum number of classes

    Parameters:
    ----------
    w: symmetric similarity matrix
    neig_values: number of eigenvector that should be calculated

    Return:
    -------
    eigen_val: eigenvalues from the eigen decomposition of the LaPlacian of W 
    eigen_vec: eigenvectors from the eign decomposition of the LaPlacian of W
    """
    offset = 0.5
    maxiters = 100
    eigsErrorTolerence = 1e-6
    eps = 2.2204e-16

    m = w.shape[1]

    d = np.abs(w).sum(0)
    dr = 0.5*(d-w.sum(0))
    d = d+offset*2
    dr = dr+offset

    # the normalized laplacian
    w = w+sparse.spdiags(dr, [0], m, m, "csc")
    d_invsqrt = sparse.spdiags((1.0/np.sqrt(d+eps)), [0], m, m, "csc")
    p = d_invsqrt*(w*d_invsqrt)

    # the eigen decomposition
    eigen_val, eigen_vec = eigsh(p, neig_values, maxiter = maxiters, tol = 1e-6, which='LA')
    
    i = np.argsort(-eigen_val)
    eigen_val = eigen_val[i]
    eigen_vec = eigen_vec[:,i]

    # normalize the returned eigenvectors
    eigen_vec = d_invsqrt*np.matrix(eigen_vec)
    norm_ones = linalg.norm(np.ones((m,1)))

    for i in range(0, eigen_vec.shape[1]):
        eigen_vec[:,i] = (eigen_vec[:,i]/linalg.norm(eigen_vec[:,i]))*norm_ones
        if eigen_vec[0,i] != 0:
            eigen_vec[:,i] = -1*eigen_vec[:,i]*np.sign(eigen_vec[0,i])
    
    return eigen_val, eigen_vec

def discretisation(eigen_vec):
    """
    Perform the second step of normalized cut clustering which assigns feature to clusters based on the eigen vectors from the LaPlacian of a similarity matrix.
    
    Parameters:
    ---------
    eigen_vec: Eigenvectors of the normalized LaPlacian calculated from the similarity matrix for the corresponding clustering problem
    
    Return:
    ---------
    eigen_vec_discrete: discretised eigenvectors
                        vectors of 0 and 1 which indicate whether or not a feature belongs to the cluster defined by the eigen vector
                        i.e. a one in the 10th row of the 4th eigenvector(column) means that feature 10 belongs to cluster #4
    """
    eps = 2.2204e-16
    n, k = eigen_vec.shape
    vm = np.kron(np.ones((1,k)), np.sqrt(np.multiply(eigen_vec, eigen_vec).sum(1)))
    eigen_vec = eigen_vec/vm
    
    svd_restarts = 0
    exitLoop = 0

    while (svd_restarts < 30) and (exitLoop == 0):
        c = np.zeros((n,1))
        R = np.matrix(np.zeros((k,k)))
        R[:,0] = eigen_vec[int(rand(1)*(n-1)),:].transpose()
    
        for j in range(1,k):
            c = c + np.abs(eigen_vec*R[:,j-1])
            R[:,j] = eigen_vec[c.argmin(),:].transpose()
    
        last_objvalue = 0
        n_iter_discrete = 0
        n_iter_discrete_max = 20

        while exitLoop == 0:
            print('iteration {}'.format(n_iter_discrete))
            n_iter_discrete += 1
            t_discrete = eigen_vec*R
        
            j = np.reshape(np.asarray(t_discrete.argmax(1)), n)
            eigenvec_discrete = sparse.csc_matrix((np.ones(len(j)),(range(0,n),\
            np.array(j))), shape=(n,k))
            tSVD = eigenvec_discrete.transpose()*eigen_vec
            try:
                U, S, Vh = linalg.svd(tSVD)            
            except LinAlgError:
                print('SVD didn''t converged, randomizing and trying again')
                break
        # test for convergence
        NcutValue = 2*(n-S.sum())
        if((np.abs(NcutValue - lastObjectValue)<eps) or (nbIterationsDiscretisation > nbIteraionsDiscretisationMax)):
            exitLoop = 1
        else:
            lastObjectiveValue = NcutValue
            R = np.matrix(Vh).transpose()*np.matrix(U).transpose()
    
    if exitLoop == 0:
        raise SVDError("SVD didn't converge after 30 retries")
    else:
        return eigenvec_discrete










