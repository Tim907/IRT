import numpy as np
import scipy.optimize as so
from numba import jit


def only_keep_k(vec, block_size, k, max_len=None, biggest=True):
    """
    Only keep the k biggest (smallest) elements for each block in a vector.

    If max_len = None, use the whole vec. Otherwise, use vec[:max_len]

    Returns: new vector, indices
    """

    if k == block_size:
        return vec, np.array(list(range(len(vec))))

    do_not_touch = np.array([])
    if max_len is not None:
        do_not_touch = vec[max_len:]
        vec = vec[:max_len]

    # determine the number of blocks
    num_blocks = int(vec.shape[0] / block_size)

    # split the vector in a list of blocks (chunks)
    chunks = np.array_split(vec, num_blocks)

    # chunks_new will contain the k biggest (smallest) elements for each chunk
    chunks_new = []
    keep_indices = []
    for i, cur_chunk in enumerate(chunks):
        if biggest:
            cur_partition_indices = np.argpartition(-cur_chunk, k)
        else:
            cur_partition_indices = np.argpartition(cur_chunk, k)
        chunks_new.append(cur_chunk[cur_partition_indices[:k]])
        keep_indices.extend(cur_partition_indices[:k] + i * block_size)

    if max_len is not None:
        chunks_new.append(do_not_touch)
        keep_indices.extend(
            list(range(vec.shape[0], vec.shape[0] + do_not_touch.shape[0]))
        )

    return np.concatenate(chunks_new), np.array(keep_indices)


@jit(nopython=True)
def calc(v):
    if v < 34:
        "prevent underflow exception"
        if(v < -200): 
            return np.exp(-200)

        return np.log1p(np.exp(v))
    else:
        "function becomes linear"
        return v


calc_vectorized = np.vectorize(calc)


def logistic_likelihood(theta, Z, weights=None, block_size=None, k=None, max_len=None):
    v = -Z.dot(theta)
    if block_size is not None and k is not None:
        v, indices = only_keep_k(v, block_size, k, max_len=max_len, biggest=True)
        if weights is not None:
            weights = weights[indices]
    likelihoods = calc_vectorized(v)
    if weights is not None:
        likelihoods = weights * likelihoods.T
    return np.sum(likelihoods)

def logistic_likelihood_3PL(theta, Z, y, c, opt_beta, weights=None, block_size=None, k=None, max_len=None):
    likelihood = logistic_likelihood(theta=theta[0:2], Z=Z, weights=weights, block_size=block_size, k=k, max_len=max_len)
    v_pos = -Z[y == 1,].dot(theta[0:2])
    if weights is not None:
        if opt_beta is True:
            likelihood = likelihood - np.log(1 - c) * sum(weights[y == -1]) - np.logaddexp(np.log(c), v_pos).dot(weights[y == 1])
        else:
            likelihood = likelihood - (np.log(1 - c[y == -1])).dot(weights[y == -1]) - np.logaddexp(np.log(c[y == 1]), v_pos).dot(weights[y == 1])
            
    else:
        if opt_beta is True:
            likelihood = likelihood - np.log(1 - c) * sum(y == -1) - sum(np.logaddexp(np.log(c), v_pos))
            
        else:
            likelihood = likelihood - sum(np.log(1 - c[y == -1])) - sum(np.logaddexp(np.log(c[y == 1]), v_pos))

    if opt_beta:
        likelihood = likelihood + c_prior_likelihood(c)
    return likelihood
    

def logistic_likelihood_grad(
    theta, Z, weights=None, block_size=None, k=None, max_len=None):

    v = Z.dot(theta)
    if block_size is not None and k is not None:
        v, indices = only_keep_k(v, block_size, k, max_len=max_len, biggest=False)
        if weights is not None:
            weights = weights[indices]
        Z = Z[indices, :]

    grad_weights = 1.0 / (1.0 + np.exp(v))

    if weights is not None:
        grad_weights *= weights

    return -1 * (grad_weights.dot(Z))

def logistic_likelihood_grad_3PL(
    theta, Z, y, c, opt_beta, weights=None, block_size=None, k=None, max_len=None):

    grad = logistic_likelihood_grad(theta=theta[0:2], Z=Z, weights=weights, block_size=block_size, k=k, max_len=max_len)

    v_pos = Z[y == 1,].dot(theta[0:2])
    
    if weights is not None:
        if opt_beta is True:
            grad = grad + (np.multiply(weights[y == 1],(1 / (1 + c * np.exp(v_pos))))).dot(Z[y == 1])
        else:
            grad = grad + (np.multiply(weights[y == 1],(1 / (1 + c[y == 1] * np.exp(v_pos))))).dot(Z[y == 1])
    else:
        if opt_beta is True:
            grad = grad + (1 / (1 + c * np.exp(v_pos))).dot(Z[y == 1])
        else:
            grad = grad + (1 / (1 + c[y == 1] * np.exp(v_pos))).dot(Z[y == 1])

    if opt_beta is True:
        grad_c = 1 / (1 - c) * np.ones(len(y))
        grad_c[y == 1] = -1 / (c + np.exp(-v_pos))
        grad = np.append(grad, np.sum(grad_c))
        grad = grad + c_prior_gradient(c)
    return grad

"""
def logistic_likelihood(theta, y, weights):
    return np.sum(np.log1p(np.exp(-y * np.matmul(theta, weights))))

def logistic_likelihood_grad(theta, y, weights):
    temp = np.outer(1.0 / (1.0 + np.exp(-y * np.matmul(theta, weights))) * np.exp(-y*np.matmul(theta, weights)), np.inner(-y, weights))
    return temp.sum(axis=0)
"""

def optimize_2PL(Z, w=None, block_size=None, k=None, max_len=None, bnds=None, theta_init=None):
    """
    Optimizes a weighted instance of logistic regression.
    """
    if w is None:
        w = np.ones(Z.shape[0])

    def objective_function(theta):
        return logistic_likelihood(theta, Z, w, block_size=block_size, k=k, max_len=max_len)

    def gradient(theta):
        return logistic_likelihood_grad(theta, Z, w, block_size=block_size, k=k, max_len=max_len)

    if theta_init is None:
        theta_init = np.zeros(Z.shape[1])
    
    return so.minimize(objective_function, theta_init, method="L-BFGS-B", jac=gradient, bounds=bnds)


def c_prior_likelihood(c):
    return np.sum((c - 0.1)**2 / (0.08 / 3))
def c_prior_gradient(c):
    return np.sum((c - 0.1) / (0.04 / 3))

def optimize_3PL(Z, y, c, opt_beta, w=None, block_size=None, k=None, max_len=None, bnds=None, theta_init=None):
    """
    Optimizes a weighted instance of logistic regression.
    """
    if w is None:
        w = np.ones(Z.shape[0])

    def objective_function(theta):
        return logistic_likelihood_3PL(theta, Z, y, c, opt_beta, w, block_size=block_size, k=k, max_len=max_len)

    def gradient(theta):
        temp = logistic_likelihood_grad_3PL(theta, Z, y, c, opt_beta, w, block_size=block_size, k=k, max_len=max_len)
        #print(temp)
        return temp

    if theta_init is None:
        theta_init = np.zeros(Z.shape[1])

    temp = so.minimize(objective_function, theta_init, method="L-BFGS-B", jac=gradient, bounds=bnds)
    #temp = so.minimize(objective_function, theta_init, method="Nelder-Mead", bounds=bnds)
    if np.isnan(temp.fun):
        raise ValueError("likelihood is nan")
    return temp

def get_objective_function(y, w):
    return lambda theta: logistic_likelihood(theta, y, weights=w)