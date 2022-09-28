import numpy as np
import scipy.optimize as so
from numba import jit


def only_keep_k(vec, block_size, k, max_len=None, biggest=True):
    """
    Only keep the k biggest (smalles) elements for each block in a vector.

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


def logistic_likelihood(theta, y, weights):
    return np.sum(np.log1p(np.exp(-y * np.matmul(theta, weights))))

def logistic_likelihood_grad(theta, y, weights):
    temp = np.outer(1.0 / (1.0 + np.exp(-y * np.matmul(theta, weights))) * np.exp(-y*np.matmul(theta, weights)), np.inner(-y, weights))
    return temp.sum(axis=0)

def optimize(y, w):
    """
    Optimizes a weighted instance of logistic regression.
    """

    def objective_function(theta):
        return logistic_likelihood(theta, y=y, weights=w)

    def gradient(theta):
        return logistic_likelihood_grad(theta, y=y, weights=w)

    theta0 = np.zeros(w.shape[0])

    return so.minimize(objective_function, theta0, method="L-BFGS-B", jac=gradient)

def get_objective_function(y, w):
    return lambda theta: logistic_likelihood(theta, y, weights=w)