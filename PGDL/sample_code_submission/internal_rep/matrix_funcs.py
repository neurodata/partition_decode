import numpy as np
import scipy
from scipy import stats

from networkx.algorithms import bipartite
import scipy.linalg as la
from numpy.linalg import matrix_rank, norm
import community
from community import community_louvain

import pandas as pd

import copy


def get_matrix_from_poly(model, dataset, poly_m, batch_size=500):
    # L_matrices = {'0/1': [], 'true_label':[], 'est_label':[], 'est_poster':[]}
    L_matrices = []
    # test_y, pred_y, test_acc = get_label_pred(model, dataset, batch_size=batch_size)
    # pred_y = get_label_pred(model, dataset, batch_size=batch_size)

    unique_poly = np.unique(poly_m)
    n_poly = len(unique_poly)

    # for key in L_matrices:
    L_mat = np.zeros((len(poly_m), n_poly))
    for idx, poly_i in enumerate(poly_m):
        poly_idx = np.where(unique_poly == poly_i)
        L_mat[idx, poly_idx] = 1  # pred_y[idx]+1
            # if key == '0/1':
            #     L_mat[idx, poly_idx] = pred_label[idx]
            # elif key == 'true_label':
            #     L_mat[idx, poly_idx] = 2*y_train[idx]-1
            # elif key == 'est_label':
            #     L_mat[idx, poly_idx] = 2*pred_label[idx]-1
            # elif key == 'est_poster':
            #     L_mat[idx, poly_idx] = 2*pred_poster[idx]-1
        # L_matrices[key].append(L_mat)

    # gen_gap = abs((1-test_acc) - (1-train_acc))
    # test_gen_err = (1-test_acc)
    return np.array(L_mat)  # , test_gen_err


def get_label_pred(model, dataset, batch_size=500):

    # model.compile(optimizer='adam',
		# 	loss='sparse_categorical_crossentropy',
		# 	metrics=['accuracy'])

    acc, size = 0, 0
    test_y = []

    # for batch in batches:
    preds = []
    for x, y in dataset.batch(batch_size):
        # test_loss, test_acc = model.evaluate(x, y, verbose=False)
        # acc += test_acc * len(y)
        # size += len(y)
        preds.extend(model.predict(x))
        # test_y.extend(y)

        break

    # acc = acc / size
    pred_y = np.argmax(preds, axis=-1)
    # print(pred_y.shape)

    return pred_y  # test_y, pred_y, acc

##********** Matrix ranks *************##


def get_stable_rank(m):
    '''
    Compute stable rank of a matrix: frobenius norm (squared) / spectral norm (squared)
    '''
    return norm(m, ord='fro')**2 / norm(m, ord=2)**2


def get_KF_Schatten_norms(L, k=5, from_evalues=False, from_gram=False):
    """
    Compute different matrix norms

    Parameters
    ----------
    L : numpy.ndarray, shape (n_samples, ...)
        internal representation matrix or precomputed eigenvalues

    k : int, default=5
        number of eigenvalues for KF and Schatten methods

    from_evalues : boolean, default=False
        If True, then L is assumed to be the precomputed eigenvalues

    from_gram : boolean, default=False
        If True, then L is assumed to be a square kernel (Gram) matrix.
        Otherwise an svd will be performed on L where the Gram matrix is LL^T
        which improves computational efficiency.

    Returns
    -------
    tuple, length (4,) of the following metrics
        - Ky-Fan results [Un-normalized], where k-th element is the sum of top-k singular values
        - Ky-Fan results [Normalized], where k-th element is the ratio of the variance explained by top-k singular values
        - Ky-Fan results on m^T @ m, where k-th element is the sum of top-k eigenvalues of m^T @ m (i.e., singular values of (m) squared)
        - Schatten results, where k-th element is the k-norm, (sum_i sigma^k)^{1/k}
    """
    if from_evalues:
        evalues = L
        ss = evalues**(1/2)
    elif from_gram:
        evalues = np.linalg.svd(L, compute_uv=False, hermitian=True)
        ss = evalues**(1/2)
    else:
        ss = np.linalg.svd(L, compute_uv=False)
        evalues = np.zeros(L.shape[0])
        evalues[:len(ss)] = ss**2

    KFs_raw = np.array([ss[:i+1].sum() for i in range(k)])
    total = ss.sum()
    KFs = KFs_raw / total
    KFs_kernel = np.array([evalues[:i+1].sum() for i in range(k)])
    Schattens = [total]
    ss_pow = copy.deepcopy(ss)
    for i in range(2, k+1):
        ss_pow = np.multiply(ss_pow, ss)
        Schattens.append(np.power(ss_pow.sum(), 1/i))
    return KFs_raw, KFs, KFs_kernel, np.array(Schattens)


def graph_metrics(m):
    '''
    Input: internal representation, n by L
    Return: 2-tuple
    - clustering coefficients of a bipartite graph built from m, a measure of local density of the connectivity
    # networkx.algorithms.bipartite.cluster.clustering
    ref: https://networkx.org/documentation/stable//reference/algorithms/generated/networkx.algorithms.bipartite.cluster.clustering.html
    - modularity: relative density of edges inside communities with respect to edges outside communities.
    ref: https://python-louvain.readthedocs.io/en/latest/api.html#community.modularity
    '''
    sM = scipy.sparse.csr_matrix(m)
    G = bipartite.matrix.from_biadjacency_matrix(sM)
    avg_c = bipartite.average_clustering(G, mode="dot")
    partition = community_louvain.best_partition(G)
    modularity = community.modularity(partition, G)

    return avg_c, modularity


def compute_complexity(L, k=5, from_evalues=False, from_gram=False):
    """
    Computes a variety of internal representation complexity metrics at once.

    Parameters
    ----------
    L : numpy.ndarray, shape (n_samples, ...)
        internal representation matrix or precomputed eigenvalues

    k : int, default=5
        number of eigenvalues for KF and Schatten methods

    from_evalues : boolean, default=False
        If True, then L is assumed to be the precomputed eigenvalues

    from_gram : boolean, default=False
        If True, then L is assumed to be a square kernel (Gram) matrix.
        Otherwise an svd will be performed on L where the Gram matrix is LL^T
        which improves computational efficiency.

    Returns
    -------
    complexity_dict : dict
        dictionary of (metric_name, metric_value) pairs for L
    """

    complexity_dict = {}

    # For efficiency across the multiple metrics
    if from_evalues:
        evalues = L
    elif from_gram:
        evalues = np.linalg.svd(L, compute_uv=False, hermitian=True)
    else:
        ss = np.linalg.svd(L, compute_uv=False)
        evalues = np.zeros(L.shape[0])
        evalues[:len(ss)] = ss**2

    KF_norms, KF_ratios, KF_kers, Schattens = get_KF_Schatten_norms(evalues, k, from_evalues=True)
    complexity_dict['KF-raw'] = KF_norms
    complexity_dict['KF-ratio'] = KF_ratios
    complexity_dict['KF-kernel'] = KF_kers
    complexity_dict['Schatten'] = Schattens

    h_star, h_argmin = get_local_rad_bound(evalues, normalize=True, from_evalues=True)
    complexity_dict['h*'] = h_star
    complexity_dict['h_argmin'] = h_argmin

    return complexity_dict


def compute_tau(gen_gap, metric, inverse=False):
    '''
    Input: array (generalization gap); array (metric computed for the model instance at such a generalization gap);
    - If inverse: first take inverse of the metric, and compute the kendall tau coefficient
    Return: kendall's tau coefficient, pvalue
    '''
    if inverse:
        metric = np.array([1/elem for elem in metric])
    tau, p_value = stats.kendalltau(gen_gap, metric)
    return tau, p_value


def get_df_tau(complexity_dict, gen_err):
    '''
    Return a dataframe of the kendall tau's coefficient for different methods
    '''
    # tau, p_value = compute_tau(result_dict[err], complexity_dict['avg_clusters'], inverse=True)
    # taus, pvalues, names, inverses = [tau], [p_value], ['cc'], ['True']
    taus, pvalues, names, inverses = [], [], [], []
    for key, value in complexity_dict.items():
        value = np.array(value)
        # if key in ['ranks', 'stable_ranks', 'avg_clusters', 'modularity']:
        #   continue
        for i in range(value.shape[1]):
            if key == 'Schatten':
                if i == 0:  # Schatten 1-norm, no inversion
                    inverse_flag = False
                elif i == 1:
                    continue  # skip trivial 2-norm
                else:
                    inverse_flag = True
            else:
                inverse_flag = True
            tau, p_value = compute_tau(gen_err, value[:,i], inverse=inverse_flag)
            taus.append(tau)
            pvalues.append(p_value)
            names.append(key + '_' +str(i+1))
            inverses.append(inverse_flag)

    kendal_cor = pd.DataFrame(
        {'metric': names,
        'kendall_tau': taus,
        'pvalue': pvalues,
        'inverse': inverses
    })

    return kendal_cor


# - ----------- New 
def compute_rad_gen_gap(m, normalize=False):
    '''
    Compute complexity measures of local rad bound for different model architecture, and generalization gap 
    Input: 
    - listOfResults: different trials; 
    - matrix: 'pen_matrices' or 'matrices';
    - mode: 'depth' or 'width' or 'merge'
    Output: rs (depth/width/depth+width)
    '''
    listOfMatrices = [ m ]
    avg_proximity = random_partition_kernel(listOfMatrices)
    r, h = get_local_rad_bound(avg_proximity, normalize)
    return r, h


def random_partition_kernel(listOfMs):
    '''
    Compute the random partition kernel (aka characteristic kernel)
    Input:  list of internal matrices from same model architecture but different trials
    Output: average proximity matrix  I_{phi(x_i)=phi(x_j)} := m @ m.T , where phi is the activated region (part) assigned to x, average over trials
    '''
    listOfInds = [m @ m.T for m in listOfMs]
    return np.mean(np.array(listOfInds), axis=0)


def evalues_from_regions(L):
    """
    Efficiently computes eigenvalues of the representation matrix/Kernel (Gram)
    matrix when it comes from assignments of samples to regions.

    Parameters
    ----------
    L : list or numpy.ndarray, length (n_samples)
        The identification of which sample occurs in which region. Can be
        a list of internal representation matrices or raw bitstring ids.

    Returns
    -------
    evalues : numpy.ndarray, shape (n_samples,)
        Eigenvalues
    """
    evalues = np.zeros(len(L))
    _, region_sizes = np.unique(L, return_counts=True, axis=0)
    region_sizes = np.sort(region_sizes)[::-1] # sort in decreasing order
    evalues[:len(region_sizes)] = region_sizes
    return evalues


def get_local_rad_bound(L, normalize=True, from_evalues=False, from_gram=False):
    '''
    Compute local rad bound from Bartlett using avg proximity matrix (i.e. characteristic kernel of NN)

    Parameters
    ----------
    L : numpy.ndarray, shape (n_samples, ...)
        Internal representation matrix, Gram matrix, or eigenvalues

    normalize : boolean, default=True
        If True, scales the eigenvalues by 1/n_samples as is done in Bartlett

    from_evalues : boolean, default=False
        If True, then L is assumed to be the precomputed eigenvalues

    from_gram : boolean, default=False
        If True, then L is assumed to be a square kernel (Gram) matrix.
        Otherwise an svd will be performed on L where the Gram matrix is LL^T
        which improves computational efficiency.

    Returns
    -------
    h_star, h_argmin : floats
        r^* upper bound, index of the upper bound h

    Notes
    -----
    [Bug: normalize should be applied with a hyper-parameter (constant), as the functions are not strictly contained in the unit-ball]
    '''
    n = L.shape[0]
    if from_evalues:
        evalues = L
    elif from_gram:
        evalues = np.linalg.svd(L, compute_uv=False, hermitian=True)
    else:
        ss = np.linalg.svd(L, compute_uv=False)
        evalues = np.zeros(n)
        evalues[:len(ss)] = ss**2

    if normalize:
        evalues = evalues / n
    num_nonzero = np.count_nonzero(evalues)
    rs = [h/n + np.sqrt((1/n) * evalues[h:].sum()) for h in range(0,num_nonzero+1)]

    return np.array(rs).min(), np.array(rs).argmin()
