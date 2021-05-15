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

def ger_matrix_from_poly(model, dataset, poly_m):
    # L_matrices = {'0/1': [], 'true_label':[], 'est_label':[], 'est_poster':[]}
    L_matrices = []
    test_y, pred_y, test_acc = get_label_pred(model, dataset)

    unique_poly = np.unique(poly_m)
    n_poly = len(unique_poly)

    # for key in L_matrices:
    L_mat = np.zeros((len(poly_m), n_poly))
    for idx, poly_i in enumerate(poly_m):
        poly_idx = np.where(unique_poly==poly_i)
        L_mat[idx, poly_idx] = pred_y[idx]+1
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
    test_gen_err = (1-test_acc)
    return np.array(L_mat), test_gen_err


def get_label_pred(model, dataset, computeOver=500, batchSize=50):

    
    model.compile(optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'])
            
    acc, size = 0, 0
    test_y = []
    
    # for batch in batches:
    preds = []
    for x, y in dataset.batch(500):
        test_loss, test_acc = model.evaluate(x, y, verbose=False)
        acc += test_acc * len(y)
        size += len(y)
        preds.extend(model.predict(x))
        test_y.extend(y)
    acc = acc / size
    pred_y = np.argmax(preds, axis=-1)
    # print(pred_y.shape)

    return test_y, pred_y, acc

##********** Matrix ranks *************##
def get_stable_rank(m):
  '''
  Compute stable rank of a matrix: frobenius norm (squared) / spectral norm (squared)
  '''
  return norm(m, ord='fro')**2 /norm(m, ord=2)**2 

def get_KF_Schatten_norms(m, num_k=5):
  '''
  Compute different matrix norms
  Input: m - 2d matrix (n by L)
  Return: 4 1D numpy arrays
  - First: Ky-Fan results [Un-normalized], where k-th element is the sum of top-k singular values
  - Second: Ky-Fan results [Normalized], where k-th element is the ratio of the variance explained by top-k singular values
  - Third: Ky-Fan results on m^T @ m, where k-th element is the sum of top-k eigenvalues of m^T @ m (i.e., singular values of (m) squared)
  - Fourth: Schatten results, where k-th element is the k-norm, (sum_i sigma^k)^{1/k}
  '''
  ss = np.linalg.svd(m, full_matrices=False, compute_uv=False)
  KFs_raw = np.array( [ss[:i+1].sum() for i in range(num_k)])
  total = ss.sum()
  KFs = KFs_raw / total
  evalues = ss**2
  KFs_kernel = np.array( [evalues[:i+1].sum() for i in range(num_k)])
  Schattens = [total]
  ss_pow = copy.deepcopy(ss)
  for i in range(2, num_k+1):
    ss_pow = np.multiply(ss_pow, ss)
    Schattens.append( np.power(ss_pow.sum(), 1/i))
  return KFs_raw, KFs, KFs_kernel, np.array(Schattens)


def graph_metrics(m):
  '''
  Input: internal representation, n by L
  Return: 2-tuple
  - clustering coefficients of a bipartite graph built from m, a measure of local density of the connectivity
  ref: https://networkx.org/documentation/stable//reference/algorithms/generated/networkx.algorithms.bipartite.cluster.clustering.html#networkx.algorithms.bipartite.cluster.clustering
  - modularity: relative density of edges inside communities with respect to edges outside communities.
  ref: https://python-louvain.readthedocs.io/en/latest/api.html#community.modularity
  '''
  sM = scipy.sparse.csr_matrix(m)
  G = bipartite.matrix.from_biadjacency_matrix(sM)
  avg_c = bipartite.average_clustering(G, mode="dot")
  partition = community_louvain.best_partition(G)
  modularity = community.modularity(partition, G)

  return avg_c, modularity


def compute_complexity(X, k=5):
    '''
    Input: internal rep matrices
    compute different notions of complexity measures for the internal representation matrices
    '''

    plot_dict = {'ranks': [], 'stable_ranks': [], 'modularity': [], 'avg_clusters': [], 'KF-raw': [], 'KF-ratio': [], 'KF-kernel':[], 'Schatten': []}

#   for i in range(len(X)):
    rep = X
    plot_dict['ranks'].append(matrix_rank(rep))
    plot_dict['stable_ranks'].append(get_stable_rank(rep))
    avg_c, modularity = graph_metrics(rep)
    plot_dict['avg_clusters'].append(avg_c)
    plot_dict['modularity'].append(modularity)
    KF_norms, KF_ratios, KF_kers, Schattens = get_KF_Schatten_norms(rep, num_k=k)
    plot_dict['KF-raw'].append(KF_norms)
    plot_dict['KF-ratio'].append(KF_ratios)
    plot_dict['KF-kernel'].append(KF_kers)
    plot_dict['Schatten'].append(Schattens)
  
    return plot_dict


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



def get_df_tau(plot_dict, gen_err):
  '''
  Return a dataframe of the kendall tau's coefficient for different methods
  '''
  #tau, p_value = compute_tau(result_dict[err], plot_dict['avg_clusters'], inverse=True)
  #taus, pvalues, names, inverses = [tau], [p_value], ['cc'], ['True']
  taus, pvalues, names, inverses = [], [], [], []
  for key, value in plot_dict.items():
    value = np.array(value)
    # if key in ['ranks', 'stable_ranks', 'avg_clusters', 'modularity']:
    #   continue
    for i in range(value.shape[1]):
      if key == 'Schatten':
        if i == 0: #Schatten 1-norm, no inversion
          inverse_flag = False
        elif i == 1:
          continue #skip trivial 2-norm
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