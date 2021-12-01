import numpy as np
import sys; sys.path.append('../')
from internal_rep.matrix_funcs import \
    get_KF_Schatten_norms, \
    compute_complexity, \
    get_df_tau, \
    evalues_from_regions, \
    get_local_rad_bound
from numpy.testing import assert_almost_equal, assert_equal


def test_evalues_from_regions(L):
    evalues = evalues_from_regions(L)
    evalues_true = [3, 2, 2, 1, 0, 0, 0, 0]
    assert_equal(evalues, evalues_true)
    assert_almost_equal(evalues[:4], np.sort(np.linalg.svd(L, compute_uv=False)**2)[::-1])


def test_get_KF_Schatten_norms(L):
    # Get regular
    metrics = get_KF_Schatten_norms(L, k=3)

    # from evalues
    evalues = evalues_from_regions(L)
    metrics_evalues = get_KF_Schatten_norms(evalues, k=3, from_evalues=True)

    # from gram
    G = L @ L.T
    metrics_gram = get_KF_Schatten_norms(G, k=3, from_gram=True)

    # test all the same
    for metric in zip(metrics, metrics_evalues, metrics_gram):
        assert_almost_equal(metric[0], metric[1])
        assert_almost_equal(metric[1], metric[2])


def test_compute_complexity(L):
    # Get regular
    metrics = compute_complexity(L, k=3)

    # from evalues
    evalues = evalues_from_regions(L)
    metrics_evalues = compute_complexity(evalues, k=3, from_evalues=True)

    # from gram
    G = L @ L.T
    metrics_gram = compute_complexity(G, k=3, from_gram=True)

    # test all the same
    for key in metrics.keys():
        assert_almost_equal(metrics[key], metrics_evalues[key], err_msg=', Failing metric=' + key)
        assert_almost_equal(metrics[key], metrics_gram[key], err_msg=', Failing metric=' + key)


def test_get_local_rad_bound(L):
    # Get regular
    metrics = get_local_rad_bound(L)

    # Get regular, unnormalized
    metrics_unnormalized = get_local_rad_bound(L, normalize=False)

    # from evalues
    evalues = evalues_from_regions(L)
    metrics_evalues = get_local_rad_bound(evalues, from_evalues=True)

    # from gram
    G = L @ L.T
    metrics_gram = get_local_rad_bound(G, from_gram=True)

    # test all the same
    for metric in zip(metrics, metrics_evalues, metrics_gram):
        assert_almost_equal(metric[0], metric[1])
        assert_almost_equal(metric[1], metric[2])


L = np.asarray([
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
])

test_evalues_from_regions(L)
test_get_KF_Schatten_norms(L)
test_get_local_rad_bound(L)
test_compute_complexity(L)
