import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.stats import *

from tensorflow.keras import backend as K

from matrix_funcs import ger_matrix_from_poly, compute_complexity


def complexityIR(model, dataset, program_dir=None, method="Schatten"):

    """
    Function to calculate internal representation based complexity measures

    Parameters
    ----------
    model : tf.keras.Model()
            The Keras model for which the complexity measure is to be computed
    dataset : tf.data.Dataset
            Dataset object from PGDL data loader
    program_dir : str, optional
            The program directory to store and retrieve additional data

    Returns
    -------
    float
            complexity measure
    """

    layers = []
    computeOver = 500
    batchSize = 50
    N = computeOver // batchSize

    poly_m = get_polytope(model, dataset, computeOver=500, batchSize=50)
    # print("********", poly_m.shape, np.unique(poly_m).shape)

    L_mat, gen_err = ger_matrix_from_poly(model, dataset, poly_m)
    complexity_dict = compute_complexity(L_mat, k=1)

    if method in complexity_dict:
        # print("**", complexity_dict[method])
        score = np.array(complexity_dict[method]).squeeze()
        # print(score)
        return score
    return -1


def get_polytope(model, dataset, computeOver=500, batchSize=50):

    layers = []

    it = iter(dataset.repeat(-1).shuffle(5000, seed=1).batch(batchSize))
    N = computeOver // batchSize
    batches = [next(it) for i in range(N)]
    polytope_memberships_list = []

    for batch in batches:

        batch_ = batch[0]

        with tf.GradientTape(persistent=True) as tape:
            intermediateVal = [batch_]
            polytope_memberships = []
            last_activations = batch_
            tape.watch(last_activations)
            for l, layer_ in enumerate(model.layers):
                if l == len(model.layers) - 1:
                    break

                preactivation = layer_(last_activations)
                binary_preactivation = K.cast((preactivation > 0), "float")
                polytope_memberships.append(binary_preactivation)
                last_activations = preactivation * binary_preactivation

        polytope_memberships = [
            np.tensordot(
                np.concatenate(polytope_memberships, axis=1),
                2
                ** np.arange(
                    0, np.shape(np.concatenate(polytope_memberships, axis=1))[1]
                ),
                axes=1,
            )
        ]
        polytope_memberships_list.append(polytope_memberships[0])

    poly_m = np.hstack(polytope_memberships_list)
    return poly_m
