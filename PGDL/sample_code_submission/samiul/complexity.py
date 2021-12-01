from collections import Iterable

import numpy as np
import tensorflow as tf

try:
    import cPickle as pickle
except ImportError:
    import pickle

from tensorflow.keras.layers import Layer


def convert_to_noisy_model(model: tf.keras.Model, shape, lambda_0):
    inp = tf.keras.layers.Input(shape)
    x = inp
    for lyr in model.layers:
        x = NoiseLayer(np.exp(-lambda_0))(x)
        x = lyr(x)
        lambda_0 += 1
    model_t = tf.keras.Model(inp, x)
    return model_t


class NoiseLayer(tf.keras.layers.Layer):
    def __init__(self, stdev, **kwargs):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        super(NoiseLayer, self).__init__(**kwargs)
        self.noise_layer = tf.keras.layers.GaussianNoise(stdev)

    def call(self, inputs, **kwargs):
        base = tf.ones_like(inputs)
        noise = self.noise_layer(base, True)
        return inputs * noise

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(NoiseLayer, self).get_config()
        return dict(list(base_config.items()))


def estimate_accuracy(
    model: tf.keras.Model, dataset, iteration: int, should_encode: (int, bool)
):
    acc = 0.0
    for _ in range(iteration):
        x, y = dataset.next()
        if should_encode:
            y = tf.one_hot(y, should_encode)
        ev = model.evaluate(x, y, verbose=0)
        acc += ev[1]
    return acc / iteration


def calculate_moments(grad, nn_out):
    if len(grad.shape) == 3:
        grad = tf.expand_dims(grad, -1)
    axes = tuple(range(len(grad.shape)))[:-1]
    norm_grads = tf.divide(
        grad, tf.reduce_mean(tf.square(grad), axis=-2, keepdims=True)
    )
    weights = tf.reduce_mean(norm_grads, axis=axes, keepdims=True)
    cam = tf.multiply(weights, nn_out)
    cam = tf.maximum(cam, 0)
    cam = cam / tf.reduce_max(cam, axes, keepdims=True)
    cam = tf.where(tf.math.is_nan(cam), tf.zeros_like(cam), cam)
    m = tf.nn.moments(cam, axes=axes)
    m = tf.stack(m, -1)
    return m


def calculate_complexity(model_t, x, y, eps):
    with tf.GradientTape() as tape:
        inputs = tf.cast(x, tf.float32)
        nn_outs, preds = model_t(inputs)  # preds after softmax
        loss = preds * y
    grads = tape.gradient(loss, nn_outs)
    elems = list(zip(grads, nn_outs))
    m = list(map(lambda args: calculate_moments(*args), elems))
    return m


def assert_model_output(model: tf.keras.Model, dataset: tf.data.Dataset) -> bool:
    for x, y in dataset.batch(1):
        yp = model(x)
        if yp.shape != y.shape and yp.shape[:-1] == y.shape:
            return yp.shape[-1]
        return False


def calculate_complexity_v2(model_t, x, y, eps):
    with tf.GradientTape() as tape:
        inputs = tf.cast(x, tf.float32)
        nn_outs, preds = model_t(inputs)  # preds after softmax
        loss = preds * y
    grads = tape.gradient(loss, nn_outs)
    elems = list(zip(grads, nn_outs))
    m = list(calculate_moments(*args) for args in elems)
    return m


def convert_model(model, shape, lambda_0=2.5):
    co = []
    inp = tf.keras.layers.Input(shape)
    x = inp
    for lyr in model.layers:
        x = tf.keras.layers.TimeDistributed(lyr)(x)
        x = NoiseLayer(np.exp(-lambda_0))(x)
        if lyr.trainable:
            co.append(x)
    model_t = tf.keras.Model(inp, [co, x])
    return model_t


def bh_dist(mu, mu_c, v, v_c, eps):
    return tf.maximum(
        0,
        (1 / 8) * ((mu - mu_c) ** 2) / (0.5 * (v + v_c + eps))
        + 0.5 * tf.math.log(eps + 0.5 * (v + v_c) / (eps + v * v_c)),
    )


def aggregate_distribution(i, mu_agg, var_agg, mu, var):
    var_agg = (
        i * var_agg / (i + 1) + var / (i + 1) + i * ((mu - mu_agg) ** 2) / (i + 1) ** 2
    )
    mu_agg = i * mu_agg / (i + 1) + mu / (i + 1)
    return mu_agg, var_agg


def complexity(model, dataset):
    model_t = None
    eps = 1e-5
    should_expand_target = assert_model_output(model, dataset)
    i = 0
    limit = 128
    mu_agg = {}
    mu_c_agg = {}
    var_agg = {}
    var_c_agg = {}
    for x, y in dataset.batch(16):
        if should_expand_target:
            y = tf.one_hot(y, should_expand_target)
        shape = x.shape
        if model_t is None:
            model_t = convert_model(model, shape, 1.5)
        x = tf.expand_dims(x, 0)
        m_l = calculate_complexity_v2(model_t, x, y, eps)
        m_c_l = calculate_complexity_v2(model_t, x, tf.ones_like(y) - y, eps)
        for k, (m, m_c) in enumerate(zip(m_l, m_c_l)):
            if k not in var_agg.keys():
                var_agg[k] = 0
                var_c_agg[k] = 0
                mu_agg[k] = 0
                mu_c_agg[k] = 0
            mu_agg[k], var_agg[k] = aggregate_distribution(
                i, mu_agg[k], var_agg[k], m[:, 0], m[:, 1]
            )
            mu_c_agg[k], var_c_agg[k] = aggregate_distribution(
                i, mu_c_agg[k], var_c_agg[k], m_c[:, 0], m_c[:, 1]
            )
        i += 1
        if i == limit:
            break
    n = []
    for k in var_agg.keys():
        v = var_agg[k]
        v_c = var_c_agg[k]
        mu = mu_agg[k]
        mu_c = mu_c_agg[k]
        dist = tf.reduce_mean(bh_dist(mu, mu_c, v, v_c, eps))
        n.append(dist.numpy())
    c = np.mean(n)
    return c
