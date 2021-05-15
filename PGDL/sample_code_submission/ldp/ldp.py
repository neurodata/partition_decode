import numpy as np
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape


class Quantize(Layer):

    def linspace_init(self, out_shape):
        minval = -2
        maxval = 2
        sep = maxval - minval
        return tf.range(minval, maxval, sep / out_shape, dtype=tf.float32)

    def pairwise_dist_reg(self, weight_matrix):
        shape = 32
        minval = -2
        maxval = 2
        sep = maxval - minval
        t = tf.range(minval, maxval, sep / shape, dtype=tf.float32)
        r = tf.reduce_sum((t - weight_matrix) ** 2)
        return 0.1 * r

    def __init__(self, numCenters, **kwargs):
        #         if 'input_shape' not in kwargs and 'input_dim' in kwargs:
        #             kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Quantize, self).__init__(**kwargs)
        self.numCenters = numCenters

    def build(self, input_shape):
        self.centers = self.add_weight(shape=(self.numCenters),
                                       initializer=self.linspace_init,
                                       name='centers',
                                       regularizer=self.pairwise_dist_reg,
                                       trainable=True)
        super(Quantize, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if type(inputs) == list:
            inp, mask = inputs
            c = self.mask_centers(mask)
        else:
            inp = inputs
            c = tf.ones(self.numCenters)
        inp = tf.expand_dims(inp, -1)
        dist = inp - self.centers
        phi = tf.square(tf.abs(dist))
        qsoft = tf.nn.softmax(-1.0 * phi, -1)
        symbols = tf.argmax(tf.abs(qsoft), -1)
        qsoft = tf.reduce_sum(qsoft * c, -1)
        one_hot_enc = tf.one_hot(symbols, self.numCenters)
        symbols = tf.cast(one_hot_enc, dtype=tf.float32)
        qhard = symbols * c
        qhard = tf.reduce_sum(qhard, axis=-1)
        qbar = qsoft + tf.stop_gradient(qhard - qsoft)
        return qbar

    def compute_output_shape(self, input_shape):
        return [input_shape] * 4

    def mask_centers(self, mask):
        c = self.centers * mask
        zero_vector = tf.zeros(shape=c.shape.as_list()[1], dtype=tf.float32)

        bool_mask = tf.not_equal(c, zero_vector)
        c = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x, bool_mask), name='bool_mask')(c)
        return c

    def get_config(self):
        config = {
            'numCenters': self.numCenters
        }
        base_config = super(Quantize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LearnableNoise(Layer):
    def __init__(self,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(LearnableNoise, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.noise_layer = tf.keras.layers.GaussianNoise(1.0)
        self.supports_masking = True

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        shape = len(input_shape.as_list()) * [1]
        shape[-1] = last_dim
        self.kernel = self.add_weight(shape=shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
        super(LearnableNoise, self).build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        (tf.ones_like(self.kernel))
        noise = self.noise_layer(tf.zeros_like(self.kernel)) * self.kernel
        self.add_loss(-tf.norm(noise))
        return inputs + noise

    def get_config(self):
        config = {
            'kernel': self.kernel
        }
        base_config = super(LearnableNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def convert_model(model: tf.keras.Model, shape):
    inp = tf.keras.layers.Input(shape)
    x = inp
    for lyr in model.layers:
        print(x.shape)
        x = LearnableNoise()(x)
        print(x.shape)

        # lyr.trainable = False
        x = lyr(x)
    model_t = tf.keras.Model(inp, x)
    return model_t


def complexity(model, ds):
    @tf.function
    def optimize(loss_fn, model_t, optimizer, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = loss_fn(y, logits)
        variables = model_t.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

    def estimate_accuracy(model, dataset, iteration):
        acc = 0.0
        for i in range(iteration):
            x, y = dataset.next()
            pred = predict(model, x).numpy()
            acc += np.mean(pred == y)
        return acc / iteration

    @tf.function
    def predict(model, x):
        logits = model(x)
        pred = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        return pred

    batched_ds = iter(ds.shuffle(1000).repeat(-1).batch(64))
    orig_acc = estimate_accuracy(model, batched_ds, 20)
    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    model_t = convert_model(model, batched_ds.next()[0].shape[1:])
    for j in range(20):
        for k in range(20):
            data_pair = batched_ds.next()
            optimize(cce, model_t, optimizer, data_pair)
        estimate_mean = estimate_accuracy(model_t, batched_ds, 5)
        if tf.abs(estimate_mean - orig_acc) < .05:
            break
        weights = model_t.get_weights()[1::2]
        init_weights = model.get_weights()
        norm = sum([np.linalg.norm(w - w_i) ** 2 for (w, w_i) in zip(weights, init_weights)])
        return norm