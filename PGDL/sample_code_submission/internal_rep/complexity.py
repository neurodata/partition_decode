
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.stats import * 

from tensorflow.keras import backend as K

from keras.utils import multi_gpu_model

from .matrix_funcs import get_matrix_from_poly, compute_complexity


def complexity(model, dataset, program_dir, mid=None, measure = 'KF-kernel'):
	'''
	Wrapper Complexity Function to combine various complexity measures

	Parameters
	----------
	model : tf.keras.Model()
		The Keras model for which the complexity measure is to be computed
	dataset : tf.data.Dataset
		Dataset object from PGDL data loader
	program_dir : str, optional
		The program directory to store and retrieve additional data
	measure : str, optional
		The complexity measure to compute

	Returns
	-------
	float
		complexity measure
	'''

	########## INTERNAL REPRESENTATION ################# 
	# if measure == 'Schatten':
	complexityScore = complexityIR(model, dataset, mid=None, program_dir=program_dir, method=measure)
	# else:
		# complexityScore = complexityIR(model, dataset, program_dir=program_dir)
		
	print('-------Final Scores---------', complexityScore)
	return complexityScore




def complexityIR(model, dataset, method, mid=None, program_dir=None):

	'''
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
	'''


	layers = []	
	batch_size=500
	# poly_m = get_polytope(model, dataset, batch_size=batch_size)
	poly_m = penultimate_activations(model, dataset, batch_size=batch_size)
	# poly_m = polytope_activations(model, dataset, batch_size=batch_size)
	# L_mat = get_matrix_from_poly(model, dataset, poly_m, batch_size=batch_size)
	L_mat = one_hot(poly_m)

	complexity_dict = compute_complexity(L_mat, k=1)

	if method in complexity_dict:
		score = np.array(complexity_dict[method]).squeeze()
		return score
	return -1

def get_polytope(model, dataset, batch_size=500):

	polytope_memberships_list = []

	
	# for batch in batches:
	for x, y in dataset.batch(batch_size):

		batch_ = x
		
		with tf.GradientTape(persistent=True) as tape:
			polytope_memberships = []
			last_activations = batch_
			tape.watch(last_activations)
			for l, layer_ in enumerate(model.layers):
				if l == len(model.layers)-2:
					break

				preactivation = layer_(last_activations)
				if hasattr(layer_, 'activation'):
					binary_preactivation = (K.cast((preactivation > 0), "float"))
					polytope_memberships.append( np.array(binary_preactivation).reshape(len(x), -1))
					last_activations = preactivation * binary_preactivation
				else:
					last_activations = preactivation
		print("*-*-*-*", np.concatenate(polytope_memberships, axis = 1).shape)
		polytope_memberships = [np.tensordot(np.concatenate(polytope_memberships, axis = 1), 2 ** np.arange(0, np.shape(np.concatenate(polytope_memberships, axis = 1))[1]), axes = 1)]
		polytope_memberships_list.append(polytope_memberships[0])
		
		# break
		
	poly_m = np.hstack(polytope_memberships_list)
	return poly_m


def one_hot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot

def polytope_activations(model, dataset, batch_size, pool_layers=True):
	# print("**** hello")
	activations = []
	for x, y in dataset.batch(batch_size):
		n = len(x)
		acts = []
		for l, layer in enumerate(model.layers):
			if l == len(model.layers)-2:
				break
			if hasattr(layer, 'activation'):
				
				if True: #isinstance(layer.activation, tf.keras.activations.relu): #relu
					x = layer(x)
					# x = tf.keras.activations.relu(x)
					act = (K.cast((x > 0), "float"))
					acts.append( np.array(act, dtype=np.int8).reshape( len(x), -1) )

				elif layer.activation == tf.keras.activations.softmax: #softmax
					break
					# x = layer(x)
	                # act = (x.numpy() > 0.5).astype(int)
	                # activations.append(act)
				else: # other actvation
					x = layer(x) 
			elif pool_layers and hasattr(layer, '_name') and 'max_pooling2d' in layer._name:
				act = tf.nn.max_pool_with_argmax(
					x, layer.pool_size, layer.strides, layer.padding.upper()
				).argmax.numpy().reshape(n, -1)
				x = layer(x)
				acts.append(act)
			else: #no activation
				x = layer(x) 
		activations.append(np.concatenate(acts, axis = 1))
	polytope_memberships = [np.tensordot(np.concatenate(activations, axis = 0), 2 ** np.arange(0, np.shape(np.concatenate(activations, axis = 0))[1]), axes = 1)]

	return np.array(polytope_memberships[0])


def penultimate_activations(model, dataset, batch_size=500):
	# penultimate layer model
	penultimate_layer = K.function([model.layers[0].input],
									[model.layers[-4].output])
	activations = []
	binary_str = []

	for x, y in dataset.batch(batch_size):
		out = np.array( penultimate_layer([x])[0] > 0, dtype=np.int8).reshape(len(x), -1) 
		activations.append( out )
	polytope_memberships = [np.tensordot(np.concatenate(activations, axis = 0), 2 ** np.arange(0, np.shape(np.concatenate(activations, axis = 0))[1]), axes = 1)]
	return np.array(polytope_memberships[0])