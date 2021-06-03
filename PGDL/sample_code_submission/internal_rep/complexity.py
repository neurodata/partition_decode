
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.stats import * 

from tensorflow.keras import backend as K

from keras.utils import multi_gpu_model


# from complexity_ir import complexityIR
from .matrix_funcs import get_matrix_from_poly, compute_complexity


def complexity(model, dataset, program_dir, mid, measure = 'h*'):
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
	complexityScore = complexityIR(model, dataset, mid, program_dir=program_dir, method=measure)
	# else:
		# complexityScore = complexityIR(model, dataset, program_dir=program_dir)
		
	print('-------Final Scores---------', complexityScore)
	return complexityScore




def complexityIR(model, dataset, mid, program_dir=None, method="KF-raw"):

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

	# try:
   	# 	model = multi_gpu_model(model)
	# except:
	# 	pass

	layers = []	
	batch_size=50
	poly_m = get_polytope(model, dataset, batch_size=batch_size)
	# poly_m = binary_pattern_mat(model, dataset, batch_size=batch_size)
	# poly_m = polytope_activations(model, dataset)
	print("********", poly_m.shape, np.unique(poly_m).shape) 
	# L_mat = get_matrix_from_poly(model, dataset, poly_m, batch_size=batch_size)
	L_mat = one_hot(poly_m)
	print("********", L_mat.shape) 
	# evalues = np.linalg.svd(m, full_matrices=False, compute_uv=False)
	# np.save('%s.npy'%mid, evalues)
	complexity_dict = compute_complexity(L_mat, k=1)

	if method in complexity_dict:
		score = np.array(complexity_dict[method]).squeeze()
		return score
	return -1


def get_polytope(model, dataset, batch_size=500):
	# print("**** hello from get_polytope")
	layers = []

	polytope_memberships_list = []

	
	# for batch in batches:
	for x, y in dataset.batch(batch_size):

		batch_ = x
		
		with tf.GradientTape(persistent=True) as tape:
			intermediateVal = [batch_]
			polytope_memberships = []
			last_activations = batch_
			tape.watch(last_activations)
			for l, layer_ in enumerate(model.layers):
				if l == len(model.layers)-1:
					break

				preactivation = layer_(last_activations)
				binary_preactivation = (K.cast((preactivation > 0), "float"))
				polytope_memberships.append( np.array(binary_preactivation).reshape(len(x), -1))
				last_activations = preactivation * binary_preactivation
			
		polytope_memberships = [np.tensordot(np.concatenate(polytope_memberships, axis = 1), 2 ** np.arange(0, np.shape(np.concatenate(polytope_memberships, axis = 1))[1]), axes = 1)]
		polytope_memberships_list.append(polytope_memberships[0])
		
		# break
		
	poly_m = np.hstack(polytope_memberships_list)
	return poly_m


def binary_pattern_mat(model, dataset, batch_size=500):
	layers = []

	polytope_memberships_list = []
	binary_str = []
	for x, y in dataset.batch(batch_size):

		batch_ = x
		
		with tf.GradientTape(persistent=True) as tape:
			intermediateVal = [batch_]
			polytope_memberships = []
			last_activations = batch_
			tape.watch(last_activations)
			for l, layer_ in enumerate(model.layers):
				if l == len(model.layers)-1:
					break
				preactivation = layer_(last_activations)
				binary_preactivation = (K.cast((preactivation > 0), "float"))
				last_activations = preactivation * binary_preactivation
			#np.unique(binary_preactivation, axis=0, return_inverse=True)
			
			for idx, pattern in enumerate(np.array(binary_preactivation).reshape(len(x), -1)):
				binary_str.append( ''.join(str(int(x)) for x in pattern) )
		# break
	return np.array(binary_str)

# def binary_pattern_mat(model, dataset, batch_size=500):
#   layers = []

#   polytope_memberships_list, polytope_penult_id = [], []

#   for x, y in dataset.batch(batch_size):

#     batch_ = x
    
#     with tf.GradientTape(persistent=True) as tape:
#       intermediateVal = [batch_]
#       polytope_memberships = []
#       last_activations = batch_
#       tape.watch(last_activations)
#       for l, layer_ in enumerate(model.layers):
#         if l == len(model.layers)-2:
#           break
#         preactivation = layer_(last_activations)
#         binary_preactivation = (K.cast((preactivation > 0), "float"))
#         last_activations = preactivation * binary_preactivation
#       #np.unique(binary_preactivation, axis=0, return_inverse=True)
#       #print(binary_preactivation.shape)
#       for idx, pattern in enumerate(np.array(binary_preactivation).reshape(len(x), -1)):
#         polytope_penult_id.append( ''.join(str(int(x)) for x in pattern))
#       break
#   return np.array(polytope_penult_id)

def one_hot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot

def polytope_activations(model, dataset, pool_layers=True):
	print("**** hello")
	activations = []
	for x, y in dataset.batch(16):
		n = len(x)
		
		for layer in model.layers:
			if hasattr(layer, 'activation'):
				if layer.activation == tf.keras.activations.relu:
					x = layer(x)
					act = (x.numpy() > 0).astype(int).reshape(n, -1)
					activations.append(act)
				elif layer.activation == tf.keras.activations.softmax:
					return activations
	#                 act = (x.numpy() > 0.5).astype(int)
	#                 activations.append(act)
				else:
					x = layer(x)
			elif pool_layers and hasattr(layer, '_name') and 'max_pooling2d' in layer._name:
				act = tf.nn.max_pool_with_argmax(
					x, layer.pool_size, layer.strides, layer.padding.upper()
				).argmax.numpy().reshape(n, -1)
				x = layer(x)
				activations.append(act)
			else:
				x = layer(x)
	return np.unique(np.hstack(activations), axis=0, return_inverse=True)[1]
