'''Test harness for the mod_gramm_schmidt module'''

import numpy as np
import matplotlib.image as mpimg
from setup import start_state
from mod_gramm_schmidt import mod_gramm_schmidt
from find_error import find_error
from check_error import check_error

test_image = mpimg.imread('images/DrawingHands.jpg')


def test_d0():
	"""Test the first iteration of mod_gramm_schmidt where d == 0"""
	#Setup parameters
	X,Y,T,d,e_cur, i,used_i, complete = start_state(test_image)
	Er = 0.0000000001

	#Apply gramm_schmidt
	gramm_schmidt_results = mod_gramm_schmidt(X,Y,T,d,e_cur,Er,i,used_i, complete)

	#Update testing parameters
	X,Y,T,d,e_cur,Er,i,used_i,complete = gramm_schmidt_results

   #Test datatypes and values
	assert type(X) == np.ndarray
	assert type(Y) == np.ndarray
	assert type(T) == np.ndarray
	assert type(d) == int

	assert type(Er) == float
	assert type(i) == int
	assert type(used_i) == list
	assert complete == False

	#Test that Y and T have non-zero basis
	assert np.count_nonzero(Y) > 0
	assert np.count_nonzero(T) > 0

	#Test that Y and T have the right shapes.
	assert Y.shape[1] == 1
	assert Y.shape[0] == X.shape[0]
	assert T.shape[1] == X.shape[1]
	assert T.shape[0] == 1

	#Assert that an index has been added to used_i"""
	assert len(used_i) == 1

def test_d1_not_done():
	"""Test mod_gramm_schmidt where d!=0 and there is a need to continue building the basis."""

	#Setup parameters
	X,Y,T,d,e_cur, i,used_i, complete = start_state(test_image)
	Er = 0.0000000001

	#Complete 1 round of the RBD decomposition algorithm
	d1_parameters = check_error(*find_error(*mod_gramm_schmidt(X,Y,T,d,e_cur, Er, i,used_i, complete)))

	#Apply gramm_schmidt at the beginning of stage d = 1
	gramm_schmidt_results = mod_gramm_schmidt(*d1_parameters)

	#Update testing parameters
	X,Y,T,d,e_cur,Er, i,used_i, complete = gramm_schmidt_results

	#Test datatypes and values
	assert type(X) == np.ndarray
	assert type(Y) == np.ndarray
	assert type(T) == np.ndarray
	assert type(d) == int
	assert type(Er) == float
	assert type(used_i) == list
	assert complete == False

	#Assert that an orthogonal projection has been added to Y, and that a vector has been added to T. The dimensions of the dot product of Y and T should equal X.
	assert Y.shape[1] == T.shape[0] == 2
	compression = np.dot(Y, T)
	assert compression.shape == X.shape

	#Definition of orthogonal matrix
	print("Y.T.dot(Y)", Y.T.dot(Y))
	assert np.isclose(np.linalg.det(Y.T.dot(Y).astype(float)), 1)
	#assert np.testing.assert_array_almost_equal(Y.T.dot(Y), np.eye(2,2))


def test_d1_done():
		"""Test mod_gramm_schmidt where d!=0 and the error requirement is met, ending the RBD algorithm."""

		#Setup parameters
		X,Y,T,d,e_cur, i,used_i, complete = start_state(test_image)
		Er = 0.00001

		#Complete 1 round of the RBD decomposition algorithm and update parameters.
		d1_parameters = check_error(*find_error(*mod_gramm_schmidt(X,Y,T,d,e_cur, Er, i,used_i, complete)))
		X,Y,T,d,e_cur, Er, i,used_i, complete = d1_parameters
		#print("d: ",d)
		#print("Y: ", Y)

		#Ensure that the algorithm will think the decomposition is complete.
		Er = 50000

		#Apply gramm_schmidt at the beginning of stage d = 1
		gramm_schmidt_results = mod_gramm_schmidt(X,Y,T,d,e_cur, Er, i,used_i, complete)

		assert gramm_schmidt_results[-1] == True







