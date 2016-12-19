import numpy as np
import matplotlib.image as mpimg
from setup import start_state
from mod_gramm_schmidt import mod_gramm_schmidt
from find_error import find_error
from check_error import check_error
from rbd import rbd

test_image = mpimg.imread('images/DrawingHands.jpg')

def test_rbd_to_completion():
    """Test case that runs RBD with no cap on dmax and an error requirement that cannot be satisfied."""
    
    #Define dmax and Er 
    dmax = test_image.shape[1]
    Er = -np.inf

    #Run RBD and capture the output
    X,Y,T,d,e_cur,Er,i,used_i,complete = rbd(test_image, dmax, Er)

    #Test that the compression YT has the same dimensions as the input matrix X
    compression = np.dot(Y,T)
    assert compression.shape == X.shape
    assert len(used_i) == d

    #Test that Y is orthogonal
    assert np.diagonal(Y.T.dot(Y)).all() == 1

def test_rbd_early_break_d():
    """Test case that affirms that RBD quits at a user-defined dmax"""
    #Define dmax and Er 
    dmax = 5
    Er = -np.inf

    #Run RBD and capture the output
    X,Y,T,d,e_cur,Er,i,used_i,complete = rbd(test_image, dmax, Er)

    #Test that the compression YT has the same dimensions as the input matrix X
    compression = np.dot(Y,T)
    assert compression.shape == X.shape

    #Assert that RBD quit at the proper stage
    assert d == 6
    assert len(used_i) == d == dmax+1

    #Test that Y is orthogonal
    assert np.diagonal(Y.T.dot(Y)).all() == 1

def test_rbd_early_break_e_cur():
    """Test case that sets the error threshold Er that will be satisfied by RBD. Affirms that RBD quits at that point."""

    #Define dmax and Er 
    dmax = test_image.shape[1]
    Er = 255

    #Run RBD and capture the output
    X,Y,T,d,e_cur,Er,i,used_i,complete = rbd(test_image, dmax, Er)

    #Test that the compression YT has the same dimensions as the input matrix X
    compression = np.dot(Y,T)
    assert compression.shape == X.shape

    #Assert that RBD quit at the proper stage
    assert len(used_i) == d+1
    assert e_cur <= Er

    #Test that Y is orthogonal
    assert np.diagonal(Y.T.dot(Y)).all() == 1




