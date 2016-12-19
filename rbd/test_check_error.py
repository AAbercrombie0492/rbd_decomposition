import numpy as np
import matplotlib.image as mpimg
from setup import start_state
from mod_gramm_schmidt import mod_gramm_schmidt
from find_error import find_error
from check_error import check_error

test_image = mpimg.imread('images/DrawingHands.jpg')

def test_decomposition_complete():
    """Test case for when e_cur <= Er, signaling that the decomposition is done."""
    
    #Setup parameters
    X,Y,T,d,e_cur, i,used_i, complete = start_state(test_image)
    Er = np.inf

    #Capture results of RBD at the point before check_error is run.
    params_before_check_error = find_error(*mod_gramm_schmidt(X,Y,T,d,e_cur, Er, i,used_i, complete))

    #Capture results after running check_error in a case that should return complete == True
    params_after_check_error = check_error(*params_before_check_error)

    #Test that check_error modifies complete to equal True.
    assert params_after_check_error[-1] == True != params_before_check_error[-1]

def test_decomposition_incomplete():
    """Test case for check_error when the error requirement is not met. d increases to 1 for the next iteration of RBD."""

    #Setup parameters
    X,Y,T,d,e_cur, i,used_i, complete = start_state(test_image)
    Er = 0.000000000000001

    #Capture results of RBD at the point before check_error is run.
    params_before_check_error = find_error(*mod_gramm_schmidt(X,Y,T,d,e_cur, Er, i,used_i, complete))
    
    #Capture results after running check_error in a case that should return complete == False
    params_after_check_error = check_error(*params_before_check_error)

    #Test that d is increased by 1 after check_error.
    d_before_check_error = params_before_check_error[3]
    d_after_check_error = params_after_check_error[3]
    assert d_before_check_error < d_after_check_error
    assert d_after_check_error - d_before_check_error == 1

     