import numpy as np
import matplotlib.image as mpimg
from setup import start_state
from mod_gramm_schmidt import mod_gramm_schmidt
from find_error import find_error
from check_error import check_error

def test_find_new_vector():
    """Test that find_error returns new e_cur and i values. """

    #Setup parameters
    test_image = mpimg.imread('images/DrawingHands.jpg')
    X,Y,T,d,e_cur, i,used_i, complete = start_state(test_image)
    Er = 0.0000000001

    #Complete a full iteration of RBD
    d1_parameters = check_error(*find_error(*mod_gramm_schmidt(X,Y,T,d,e_cur, Er, i,used_i, complete)))

    #Take a snapshot of e_cur and i before running the find_error function
    e_cur_before_find_error = d1_parameters[4]
    i_before_find_error = d1_parameters[6]

    #Capture the results of find_error
    find_error_results = find_error(*mod_gramm_schmidt(*d1_parameters))
    e_cur_after_find_error = find_error_results[4]
    i_after_find_error = find_error_results[6]
    used_i = find_error_results[7]

    #Assert that the new e_cur and i values returned by find_error are distinct.
    assert e_cur_before_find_error != e_cur_after_find_error
    assert i_before_find_error != i_after_find_error
    assert i_after_find_error not in used_i