import numpy as np
import matplotlib.image as mpimg
from setup import start_state

def test_rbd_setup():
    '''Test harness for the setup.py module'''   
    
    test_image = mpimg.imread('images/DrawingHands.jpg')
    dmax = 40
    X,Y,T,d,e_cur, i,used_i, complete = start_state(test_image)

    #Test datatypes and values
    assert type(X) == np.ndarray
    assert type(Y) == np.ndarray
    assert type(T) == np.ndarray
    assert type(d) == int
    assert type(e_cur) == float
    assert type(i) == int
    assert type(used_i) == list
    assert complete == False
    assert d < dmax

    #Test shapes
    assert Y.shape[0] == X.shape[0]
    assert Y.shape[1] == 1
    assert T.shape[0] == 1
    assert T.shape[1] == X.shape[1]
    assert len(used_i) == 0
    assert d < dmax




