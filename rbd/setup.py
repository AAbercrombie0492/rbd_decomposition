

import numpy as np

#np.random.seed(0)

def start_state(X):
    '''Read in a matrix and establish core attributes for RBD decomposition.'''

    #Ensure the input matrix X will be an array of floats
    if type(X) == list:
        X = np.array(X).astype(np.float128)

    elif type(X) == np.ndarray:
        X = X.astype(np.float128) 
    else:
        raise TypeError('Input must be a list or an np.ndarray')

    #d is used to track the orthogonal vector that is under construction. It increases in steps of one.                
    d = 0

    #e_cur is a measure of maximum error in regards to the column vectors of X and the column vectors of the decomposition. 
    e_cur = np.inf

    #i is used to index the input matrix X. i is determined so that X[:,i] corresponds with e_cur
    i = np.random.randint(0, X.shape[1])

    #Y is an orthogonal matrix of basis vectors that is grown sequentially by the Gramm-Schmidt process. The dot product of Y and T produces a compression of X.
    Y = np.zeros(shape=(X.shape[0], 1)).astype(np.float128)
    T = np.zeros(shape=(1,X.shape[1])).astype(np.float128)

    #used_i holds the indexes of X used to construct Y and T in the order they were selected. Each element in used_i should be unique.
    used_i = []

    #complete is used to halt the algorithm and is subject to change if the error requirement Er is met.
    complete = False

    
    attributes = (X,Y,T,d,e_cur, i,used_i, complete)

    return attributes
                

if __name__ == '__main__':
    alist = np.array([[1,1,0],[1,0,1],[0,1,1]])
    base_attributes = start_state(alist)

    def display_start_state(attributes):
        names = ['X','Y','T','d','e_cur','i','used_i','complete']
        for i in range(len(names)):
            print("\n\n", names[i], " :\n", attributes[i])




