import numpy as np

def find_error(X,Y,T,d,e_cur,Er,i,used_i,complete):
    """Evaluate the error discrepancy of each X column vector and the current decomposition's column vectors and select the optimal X vector for the next round of mod_gramm_schmidt."""

    #Calculate the error discrepancy of between each X column vector and the decomposition's corresponding column vector.
    error_list = [np.linalg.norm(X[:,j] - np.dot(Y, T[:,j])) for j in range(X.shape[1])]

    #current error is set to the highest error from the comparisons.
    new_e_cur = np.max(error_list)
    
    #The next column-index of X to project into the Y space is selected as the column vector with the highest error discrepancy.
    next_i = np.argmax(error_list)

    #Return updated parameters, leading to the check_error function
    updated_attributes = X,Y,T,d,new_e_cur,Er,next_i,used_i,complete
    return updated_attributes

if __name__ == '__main__':
    from setup import start_state
    from mod_gramm_schmidt import mod_gramm_schmidt
    from error_statistics import calculate_error_stats

    alist = np.array([[1,1,0,1,0],[1,0,1,1,0],[0,1,1,1,0]])
    X,Y,T,d,e_cur,i,used_i, complete = start_state(alist)
    Er = 0.000001

    orthonormalized = mod_gramm_schmidt(X,Y,T,d,e_cur,Er,i,used_i,complete)
    error_comparison_results = find_error(*orthonormalized)

    def display_state(attributes):
        names = ['X', 'Y', 'U' ,'d' ,'e_cur', 'Er', 'i', 'used_i', 'complete']
        for i in range(len(names)):
            print("\n\n", names[i], " :\n", attributes[i])


