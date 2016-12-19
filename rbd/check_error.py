import numpy as np

def check_error(X,Y,T,d,e_cur,Er,i,used_i,complete):
    """Evaluate whether the decomposition is complete and prepare for the next round of mod_gramm_schmidt"""

    #If the error discrepancy of the next X column vector meets the error requirement, finalize the RBD process.
    if e_cur <= Er:
        Y = Y[:, :d]
        T = T[:d, :]

        #Complete == True signals the end of the process.
        update_complete = True
        update_attributes = X,Y,T,d,e_cur,Er,i,used_i,update_complete

    #Otherwise, there is a need to continue adding basis vectors to Y. Increasing d is the last step before repeating the gramm-schmidt process.
    else:
        grow_d = d + 1
        update_attributes = X,Y,T,grow_d,e_cur,Er,i,used_i,complete

    #Return the parameters to the main function: rbd 
    return update_attributes


if __name__ == '__main__':
    from setup import start_state
    from mod_gramm_schmidt import mod_gramm_schmidt
    from find_error import find_error
    from error_statistics import calculate_error_stats
    
    alist = np.array([[1,1,0,1,0],[1,0,1,1,0],[0,1,1,1,0]])
    X,Y,T,d,e_cur,i,used_i, complete = start_state(alist)
    Er = 0.000001

    check_error_test = check_error(*find_error(*mod_gramm_schmidt(X,Y,T,d,e_cur,Er,i,used_i,complete)))
  

    def display_state(attributes):
        names = ['X', 'Y', 'T' ,'d' ,'e_cur', 'Er', 'i', 'used_i', 'complete']
        for i in range(len(names)):
            print("\n\n", names[i], " :\n", attributes[i])

    #display_state(check_error_test)