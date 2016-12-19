import numpy as np
from setup import start_state
from mod_gramm_schmidt import mod_gramm_schmidt
from find_error import find_error
from check_error import check_error
import matplotlib.image as mpimg
from error_statistics import calculate_error_stats

def rbd(X, dmax, Er=0.01):
    """Decompose a matrix X using reduced basis decomposition. X = YT. Y basis vectors are constructed using mod_gramm_schmidt until an maximum basis number threshold (dmax) or an error requirement (Er) are satisfied."""

    #Call start_state from the setup.py module to instantiate the parameters for the RBD
    X,Y,T,d,e_cur, i,used_i, complete = start_state(X)

    #Set the current_state to the initial parameters; subject to iterative change.
    current_state = [X,Y,T,d,e_cur,Er,i,used_i,complete]

    #While the requirements for the RBD have not been met...
    while d <= dmax and e_cur > Er and complete == False:

        #Orthogonally project the X[:,i] vector into Y, calling the mod_gramm_schmidt.py module.
        orthonormalization = mod_gramm_schmidt(*current_state)

        #mod_gramm_schmidt has a clause that signals complete to equal True.
        complete = orthonormalization[-1]

        #Break the while loop if mod_gramm_schmidt tells us the decomposition is complete.
        if complete == True:
            current_state = orthonormalization
            break

        else: 
            #Prepare for the next round of orthonormalization by calling the find_error.py and check_error.py modules. 
            current_state  = check_error(*find_error(*orthonormalization))

            #Update the active parameters, some of which govern the while loop. check_error() has a clause that signals completion.
            X,Y,T,d,e_cur,Er,i,used_i,complete = current_state
           
    #Final output of the function is a list of all the parameters.
    return current_state

if __name__ == '__main__':
    print("SUCCESSFUL COMPILE!")
    alist = np.array([[1,1,0,1,0],[1,0,1,1,0],[0,1,1,1,0]])
    test_rbd = rbd(alist, dmax=4)

    def display_state(attributes):
        names = ['X', 'Y', 'T','d' ,'e_cur', 'Er', 'i', 'used_i', 'complete']
        for i in range(len(names)):
            print("\n\n", names[i], " :\n", attributes[i])


    def image_test():
        image = mpimg.imread('images/DrawingHands.jpg')
        #tenbyten = np.array(image)[:,:,0] 
        tenbyten = image
        tenbyten_test = rbd(tenbyten, dmax=30, Er = 0.05)
        X,Y,T,d,e_cur,Er,i,used_i,complete = tenbyten_test
        decomp = np.dot(Y,T)
        print("DECOMP: \n", decomp)
        print("INPUT: \n", X)

        display_state(tenbyten_test)
        MSE, NMSE, RMSD, NRMSD, PSNR = calculate_error_stats(X, decomp)

        print("MSE: ", MSE)
        print("NMSE: ", NMSE)
        print("RMSD: ", RMSD)
        print("NRMSD: ", NRMSD)
        print("PSNR: ", PSNR)

    
