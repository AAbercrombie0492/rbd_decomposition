import numpy as np

def mod_gramm_schmidt(X,Y,T,d,e_cur,Er,i,used_i,complete):
    """orthogonally project an X column vector into the orthonormal Y basis."""

    #Pick a vector from the input matrix X
    Xvector = X[:, i]

    #If this is the first vector to be projected into the orthonormal basis Y...
    if d == 0:
        #Normalize the vector by dividing the vector by its magnitude
        Xvector_norm = np.linalg.norm(Xvector)
        first_Ybasis = Xvector/Xvector_norm

        #Change the 1st column of Y from a zero vector to the normalized vector
        Y[:, d] = first_Ybasis.copy()

        #Change the 1st row of T from a zero vector to the dot product of the normalized vector's transpose and X. This ensures that an image of X can be reconstructed by the dot product of Y and T.
        T[d, :] = np.dot(Y[:,d].T, X)

        #append the index of the chosen vector to a list for book-keeping purposes.        
        used_i.append(i)

        #Return an updated state of all the RBD parameters, leading to the find_error process of the algorithm.
        update_state = (X,Y,T,d,e_cur,Er,i,used_i,complete)
        return update_state

    #Otherwise, Gramm-Schmidt is used to orthogonally project a chosen vector into the Y space.
    else:
        #u_d is the dth vector to be chosen by the find_error module.
        u_d = Xvector.copy()
        for j in range(d):
            #Subtract the projection of u_d onto the jth vector in Y from u_d   
            proj = np.dot(u_d, Y[:,j]) * Y[:,j]
            u_d -= proj

        #If the magnitude of u_d, which represents its error correcting power, falls below the error requirement, complete the RBD decomposition process.
        if np.linalg.norm(u_d) <= Er:
            Y = Y[:, d-1]
            T = T[:, d-1]
            update_complete = True

            #return parameters with complete == True signaling the end of RBD.
            decomposition = (X,Y,T,d,e_cur,Er,i,used_i,update_complete)
            return decomposition

        #If the error requirement has not been met, normalize u_d and add it to Y as a column vector. Add the dot product of u_d.transpose and X to T
        else:
            basis_vector = np.divide(u_d, np.linalg.norm(u_d))
            Y = np.column_stack((Y, basis_vector))
            T = np.row_stack((T, np.dot(Y[:,d].T, X)))
            used_i.append(i)

            #Return updated parameters, leading to the find_error process of the algorithm.
            update_state = (X,Y,T,d,e_cur,Er,i,used_i,complete)

            return update_state

if __name__ == '__main__':
    from setup import start_state
    alist = np.array([[1,1,0,1,0],[1,0,1,1,0],[0,1,1,1,0]])
    X,Y,T,d,e_cur, i,used_i, complete = start_state(alist)
    Er = 0.000001

    test_results = mod_gramm_schmidt(X,Y,T,d,e_cur,Er,i,used_i,complete)

    X,Y,T,d,e_cur,Er,i,used_i,complete = test_results

    def display_state(attributes):
        names = ['X', 'Y', 'U' ,'d' ,'e_cur', 'Er', 'i', 'used_i', 'complete']
        for i in range(len(names)):
            print("\n\n", names[i], " :\n", attributes[i])





