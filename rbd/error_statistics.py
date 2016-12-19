import numpy as np
import math

def calculate_error_stats(X, decomp):
    """Function for capturing error statistics of a particular decomposition. Calculates Mean Square Error, Normalized Mean Square Error, Peak Signal to Noise Ratio, Root Mean Standard Deviation, and Normalized Root Mean Standard Deviation."""


    #Collect square errors of each corresponding cells in X and the decomposition
    square_errors = []
    square_xs = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            #sq_er metric is all that is needed for MSE
            sq_er = (X[i,j] - decomp[i,j])**2
            square_errors.append(sq_er)
            
            #square_x norm is the distance from a cell in the decomposition and the point of origin. Necessary for the NMSE metric.
            square_xs.append(decomp[i,j]**2)

    #Mean Square Error as the sum of squares divided by the size of the matrix.
    MSE = np.sum(square_errors)/X.size

    #Normalized Mean Square Error 
    NMSE_denominator = np.sum(square_xs)/X.size
    NMSE= MSE/NMSE_denominator

    #Peak Signal to Noise Ratio
    PSNR = 20 * math.log((255/np.sqrt(MSE)), 10)

    #Root Mean Standard Deviation
    RMSD = np.sqrt(MSE)

    #Normalized Root Mean Standard Deviation.
    NRMSD = RMSD/np.mean(square_errors)

    return MSE, NMSE, RMSD, NRMSD, PSNR
