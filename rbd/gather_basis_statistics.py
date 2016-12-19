import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from error_statistics import calculate_error_stats
from rbd import rbd
from timeit import default_timer

def iterate_stats(matrix, return_df=False):
    """Calculates error and runtime statistics for RBD at each instance of dmax. Option to return output as a Pandas DataFrame."""

    #Setup list repositories.
    number_of_basis = []
    runtime = []
    e_cur_data = []
    MSE_data = []
    NMSE_data = []
    RMSD_data = []
    NRMSD_data = []
    PSNR_data = []

    for j in range(matrix.shape[1]):
        #Capture the runtime of RBD
        start = default_timer()
        X,Y,T,d,e_cur,Er,i,used_i,complete = rbd(matrix, dmax=j, Er= -np.inf)
        end = default_timer()

        #Gather error metrics using the error_statistics.py module
        MSE, NMSE, RMSD, NRMSD, PSNR = calculate_error_stats(X, np.dot(Y,T))
        
        #Append values to list repositories.
        number_of_basis.append(d)
        runtime.append(end - start)
        e_cur_data.append(e_cur)
        MSE_data.append(MSE)
        NMSE_data.append(NMSE)
        RMSD_data.append(RMSD)
        NRMSD_data.append(NRMSD)
        PSNR_data.append(PSNR)

    if return_df == True:
        #Setup Pandas DataFrame and return.
        stats_dataframe = pd.DataFrame(
            {'basis_count': number_of_basis,
            'runtime': runtime,
            'e_cur': e_cur_data,
            'MSE': MSE_data,
            'NMSE': NMSE_data,
            'RMSD': RMSD_data,
            'NRMSD': NRMSD_data,
            'PSNR': PSNR_data})

        return stats_dataframe

    else:
        #Return each series as lists
        return number_of_basis, runtime, e_cur, MSE_data, NMSE_data, RMSD_data, NRMSD_data, PSNR_data


if __name__ == '__main__':
      image = mpimg.imread('images/1a000.pgm')
      matrix = image
      #matrix = np.array(image)[:,:,0]
      stats_dataframe = iterate_stats(matrix)

