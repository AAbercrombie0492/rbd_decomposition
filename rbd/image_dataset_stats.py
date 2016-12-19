import os
import glob
from gather_basis_statistics import iterate_stats
import numpy as np
import pandas as pd
import matplotlib.image

def crunch_images(return_df=False):
    """Crawls through the image directory and gathers RBD statistics on each .pgm image in the directory. Option to return a flattened Pandas DataFrame. This function takes a very long time to run."""


      #Setup repositories
    huge_number_of_basis = []
    huge_runtime = []
    huge_e_cur_data = []
    huge_MSE_data = []
    huge_NMSE_data = []
    huge_RMSD_data = []
    huge_NRMSD_data = []
    huge_PSNR_data = []
    for filename in glob.glob(os.path.join('images','*.pgm')):
        image = matplotlib.image.imread(filename)

        #Use the iterate_stats function from the gather_basis_statistics.py module to collect data.
        number_of_basis, runtime, e_cur, MSE_data, NMSE_data, RMSD_data,NRMSD_data, PSNR_data = iterate_stats(image, return_df=False)

        #Append observations to data repositories
        huge_number_of_basis.append(number_of_basis)
        huge_runtime.append(runtime)
        huge_e_cur_data.append(e_cur)
        huge_MSE_data.append(MSE_data)
        huge_NMSE_data.append(NMSE_data)
        huge_RMSD_data.append(RMSD_data)
        huge_NRMSD_data.append(NRMSD_data)
        huge_PSNR_data.append(PSNR_data)

    #Flatten the repositories and alter their datatypes
    huge_number_of_basis = np.array(huge_number_of_basis).flatten()
    huge_runtime = np.array(huge_runtime).flatten().astype(float)
    huge_e_cur_data = np.array(huge_e_cur_data).flatten().astype(float)
    huge_MSE_data = np.array(huge_MSE_data).flatten().astype(float)
    huge_NMSE_data = np.array(huge_NMSE_data).flatten().astype(float)
    huge_RMSD_data = np.array(huge_RMSD_data).flatten().astype(float)
    huge_NRMSD_data = np.array(huge_NRMSD_data).flatten().astype(float)
    huge_PSNR_data = np.array(huge_PSNR_data).flatten().astype(float)

    if return_df == True:
      #Setup the Pandas DataFrame.
        stats_dataframe = pd.DataFrame(
            {'basis_count': huge_number_of_basis,
            'runtime': huge_runtime,
            'e_cur': huge_e_cur_data,
            'MSE': huge_MSE_data,
            'NMSE': huge_NMSE_data,
            'RMSD': huge_RMSD_data,
            'NRMSD': huge_NRMSD_data,
            'PSNR': huge_PSNR_data})

        return stats_dataframe

    else:
        #Return each series as lists
        return huge_number_of_basis, huge_runtime, huge_e_cur_data, huge_MSE_data, huge_NMSE_data, huge_RMSD_data, huge_NRMSD_data, huge_PSNR_data
