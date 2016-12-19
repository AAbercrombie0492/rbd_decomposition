# Image compression with Reduced Basis Decomposition

The directory rbd contains a set of modules that constitute the RBD algorithm designed by Doctor Yanlai Chen from the department of mathematics at the University of Massachusetts, Dartmouth (2015). The paper has been included in this directory.

In the rbd directory, you will also find an ipython notebook that features a walk through of the RBD as applied to a collection of 150 images. The notebook makes use of the following files:
  - RBD_Visualization.ipynb
  - images
  - Escher_img_stats.csv
  - face150_stats.csv

rbd.py is the master module. It is built on the following sub-modules:
  - setup.py
  - mod_gramm_schmidt.py
  - find_error.py
  - check_error.py

You will also find a set of test modules that can be ran in bulk with nosetests:
  - test_rbd.pyy
  - test_setup.py
  - test_mod_gramm_schmidt.py
  - test_find_error.py
  - test_check_error.py

Finally, I've included a set of modules that collect statistics on rbd decompositions.
  - error_statistcs.py (computes runtime, MSE, PSNR, and a few other metrics)
  - gather_basis_statistics.py (computes error statistics for each count of basis vectors that can constitute the decomposition)
  - image_dataset_stats.py (Computes error stats for each count of basis vectors for each .pmg image in the images directory)
