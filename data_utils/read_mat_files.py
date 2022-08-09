import scipy.io
import h5py
import numpy as np
from mat4py import loadmat

path = '/home/ducanh/hain/dataset/MPI_INF/mpi_inf_3dhp/S1/Seq1/annot.mat'
mat = scipy.io.loadmat(path)

# mat = loadmat(path)

print(mat['cameras'])