# -*- coding: UTF-8 -*-
import os
import h5py
f = h5py.File("/home/jensen/ProgramData/PycharmProjects/PCNet/data/modelnet40_ply_hdf5_2048/ply_data_train0.h5")
# data = f['data'][:]
# label = f['label'][:]
# return (data, label)
print f['data'].shape
