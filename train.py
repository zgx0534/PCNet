# -*- coding: UTF-8 -*-
import os
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#设置环境变量和路径信息
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join('utils'))
sys.path.append(os.path.join('models'))
TRAIN_FILES_DIR=os.path.join(BASE_DIR,'data/modelnet40_ply_hdf5_2048/train_files.txt')

import providerUtil

#获得数据
TRAIN_FILES=providerUtil.getDataFiles(os.path.join(TRAIN_FILES_DIR))
FileHandle=h5py.File(TRAIN_FILES[0],'r')
p=FileHandle['data'][0]

#绘制空间形状



