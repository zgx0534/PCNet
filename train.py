# -*- coding: UTF-8 -*-
import os
import sys
import importlib
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import myProvider

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):

            TRAIN_FILES = myProvider.getDataFiles( \
                os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))



if __name__ == "__main__":
    train()
