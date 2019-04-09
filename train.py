# -*- coding: UTF-8 -*-
import os
import sys
import importlib
import tensorflow as tf
import numpy as np

# 环境变量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import myProvider

BATCH_SIZE=32
TRAIN_FILES = myProvider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
MODEL = importlib.import_module("PCNet")

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            pass

        # 声明会话(动态分配显存 允许自动调用设备 不输出设备信息 )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        MAX_EPOCH=250
        for epoch in range(MAX_EPOCH):
            train_one_epoch(sess)

def train_one_epoch(sess):
    train_file_index = np.arange(0, len(TRAIN_FILES))
    # 每次以不痛顺序读取这5个文件
    np.random.shuffle(train_file_index)
    for idex in range(len(TRAIN_FILES)):
        # 得到文件中点云和标签,并打乱
        current_data, current_label = myProvider.load_h5(TRAIN_FILES[train_file_index[idex]])
        NUM_DATA = current_data.shape[0]
        current_data, current_label = myProvider.shuffle_data(current_data, current_label)

        # 取出一个BATCH_SIZE的点云和标签


if __name__ == "__main__":
    train()
