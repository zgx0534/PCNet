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
import tf_util

BATCH_SIZE = 32
NUM_POINT = 2048
LEARN_RATE_BASE = 0.5
LEARN_RATE_DECAY = 0.5
LEARN_RATE_BATCH_SIZE = 20000

TRAIN_FILES = myProvider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
MODEL = importlib.import_module("PCNet")


def get_learning_rate(global_step):
    bn_momentum = tf.train.exponential_decay(
        learning_rate=LEARN_RATE_BASE,
        global_step=global_step,
        decay_steps=LEARN_RATE_BATCH_SIZE,
        decay_rate=LEARN_RATE_DECAY,
        staircase=True
    )
    # bn_decay不能大于0.99
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            # 通过前向传播得到一组的平均loss
            pointclouds_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
            labels_ph = tf.placeholder(tf.int32, shape=(BATCH_SIZE))
            pred = MODEL.forward(pointclouds_ph)
            loss = tf_util.get_loss(pred, labels_ph)
            # 获得正确率(一组正确数/一组总数)
            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_ph))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)

            # 后向传播定义(学习率+优化器)
            global_step = tf.Variable(0, trainable=False)
            learning_rate = get_learning_rate(global_step)
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

            train_op = optimizer.minimize(loss, global_step=global_step)
            saver = tf.train.Saver()

        # 声明会话(动态分配显存 允许自动调用设备 不输出设备信息 )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)

        param_dict={'pointclouds_ph':pointclouds_ph,
                    'labels_ph':labels_ph,

        }



        MAX_EPOCH = 250
        for epoch in range(MAX_EPOCH):
            train_one_epoch(sess,param_dict)


def train_one_epoch(sess,param_dict):
    train_file_index = np.arange(0, len(TRAIN_FILES))
    # 每次以不同顺序读取这5个文件
    np.random.shuffle(train_file_index)
    for idex in range(len(TRAIN_FILES)):
        # 得到文件中点云和标签,并打乱
        data_file_size, labels_file_size = myProvider.load_h5(TRAIN_FILES[train_file_index[idex]])
        num_data = data_file_size.shape[0]
        data_file_size, labels_file_size = myProvider.shuffle_data(data_file_size, labels_file_size)

        num_batch = num_data // BATCH_SIZE

        for batch_idx in range(num_batch):
            # 取出一个BATCH_SIZE的点云喂入网络
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            data_batch_size = data_file_size[start_idx:end_idx]
            labels_batch_size = labels_file_size[start_idx:end_idx]

            # ×××××××××××××××××××××××××××××××××××××××
            # ××××××××此处待添加旋转和抖动××××××××××××
            # ×××××××××××××××××××××××××××××××××××××××
            feed_dict = {param_dict['pointclouds_ph']: data_batch_size, param_dict['labels_ph']: labels_batch_size}
            accuracy = sess.run('accuracy', feed_dict=feed_dict)
            print ('accuracy: %f' % (accuracy))


if __name__ == "__main__":
    train()
