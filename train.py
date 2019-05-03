# -*- coding: UTF-8 -*-
import os
import sys
import importlib
import tensorflow as tf
import numpy as np


# 环境变量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'log')
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import myProvider
import tf_util

BATCH_SIZE = 32
NUM_POINT = 2048
LEARN_RATE_BASE = 0.004
LEARN_RATE_DECAY = 0.9
LEARN_RATE_BATCH_SIZE = 500

TRAIN_FILES = myProvider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
MODEL = importlib.import_module("net_improved")

# 写log方法
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(global_step):
    learning_rate = tf.train.exponential_decay(
        learning_rate=LEARN_RATE_BASE,
        global_step=global_step,
        decay_steps=LEARN_RATE_BATCH_SIZE,
        decay_rate=LEARN_RATE_DECAY,
        staircase=True
    )
    learning_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            # 通过前向传播得到一组的概率
            pointclouds_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
            labels_ph = tf.placeholder(tf.int32, shape=(BATCH_SIZE))
            pred = MODEL.forward(pointclouds_ph)

            # 获得正确率(一组正确数/一组总数)
            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_ph))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # 后向传播定义(学习率+优化器)
            global_step = tf.Variable(0, trainable=False)
            learning_rate = get_learning_rate(global_step)
            tf.summary.scalar('learning_rate', learning_rate)

            loss = tf_util.get_loss(pred, labels_ph)
            tf.summary.scalar('loss', loss)
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

            train_op = optimizer.minimize(loss, global_step=global_step)
            saver = tf.train.Saver()

        # 声明会话(动态分配显存 允许自动调用设备 不输出设备信息 )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)

        param_dict = {'pointclouds_ph': pointclouds_ph,
                      'labels_ph': labels_ph,
                      'train_op': train_op,
                      'accuracy': accuracy,
                      'learning_rate': learning_rate,
                      'global_step': global_step,
                      'loss': loss,
                      'merged': merged,
                      }

        MAX_EPOCH = 250
        for epoch in range(MAX_EPOCH):
            log_string('******  EPOCH %3d  ******' % epoch)
            train_one_epoch(sess, param_dict, train_writer)


def train_one_epoch(sess, param_dict, train_writer):
    train_file_index = np.arange(0, len(TRAIN_FILES))

    # 每次以不同顺序读取这5个文件
    np.random.shuffle(train_file_index)
    for idex in range(len(TRAIN_FILES)):
        # 得到文件中点云和标签,并打乱
        data_file_size, labels_file_size = myProvider.load_h5(TRAIN_FILES[train_file_index[idex]])
        num_data = data_file_size.shape[0]
        data_file_size, labels_file_size = myProvider.shuffle_data(data_file_size, labels_file_size)

        num_batch = num_data // BATCH_SIZE

        accuracy_total = 0
        loss_total = 0

        # 取出一个batch训练
        for batch_idx in range(num_batch):
            # 取出一个BATCH_SIZE的点云喂入网络
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            data_batch_size = data_file_size[start_idx:end_idx]
            labels_batch_size = tf.squeeze(labels_file_size[start_idx:end_idx], 1)
            labels = sess.run(labels_batch_size)


            # ××××××  旋转和抖动  ××××××
            #rotated_data = myProvider.rotate_point_cloud(data_batch_size)
            # jittered_data = myProvider.jitter_point_cloud(data_batch_size)


            feed_dict = {param_dict['pointclouds_ph']: data_batch_size, param_dict['labels_ph']: labels}
            _, accuracy_val, learn_rate_val, loss_val, summary, step_val = sess.run(
                [param_dict['train_op'], param_dict['accuracy'], param_dict['learning_rate'], param_dict['loss'],
                 param_dict['merged'], param_dict['global_step']], feed_dict=feed_dict)
            train_writer.add_summary(summary,step_val)
            accuracy_total += accuracy_val
            loss_total += loss_val

        accuracy_total = (accuracy_total / num_batch) * 100
        loss_total /= num_batch
        log_string('accuracy: %5.2f%% loss: %f' % (accuracy_total, loss_total))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
