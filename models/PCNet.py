# -*- coding: UTF-8 -*-
import tensorflow as tf
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

# 定义前向网络
# point_cloud:(32, 2048, 3)
def forward(point_clouds):

    batch_size = point_clouds.get_shape()[0].value
    num_point = point_clouds.get_shape()[1].value

    point_clouds_expand=tf.expand_dims(point_clouds,-1)
    # point_cloud_expands:(32, 2048, 3, 1)
    net = tf_util.conv2d(point_clouds_expand,[1,3,1,64])
    net = tf_util.conv2d(net,[1,1,64,64])
    net = tf_util.conv2d(net,[1,1,64,128])
    net = tf_util.conv2d(net,[1,1,128,1024])
    # net (3, 2048, 1, 1024)-->池化
    net = tf.nn.max_pool(net,[1,2048,1,1],[1,1,1,1],padding='VALID')
    # net (32, 1, 1, 1024)
    net = tf.reshape(net, [32, -1])
    net = tf_util.fully_connected(net,512)
    net = tf.nn.dropout(net,keep_prob=0.7)
    net = tf_util.fully_connected(net, 256)
    net = tf.nn.dropout(net, keep_prob=0.7)
    net = tf_util.fully_connected(net, 128)
    net = tf.nn.dropout(net, keep_prob=0.7)
    net = tf_util.fully_connected(net, 40)
    return net


