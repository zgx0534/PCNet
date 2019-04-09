# -*- coding: UTF-8 -*-
import tensorflow as tf
# 定义前向网络
# point_cloud:(32, 2048, 3)
def forward(point_cloud):
    point_cloud_expand=tf.expand_dims(point_cloud,-1)
    # point_cloud_expand:(32, 2048, 3, 1)
    print point_cloud_expand.shape
    net = tf.nn.conv2d(point_cloud_expand,[1,3,1,64],strides=[1,1])

