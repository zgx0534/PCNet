import tensorflow as tf

def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def conv2d(point_clouds,kernal_shape):
    return tf.nn.conv2d(point_clouds,get_weight(kernal_shape),[1,1,1,1],padding='VALID')
