import tensorflow as tf


def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def conv2d(point_clouds, kernal_shape):
    return tf.nn.conv2d(point_clouds, get_weight(kernal_shape), [1, 1, 1, 1], padding='VALID')


def fully_connected(inputs, num_outputs):
    num_input_units = inputs.get_shape()[-1].value
    weights = get_weight([num_input_units, num_outputs])
    outputs = tf.matmul(inputs, weights)
    biases = get_weight([num_outputs])
    outputs=tf.nn.bias_add(outputs,biases)
    return outputs

def get_loss(pred,labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels)
    batch_avg_loss = tf.reduce_mean(loss)
    return batch_avg_loss

