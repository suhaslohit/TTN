import tensorflow as tf
import numpy as np
import ttn

seq_len = 100
num_classes = 2


def weight_variable_zero_init(shape,name):
  w = tf.get_variable(name,shape=shape, initializer=tf.constant_initializer(0.0))
  return w

def weight_variable(shape,name):
  w = tf.get_variable(name,shape=shape, initializer=tf.contrib.layers.xavier_initializer())
  return w

def bias_variable(shape,name):
  b = tf.get_variable(name,shape=shape, initializer=tf.constant_initializer(0.1))
  return b

def conv1d(x, W, stride):
  return tf.nn.conv1d(x, W, stride=stride, padding='SAME')

def mapping(sequence, batch_size):

  # Normalizing the sequence
  sequence1 = sequence - tf.tile(tf.reduce_mean(sequence, 1, keep_dims=True), [1, seq_len])
  sequence_norm = tf.sqrt(tf.reduce_sum(tf.square(sequence1), 1, keep_dims=True))
  sequence_norm_tile = tf.tile(sequence_norm, [1, seq_len])
  sequence1 = tf.div(sequence1, sequence_norm_tile)

  sequence1 = tf.reshape(sequence1, [batch_size, seq_len, 1])

  # TTN
  with tf.variable_scope('ttn'):
    conv_size = 8
    num_channels = 1

    wg1 = weight_variable([conv_size,1,num_channels],'wg1')
    bg1 = bias_variable([num_channels],'bg1')
    hg1 = tf.nn.tanh(conv1d(sequence1, wg1, 1) + bg1)

    hg2 = tf.reshape(hg1, [batch_size, 100*num_channels])

    W_fc1g = weight_variable_zero_init([100*num_channels,99],'W_fc1g')
    b_fc1g = bias_variable([99],'b_fc1g')
    out1g = tf.nn.tanh(tf.nn.dropout(tf.matmul(hg2, W_fc1g) + b_fc1g, keep_prob=0.8))

  # Constraint satisfaction layers
  temp = tf.sqrt(tf.reduce_sum(tf.square(out1g), 1, keep_dims=True))
  batch_temp = tf.tile(temp, [1, seq_len-1])
  gamma_dot = tf.square(tf.div(out1g, batch_temp))
  gamma = tf.cumsum(gamma_dot, axis=1)
  zeros_vector = tf.zeros([batch_size,1])
  gamma = tf.concat([zeros_vector, gamma],1)
  gamma = gamma*(seq_len-1)

  # Warping layer
  sequence_warped = ttn.warp(tf.reshape(sequence1,[batch_size, seq_len, 1]), gamma)
  sequence_warped = tf.reshape(sequence_warped, [batch_size, seq_len])


  tf.set_random_seed(1234)

  # Classifier
  with tf.variable_scope('classifier'):
    W_fc1 = weight_variable([100, num_classes],'W_fc1')
    b_fc1 = bias_variable([num_classes],'b_fc1')
    logits = tf.matmul(sequence_warped, W_fc1) + b_fc1

  return logits, sequence_warped, gamma, sequence1

def loss(logits, labels):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
	return cross_entropy

def training(loss, learning_rate, var_list):
  optimizer = tf.train.AdamOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step, var_list=var_list)
  return train_op
