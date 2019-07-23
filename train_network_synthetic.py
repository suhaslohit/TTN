import tensorflow as tf
import numpy as np 
import network_ttn_synthetic as network
import os.path
import scipy.io

batch_size = 32
learning_rate_1 = 0.0001
num_train = 8000
numBatches = num_train/batch_size
maxIters = 10000
num_classes = 2
seq_len = 100

train_data = scipy.io.loadmat('synthetic_data_train_2_gaussians.mat')['train_data']
train_label = scipy.io.loadmat('synthetic_data_train_2_gaussians.mat')['train_label']

def placeholder_inputs(batch_size):
	x_placeholder = tf.placeholder(tf.float32, shape=(batch_size, seq_len))
	y_placeholder = tf.placeholder(tf.float32, shape=(batch_size, num_classes))
	learning_rate_placeholder = tf.placeholder(tf.float32, shape=())
	return x_placeholder, y_placeholder, learning_rate_placeholder


with tf.Graph().as_default():
	x_placeholder, y_placeholder, learning_rate_placeholder = placeholder_inputs(batch_size)
	output, sequence_unwarped, gamma, sequence1 = network.mapping(x_placeholder, batch_size)
	loss = network.loss(output, y_placeholder)
	
	var_list_ttn = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ttn')
	var_list_classifier = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')

	train_op_classifier = network.training(loss, learning_rate_placeholder, var_list_classifier)
	train_op_ttn = network.training(loss, learning_rate_placeholder/10.0, var_list_ttn)
	train_op = tf.group(train_op_classifier, train_op_ttn)

	# train_op = network.training(loss, learning_rate_placeholder)

	init = tf.initialize_all_variables()
	saver = tf.train.Saver(max_to_keep=500)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	sess.run(init)

	
	batchIdx = 0
	for step in xrange(maxIters):
		batchIdx = batchIdx % numBatches

		if batchIdx == 0:
			randIdx = np.random.permutation(train_data.shape[0])

		trainIdx = randIdx[batchIdx*batch_size:(batchIdx+1)*batch_size]

		_, loss_value = sess.run([train_op, loss], 
								  feed_dict={x_placeholder: train_data[trainIdx,:],
								  			y_placeholder: train_label[trainIdx,:],
								  			learning_rate_placeholder: learning_rate_1})

		if step % 1000 == 0:
			print '---------------------------------'
			print step
			print loss_value

		batchIdx = batchIdx + 1

	saver.save(sess, './2_gaussians_github_ttn')