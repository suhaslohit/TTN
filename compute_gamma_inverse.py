# This function inverts a warping function batch-wise
# Input: gamma is a B x N array where B is the batch size and N is the sequence length

import tensorflow as tf 
import numpy as np

def invert_gamma(gamma, batch_size, seq_len):

	'''
	a = np.array([[0.0,0.8,2.0],[0.0,1.2,2.0]])

	gamma = tf.constant(a)

	batch_size = gamma.get_shape()[0] #tf.shape(gamma)[0]
	seq_len = gamma.get_shape()[1] #tf.shape(gamma)[1]

	batch_size = 2
	seq_len = 3
	'''

	false_vector = tf.constant(dtype=tf.bool,shape=[batch_size*(seq_len-2),1],value=False)

	input_indices = tf.reshape(tf.range(1,seq_len-1), [seq_len-2, 1])
	input_indices_tile = tf.tile(input_indices, [batch_size, 1])
	input_indices_tile_col = tf.tile(input_indices_tile, [1, seq_len])

	gamma = tf.reshape(gamma, [batch_size, seq_len, 1])
	gamma_tile = tf.tile(gamma, [1,1,seq_len-2])
	gamma_transpose = tf.transpose(gamma_tile, [0,2,1])
	gamma_reshape = tf.reshape(gamma_transpose, [batch_size*(seq_len-2), seq_len])

	tempa = tf.less_equal(gamma_reshape, tf.cast(input_indices_tile_col, tf.float64))
	tempb = tf.greater(gamma_reshape, tf.cast(input_indices_tile_col, tf.float64))

	not_tempa_ext = tf.logical_not(tf.slice(tf.concat(1, (tempa, false_vector)), [0,1], [batch_size*(seq_len-2), seq_len]))
	not_tempb_ext = tf.logical_not(tf.slice(tf.concat(1, (false_vector, tempb)), [0,0], [batch_size*(seq_len-2), seq_len]))

	tempa_and = tf.logical_and(tempa, not_tempa_ext)
	tempb_and = tf.logical_and(tempb, not_tempb_ext)

	temp1 = tf.where(tempa_and)
	temp2 = tf.where(tempb_and)

	index1 = tf.slice(temp1, [0,1], [batch_size*(seq_len-2), 1]) #temp1[:,0]
	index2 = tf.slice(temp2, [0,1], [batch_size*(seq_len-2), 1]) #temp2[:,0]

	# Now do interpolation
	gamma_reshape_flat = tf.reshape(gamma_reshape, [batch_size*(seq_len-2)*seq_len])
	index1_flat = tf.reshape(index1, [batch_size*(seq_len-2)])
	index2_flat = tf.reshape(index2, [batch_size*(seq_len-2)])

	# The offset vector for tf.gather
	range_vec = tf.range(batch_size)
	range_vec_tile = tf.tile(tf.expand_dims(range_vec, 1),[1,seq_len-2]) # CHECK
	range_vec_tile_vec = tf.reshape(range_vec_tile, [batch_size*(seq_len-2)])
	offset = tf.cast(range_vec_tile_vec*(seq_len), tf.int64)  # CHECK

	gamma_index1 = tf.gather(gamma_reshape_flat, index1_flat+offset)
	gamma_index2 = tf.gather(gamma_reshape_flat, index2_flat+offset)

	input_indices_tile_flat = tf.reshape(input_indices_tile, [batch_size*(seq_len-2)])

	temp_alpha = tf.div(tf.cast(index2_flat, tf.float64) - tf.cast(index1_flat, tf.float64), gamma_index2-gamma_index1)
	index = tf.squeeze(tf.cast(index1, tf.float64)) + tf.mul(temp_alpha, (tf.cast(input_indices_tile_flat, tf.float64)-gamma_index1))
	index_reshape = tf.reshape(index, [batch_size, seq_len-2])

	gamma_inverse = tf.concat(1,(tf.zeros((batch_size, 1)), tf.cast(index_reshape, tf.float32), (seq_len-1)*tf.ones((batch_size, 1))))

	return gamma_inverse

#sess=tf.Session()
#chk = gamma_inverse.eval(session=sess)
#print chk