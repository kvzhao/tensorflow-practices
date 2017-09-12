import tensorflow as tf
import numpy as np

from toy_seq_data import ToySequenceData

tf.reset_default_graph()
sess = tf.Session()

batch_size = 2
max_seq_len = 8
x_dim = 1

# Shape = [batch_size, max_time, n_inputs (num of features)]
X = np.random.randn(batch_size, max_seq_len, x_dim)

# The second example is of length 6 
X[1,6:] = 0
X_lengths = [10, 6]

# shape = batchsize, seq_len value
seq_len = tf.placeholder(tf.int32, [None, ])
# shape = batchsize, max_time, dim of feature
seq     = tf.placeholder(tf.float32, [None, max_seq_len, x_dim])

cell = tf.nn.rnn_cell.LSTMCell(num_units=12, state_is_tuple=True)
out, h = tf.nn.dynamic_rnn(cell, 
                            inputs=seq,
                            sequence_length=seq_len,
                            dtype=tf.float32,
                            time_major=False)

# shape of output = [batchsize, max_time, cell_state_size]

sess.run(tf.global_variables_initializer())

res = sess.run(out, 
            feed_dict={
                seq: X,
                seq_len: X_lengths
            })
print (res)
print (res.shape)
# [batchsize, max_len(time), hidden size]

print (res)
print (res.shape)