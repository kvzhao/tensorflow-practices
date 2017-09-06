import tensorflow as tf
import numpy as np

from toy_seq_data import ToySequenceData

tf.reset_default_graph()
sess = tf.Session()

batch_size = 3
max_seq_len = 8
x_dim = 1

toy = ToySequenceData(n_samples=100, max_seq_len=max_seq_len, min_seq_len=3)

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

sequence, label, seqlen = toy.next(batch_size)

print ('the input sequences are')
print (sequence)
print ('with correspoing length')
print (seqlen)

for b, x in enumerate(sequence):
    print ('the batch {} with len {}'.format(b, len(x)))
    print ('num of non-zero elems is {}'.format(len(np.nonzero(x)[0])))
    for t, xi in enumerate(x):
        print ('\tx[{}] = {}'.format(t, xi))

res = sess.run(out, 
            feed_dict={
                seq: sequence,
                seq_len: seqlen
            })

print (res)
print (res.shape)