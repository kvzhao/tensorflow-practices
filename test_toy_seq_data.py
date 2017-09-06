from toy_seq_data import ToySequenceData

import tensorflow as tf

sess = tf.Session()

max_seq_len = 12
batch_size = 5

data = ToySequenceData(n_samples=100, max_seq_len=max_seq_len)

seq_data, labels, seq_len = data.next(batch_size)

'''
    data shape is [batch_size, seqence_length (n_input)]
'''

print (seq_data)
print (seq_data[0])
print (seq_len)
print (labels)

# unstack (value, num, axis)
# Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
x = tf.unstack(seq_data, max_seq_len, 1)

out = tf.stack(x)

print (x)
print (sess.run(x))

print (out)
print (sess.run(out))