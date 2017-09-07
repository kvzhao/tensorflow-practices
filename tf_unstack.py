import tensorflow as tf
import numpy as np

sess = tf.Session()

batch_size = 2
max_seq_len = 5
x_dim = 5

# format similars to my current data
x = np.random.randn(batch_size, x_dim)

'''
    unstack
    args:
        value:
        number:
        axis:
        name:
'''
x = tf.unstack(x, max_seq_len, 1)

print (x)