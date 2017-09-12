'''
    https://www.tensorflow.org/api_docs/python/tf/one_hot
'''

import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

targets = [12, 3, 5, 8]
print ('input target: {}'.format(targets))

onehot = tf.one_hot(indices=targets, depth=15)
out = sess.run(onehot)

print ('one hot repr {}'.format(out))
print ('Nonzero indices {}'.format(np.nonzero(out)[1]))

# spin configuration
print (sess.run(tf.one_hot([1, -1, 1, 0, -1], depth=10)))