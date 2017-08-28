import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

targets = [12, 3, 5, 8]
print ('input target: {}'.format(targets))

onehot = tf.one_hot(indices=targets, depth=15)
out = sess.run(onehot)

print ('one hot repr {}'.format(out))
print ('Nonzero indices {}'.format(np.nonzero(out)[1]))
