import numpy as np
import tensorflow as tf

sess = tf.Session()
x = tf.constant([1, 2, 3, 4, 5, 6])
gi = [2, 0, 1, 2]
y = tf.gather(x, gi)

print (sess.run(x))
print ('gather indices: {}'.format(gi))
print (sess.run(y))