import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 32])
print (x)
y = tf.reshape(x, [-1, 8, 4])
print (y)