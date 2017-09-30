import tensorflow as tf

sess = tf.Session()

const = tf.constant(100)
fill_mat = tf.fill((3,3), 123)
zeros = tf.zeros((3,3))
ones = tf.ones((3,3))
randn = tf.random_normal((3,3), mean=0.0, stddev=1.0)
randu = tf.random_uniform((3,3))

operations = [const, fill_mat, zeros, ones, randn, randu]

for op in operations:
    print (sess.run(op))
