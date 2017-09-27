import tensorflow as tf

sess = tf.Session()

t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
print(sess.run(tf.concat([t1, t2], 0)))  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
print(sess.run(tf.concat([t1, t2], 1)))  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
