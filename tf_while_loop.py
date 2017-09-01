import tensorflow as tf

'''
    tf.while_loop(cond, body, loop_vars, parallel_iterations=10, swap_memory=10)
'''

ijk_0 = (0, (1, 2))
condition = lambda i, _: i < 10
body = lambda i, jk: (i+1, (jk[0]+jk[1], jk[0]-jk[1]))
(i_final, j_final) = tf.while_loop(condition, body, ijk_0)

sess = tf.Session()
print (sess.run([i_final, j_final]))

# 1, 2
# 3, -1
# 2, 4
# 6, -2
# 4, 8
# ...
# 32, 64
