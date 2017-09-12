import tensorflow as tf
import numpy as np

sess = tf.Session()

inputs =[
            [[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]]
        ]
print (inputs[0])
print (inputs[1])
print (inputs[2])
print ('\n')

print (sess.run(tf.slice(inputs, begin=[1, 0, 0], size=[1, 1, 3])))
print ('\n')
print (sess.run(tf.slice(inputs, begin=[1, 0, 0], size=[1, 2, 3])))
print ('\n')
print (sess.run(tf.slice(inputs, begin=[1, 0, 0], size=[2, 1, 3])))
print ('\n')
print (sess.run(tf.slice(inputs, begin=[1, 0, 0], size=[2, 2, 3])))