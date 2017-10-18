""" Distinguish shape and get_shape()
"""
import tensorflow as tf


input = tf.placeholder(dtype=tf.float32, shape=[None, 256], name="UnkonwBatchSize")

batch_size = input.shape[0]
print (batch_size)

batch_size = input.get_shape()[0]
print (batch_size)

# The Graph operation 
batch_size = tf.shape(input)[0]
print (batch_size)