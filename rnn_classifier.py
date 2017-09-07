import numpy as np
import tensorflow as tf

sess = tf.Session()

num_units = 32
num_layers = 3
batch_size = 2
time_steps = 10
num_features = 28
num_classes = 5

print ('{} layers of GRU with {} hidden units'.format(num_layers, num_units))

dropout = tf.placeholder(tf.float32)

cells = []
for _ in range(num_layers):
    cell = tf.contrib.rnn.GRUCell(num_units)  # Or LSTMCell(num_units)
    cells.append(cell)
cell = tf.contrib.rnn.MultiRNNCell(cells)

# Batch size x time steps x features.
data = tf.placeholder(tf.float32, [batch_size, time_steps, num_features], name='data')
target = tf.placeholder(tf.float32, [batch_size, num_classes], name='target')

# For classification, we only look at the output activation at the last time step
output, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

print ('output shape: {}'.format(output.get_shape()))

# transpose to time steps x batch x features
output = tf.transpose(output, [1, 0, 2])

print ('transposed output shape: {}'.format(output.get_shape()))
# indices = last index of time direction
last = tf.gather(output, int(output.get_shape()[0]-1))
print ('last shape: {}'.format(last.get_shape()))

outsize = target.get_shape()[-1].value
print ('outsize: {} = num_classes'.format(outsize))

# connect the last output state to projection layer
# whose shape is num_units x num_classes
logit = tf.contrib.layers.fully_connected(
    last, outsize, activation_fn=None)

print (logit)

# define the loss function
prediction = tf.nn.softmax(logit)
loss = tf.losses.softmax_cross_entropy(target, logit)

# write the graph
writer = tf.summary.FileWriter('./graph/rnn_classifier/', sess.graph)
writer.close()
