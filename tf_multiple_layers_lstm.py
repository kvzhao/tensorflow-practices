import tensorflow as tf

num_units = 32
num_layers = 3
dropout = tf.placeholder(tf.float32)

cells = []
for _ in range(num_layers):
    cell = tf.contrib.rnn.GRUCell(num_units)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob = 1.0-dropout
    )
    cells.append(cell)
cell = tf.contrib.rnn.MultiRNNCell(cells)

print (cell)
print ('done.')