"""
    Example of many-to-many LSTM with variable length input sequence.

    References:
        * https://github.com/dennybritz/tf-rnn/blob/master/loss_masking.py.ipynb
        * https://github.com/dennybritz/tf-rnn/blob/master/dynamic_rnn.ipynb
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()
tf.set_random_seed(10)
np.random.seed(10)

# Dataset Specs
max_time_steps = 16
vocab_size = 99
example_len = [4, 2, 11, 8, 15]

batch_size = len(example_len)

# Network hyper-params
rnn_hidden_size = 32
input_embedding_size = 32
num_layers = 2

# Training params
num_of_train_step = 800
learning_rate = 0.01

""" Input sequence processing """

# shape: [batch, seq_len]
sequence = np.random.randint(1, vocab_size, [batch_size, max_time_steps])
# padding zeros
for batch, length in enumerate(example_len):
    sequence[batch, length:] = 0

# batch x max_seq_len -1
input_seq = sequence[:, :-1]
target_seq = sequence[:, 1:]

print ("Shape of original sequence: {}".format(sequence.shape))
print ("Shape of input sequence: {}".format(input_seq.shape))
print ("Shape of target sequence: {}".format(target_seq.shape))

print ("original seq:\n{}".format(sequence))
print ("input seq:\n{}".format(input_seq))
print ("target seq:\n{}".format(target_seq))

""" naive rnn """

# shape: [batch, seq_len]
inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputs")
targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="targets")
seqlen = tf.placeholder(dtype=tf.int32, shape=[None, ], name="seqlen")

# shape: [batch, vocab_size, embed_dim]
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
embedded_inputs = tf.nn.embedding_lookup(embeddings, inputs)

print ("Shape of embedded inputs : {}".format(embedded_inputs.get_shape()))

cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

with tf.variable_scope("rnn") as scope:
    outputs, last_states = tf.nn.dynamic_rnn(
        cell = cell,
        inputs=embedded_inputs,
        dtype=tf.float32,
        sequence_length=seqlen,
        time_major=False,
        scope=scope
    )

flatten_outputs = tf.reshape(outputs, [-1, cell.output_size])
flatten_targets = tf.reshape(targets, [-1])
print ("Shape of flatten targets: {}".format(flatten_targets.get_shape()))
print ("Shape of flatten output: {}".format(flatten_outputs.get_shape()))

logits = tf.contrib.layers.fully_connected(
    inputs=flatten_outputs,
    num_outputs=vocab_size)

probs = tf.nn.softmax(logits)
predicted_targets = tf.argmax(probs, axis=1)
# reshape back to [batch, max_seq_len-1]
predicted_targets = tf.reshape(predicted_targets, shape=[-1, max_time_steps-1])

mask = tf.sign(tf.to_float(flatten_targets))
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=flatten_targets)
masked_losses = mask * losses
masked_losses = tf.reshape(masked_losses,  tf.shape(targets))

# Calculate mean loss
mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / example_len
mean_loss = tf.reduce_mean(mean_loss_by_example)

## optimization and update
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
updates = optimizer.minimize(mean_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    fetches = [updates, mean_loss, masked_losses, predicted_targets]

    feedins = {
        inputs: input_seq,
        targets: target_seq,
        seqlen: example_len
    }

    for i in range(num_of_train_step):
        _, loss, maskloss, predicted = sess.run(fetches, feedins)
        
        if i % 100 == 0:
            print ("Iter {}: total loss = {}".format(i, loss))
    
    # trimming output
    for batch, length in enumerate(example_len):
        predicted[batch, length:] = 0

    print ("Masked losses:\n{}".format(maskloss))
    print ("Target:\n{}".format(target_seq))
    print ("Predicted:\n{}".format(predicted))
