'''
    this code is refered to a great tutorial
    https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/1-seq2seq.ipynb
'''

from __future__ import division
import numpy as np
import tensorflow as tf

import simple_seq2seq_helpers as helpers

sess = tf.Session()

PAD = 0
EOS = 1

vocab_size = 10
embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = 20

# PLACEHOLDERS
# [max_time, batch_size] v.s. RNN layer [max_time, batch_size, input_embedding_size]
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

# EMBEDDING
# weights of embedding look-up layers, shape = vocab_size x embedding size
embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

# ENCODER
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
encoder_outputs, encoder_final_states = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
    dtype=tf.float32, time_major=True, scope='encoder'
)
#del encoder_outputs

# DECODER
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
decoder_outputs, decoder_final_states = tf.nn.dynamic_rnn(
    decoder_cell, decoder_inputs_embedded,
    dtype=tf.float32, time_major=True, scope='plain_decoder'
)
decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
decoder_prediction = tf.argmax(decoder_logits, 2)

# LOSS FUNCTION & OPTIMIZER
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits
)
loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

## WIRING TEST ##
sess.run(tf.global_variables_initializer())

batch_ = [[6], [3, 4], [9, 8, 7]]
batch_, batch_length_ = helpers.batch(batch_)
print ('batch_encoded:\n' + str(batch_))
print ('batch_encoded_length:\n' + str(batch_length_))

din_, dlen_ = helpers.batch(np.ones(shape=(3, 1), dtype=np.int32),
                            max_sequence_length=4)
print('decoder inputs:\n' + str(din_))

pred_ = sess.run(decoder_prediction, 
            feed_dict={
                encoder_inputs: batch_,
                decoder_inputs: din_,
            })
print('decoder predictions:\n' + str(pred_))

# TRAIN ON RANDOM SEQUENCES
batch_size = 100
batches = helpers.random_sequences(length_from=3, length_to=8,
                                    vocab_lower=2, vocab_upper=10,
                                    batch_size=batch_size)
print('head of the batch:')
for seq in next(batches)[:10]:
    print(seq)

## PREDICTING NEXT NUMBER
def next_feed():
    batch = next(batches)
    encoder_inputs_, _ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
        [(sequence) + [EOS] for sequence in batch]
    )
    decoder_inputs_, _ = helpers.batch(
        [[EOS] + (sequence) for sequence in batch]
    )
    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }


loss_track = []
max_batches = 3001
batches_in_epoch = 1000

for batch in range(max_batches):
    fd = next_feed()
    _, l = sess.run([train_op, loss], feed_dict=fd)
    loss_track.append(l)

    if batch == 0 or batch % batches_in_epoch == 0:
        print('batch {}'.format(batch))
        print('  minibatch loss: {}'.format(sess.run(loss, fd)))
        predict_ = sess.run(decoder_prediction, fd)
        # tranpose used as swapping time-major index with batch-major index
        for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
            print ('  sample {}:'.format(i+1))
            print ('    input   > {}'.format(inp))
            print ('    predict > {}'.format(pred))

            if (i > 2):
                break


#%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(loss_track)
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))
plt.savefig('seq2seq_loss.png')