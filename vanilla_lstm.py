'''
    this code is refered to a great tutorial
    https://github.com/keveman/tensorflow-tutorial/blob/master/PTB%20Word%20Language%20Modeling.ipynb
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

UNROLLS = 4

sess = tf.Session()

class LSTMCell(object):
    def __init__ (self, state_size):
        self.state_size = state_size
        self.W_f = tf.Variable(self.initializer())
        self.W_i = tf.Variable(self.initializer())
        self.W_o = tf.Variable(self.initializer())
        self.W_C = tf.Variable(self.initializer())
        self.b_f = tf.Variable(tf.zeros([state_size]))
        self.b_i = tf.Variable(tf.zeros([state_size]))
        self.b_o = tf.Variable(tf.zeros([state_size]))
        self.b_C = tf.Variable(tf.zeros([state_size]))
    def __call__(self, x_t, h_t1, C_t1):
        X = tf.concat([h_t1, x_t], 1)
        f_t = tf.sigmoid(tf.matmul(X, self.W_f) + self.b_f)
        i_t = tf.sigmoid(tf.matmul(X, self.W_i) + self.b_i)
        o_t = tf.sigmoid(tf.matmul(X, self.W_o) + self.b_o)
        Ctilde_t = tf.tanh(tf.matmul(X, self.W_C) + self.b_C)
        C_t = f_t * C_t1 + i_t * Ctilde_t
        h_t = o_t * tf.tanh(C_t)
        return h_t, C_t
        
    def initializer(self):
        return tf.random_uniform([2*self.state_size, self.state_size], -0.1, 0.1)

# reading data
words = open('simple-examples/data/ptb.train.txt').read().replace('\n', '<eos>').split()
words_as_set = set(words)
print('Number of words %d' % len(words_as_set))

word_to_id = {w: i for i, w in enumerate(words_as_set)}
id_to_word = {i: w for i, w in enumerate(words_as_set)}
data = [word_to_id[w] for w in words]

num_of_data = len(data)
num_of_words = len(words_as_set)

# declaring embedding vectors (figure out)
state_size = 128
# [10000x128] 
embedding_params = tf.Variable(tf.random_uniform([num_of_words, state_size], -0.02, 0.02))

lstm = []
for _ in range(UNROLLS):
    lstm.append(LSTMCell(state_size))

sm_w = tf.Variable(tf.random_uniform([state_size, num_of_words], -0.1, 0.1))
sm_b = tf.Variable(tf.random_uniform([num_of_words]))

# for [batch_size, num_steps]
words_ph = tf.placeholder(tf.int64, name='words')
targets_ph = tf.placeholder(tf.int64, name='targets')

def forward(batch_size, num_steps):
    output = [tf.zeros([batch_size, state_size])] * UNROLLS
    state = [tf.zeros([batch_size, state_size])] * UNROLLS
    preds = []
    cost = 0.0
    for i in range(num_steps):
        # Get embedding for words
        embedding = tf.nn.embedding_lookup(embedding_params, words_ph[:, i])
        output[0], state[0] = lstm[0](embedding, output[0], state[0])
        for d in range(1, UNROLLS):
            output[d], state[d] = lstm[d](output[d-1], output[d], state[d])
        logits = tf.matmul(output[-1], sm_w)+ sm_b
        preds.append(tf.nn.softmax(logits))
        # loss per step
        cost = cost + tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets_ph[:, i]))
    cost = cost / np.float32(num_steps)
    return preds, cost

def get_one_example(num_steps):
    offset = np.random.randint(num_of_data-num_steps-1)
    # return sentence and target
    return (data[offset:offset+num_steps],
            data[offset+1:offset+1+num_steps])

def get_mini_batch(batch_size, num_steps):
    words, targets = [], []
    for _ in range(batch_size):
        w, t = get_one_example(num_steps)
        words.append(w)
        targets.append(t)
    return words, targets

# print 4 examples
w, t = get_one_example(4)
print([id_to_word[x] for x in w], [id_to_word[x] for x in t])
print ('indices of word in V')
print (w)

ws, ts = get_mini_batch(2, 4)
for i in range(2):
    print([id_to_word[x] for x in ws[i]], [id_to_word[x] for x in ts[i]])

preds, cost = forward(1, 8)
sess.run(tf.global_variables_initializer())
w, t = get_mini_batch(1, 8)
p = sess.run(preds[0], feed_dict={words_ph: w, targets_ph: t})

np.set_printoptions(formatter={'float': lambda x: '%.04f'%x}, threshold=10000)
print(p[0][:100])

global_step = tf.Variable(0, trainable=False)

def train(learning_rate, batch_size, num_steps):
    _, cost_value = forward(batch_size, num_steps)
    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    grads = tf.gradients(cost_value, all_vars)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    lr = tf.train.exponential_decay(learning_rate, global_step, 1000, 0.8)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(grads_and_vars=zip(grads, all_vars),
                                            global_step=global_step)
    return cost_value, train_op

batch_size = 32
num_timesteps = 16
cost_value, train_op = train(1.0, batch_size, num_timesteps)
sess.run(tf.global_variables_initializer())

for step_number in range(100):
    w, t = get_mini_batch(batch_size, num_timesteps)
    c, _ = sess.run([cost_value, train_op], feed_dict={words_ph: w, targets_ph: t})
    if step_number % 10 == 0:
        print('step %d: %.3f' % (step_number, c))