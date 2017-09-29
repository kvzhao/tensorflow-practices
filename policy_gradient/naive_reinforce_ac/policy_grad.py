from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

class PolicyEstimator(object):
    '''Single layer Policy Gradient
    '''

    def __init__(self, action_dim, obs_dim):
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def build(self):
        print ('Start building Policy Gradient Method')
        with tf.variable_scope('PolicyEstimator'):
            self.init_placeholder()
            self.init_network()
            self.init_optimizer()
        print ('Done.')
        
    def init_placeholder(self):
        print ('\tinit intput placeholder')
        with tf.variable_scope('inputs'):
            self.state = tf.placeholder(tf.float32, [None, self.obs_dim], name='state')
            self.action = tf.placeholder(tf.int32, [None, ], name='action')
            self.reward = tf.placeholder(tf.float32, [None, ], name='reward')
    
    def init_network(self):
        # single layer network
        print ('\tinit network')
        with tf.variable_scope('network'):
            '''
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=self.action_dim,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            '''
            self.output_layer = tf.layers.dense(
                inputs=self.state,
                units=self.action_dim,
                activation=None,
                kernel_initializer=tf.zeros_initializer())
            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.selected_action_prob = tf.gather(self.action_probs, self.action)

        with tf.variable_scope('loss'):
            self.loss = -tf.log(self.selected_action_prob) * self.reward

    def init_optimizer(self):
        print ('\tinit optimizer')
        with tf.variable_scope('solver'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            # remeber this method
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())

    def predict(self, sess, state):
        return sess.run(self.action_probs, feed_dict={self.state: state})

    def update(self, sess, state, reward, action):
        feeds = {self.state: state,
                self.reward: reward,
                self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feeds)
        return loss


class ValueEstimator(object):
    def __init__(self, action_dim, obs_dim):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
    
    def build(self):
        print ('Start building Value Estimator')
        with tf.variable_scope('ValueEstimator'):
            self.init_placeholder()
            self.init_network()
            self.init_optimizer()
        print ('Done.')
    
    def init_placeholder(self):
        with tf.variable_scope('inputs'):
            self.state = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name='state')
            self.target = tf.placeholder(tf.float32, shape=[None], name='target')
    
    def init_network(self):
        with tf.variable_scope('network'):
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())
            
            self.value_estimate = tf.squeeze(self.output_layer)

        with tf.variable_scope('loss'):
            self.loss = tf.squared_difference(self.value_estimate, self.target)
    
    def init_optimizer(self):
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())
    
    def predict(self, sess, state):
        return sess.run(self.value_estimate, feed_dict={self.state: state})
    
    def update(self, sess, state, target):
        feeds = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feeds)
        return loss
