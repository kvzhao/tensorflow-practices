from __future__ import print_function
from __future__ import division

import sys, os
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

'''Batch policy gradient algorithm
    In this file, I want to implement the most basic policy grad. methods
'''


# include the path of environment
if "../../" is not sys.path:
    sys.path.append("../../")

from rlenv.envs.cliff_walking import CliffWalkingEnv
from rlenv import plotting

tf.app.flags.DEFINE_string('log_dir', 'log_batch_policy_gradient', 'Path to the logfiles')
tf.app.flags.DEFINE_bool('save_graph', True, 'Option of saving computational graph.')

tf.app.flags.DEFINE_integer('max_episodes', 500, 'Maximum episodes')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
tf.app.flags.DEFINE_float('discount', 1.0, 'Discount value')

FLAGS = tf.app.flags.FLAGS

if not tf.gfile.IsDirectory(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)
    graph_dir = os.path.join(FLAGS.log_dir, 'graph')
    if not tf.gfile.IsDirectory(graph_dir) and FLAGS.save_graph:
        tf.gfile.MakeDirs(graph_dir)

class PolicyEstimator(object):
    def __init__(self, observ_dim, action_dim, learning_rate):
        print ('Define a Policy Estimator')
        self.observ_dim = observ_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        with tf.variable_scope('from_MDP'):
            self.state = tf.placeholder(tf.int32, [], 'state')
            self.action = tf.placeholder(tf.int32, name='action')
            self.reward = tf.placeholder(tf.float32, name='reward')

        print ('Building network...')
        with tf.variable_scope('policy_net'):
            state_one_hot = tf.one_hot(self.state, self.observ_dim)

            self.output = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=self.action_dim,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            # why squeeze?
            self.action_probs = tf.squeeze(tf.nn.softmax(self.output))
            # what is this?
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

        with tf.variable_scope('objective'):
            # trick: weighted likelihood with rewards
            self.loss = - tf.log(self.picked_action_prob) * self.reward
            tf.summary.scalar('loss', self.loss)
        
        with tf.variable_scope('solver'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

        print ('Print the variables in network')
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print ('\t{}\t{}'.format(var.name, var.get_shape()))
        
        self.summary_op = tf.summary.merge_all()
    
    def updates(self, sess, state, reward, action):
        feeds = {
            self.state: state,
            self.reward: reward,
            self.action: action
        }
        _, loss = sess.run([self.train_op, self.loss], feeds)
        return loss

    def predict(self, sess, state):
        return sess.run(self.action_probs, {self.state: state})


def training_algorithm(sess, env, model, max_episodes, discount=1.0):
    
    Transition = collections.namedtuple("Transition",
        ["state", "action", "reward", "next_state", "done"])

    # Keeps track of useful statistics
    statistics = plotting.EpisodeStats(
        episode_lengths=np.zeros(max_episodes),
        episode_rewards=np.zeros(max_episodes))  
    
    for episode_i in range(max_episodes):

        state = env.reset()
        done = False

        episode = []
        episode_reward = 0.0

        while not done:
            # Phase I: collecting data with policy
            pi = model.predict(sess, state)
            # greedy?
            action = np.random.choice(np.arange(len(pi)), p=pi)
            next_state, reward, done, _ = env.step(action)

            episode.append(Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            ))

            state = next_state
            episode_reward += reward

            # Update statistics
            statistics.episode_rewards[episode_i] += reward
            statistics.episode_lengths[episode_i] += 1
        
        print ('episode {}: Total Reward = {}'.format(episode_i, episode_reward))

        # Phase II: Fit the model 
        # Phase III: Improve the policy

        for t, tranistion in enumerate(episode):
            reward_to_go = np.sum(discount ** i * trans.reward for i, trans in enumerate(episode[t:] ))

            model.updates(sess, tranistion.state, reward_to_go, tranistion.action)
    
    return statistics


def main():
    
    env = CliffWalkingEnv()
    print ('Env: {}'.format(env.__class__.__name__))
    print ('Observation space size: {}'.format(env.observation_space.n))
    print ('Action space size: {}'.format(env.action_space.n))
    obersv_dim = env.observation_space.n 
    action_dim = env.action_space.n

    with tf.Session() as sess:
        global_step = tf.Variable(0, name="global_step", trainable=False)

        policy = PolicyEstimator(obersv_dim, action_dim, FLAGS.learning_rate)

        if FLAGS.save_graph:
            graph_dir = os.path.join(FLAGS.log_dir, 'graph')
            writer = tf.summary.FileWriter(graph_dir, sess.graph)

        # initailize the parameters
        sess.run(tf.global_variables_initializer())

        # running the reinforcement learning algorithm
        stats = training_algorithm(sess, env, policy, FLAGS.max_episodes, FLAGS.discount)

        f1, f2, f3 = plotting.plot_episode_stats(stats, smoothing_window=10)
        f2.savefig(os.path.join(FLAGS.log_dir, 'reward.png'))


if __name__ == '__main__':
    main()