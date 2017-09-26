from __future__ import division

import sys, os
import numpy as np

import itertools
# Use named tuple for replay buffer
import collections

import tensorflow as tf
from policy_grad import PolicyEstimator, ValueEstimator
import gym

env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

action_dim = env.action_space.n
obs_dim = env.observation_space.shape[0]

tf.app.flags.DEFINE_bool('save_graph', False, 'Option to save graph to logs')
tf.app.flags.DEFINE_integer('num_episodes', 100, 'Number of training episodes')

FLAGS = tf.app.flags.FLAGS


def reinforce(env, sess, policy_func, value_func, num_episodes, discount=1.0):
    print ('REINFORCE Algorithm')

    Transition = collections.namedtuple("Transition",
                ["state", "action", "reward", "next_state", "done"])

    for epi in range(num_episodes):
        state = env.reset()

        episode = []
        episode_reward = 0.0
        
        # Sample Data
        for t in itertools.count():
            state_vec = np.expand_dims(state, axis=0)
            action_probs = policy_func.predict(sess, state_vec)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            episode.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

            episode_reward += reward
            
            if done:
                break

            state = next_state

        # Policy & Value Updates
        for t, transition in enumerate(episode):
            # compact form of computing \sigma(\gamma*R_t) := Rt + g*Rt+1 + g^2*Rt+2 + ... 
            total_return = sum(discount**i * t.reward for i, t in enumerate(episode[t:]))

            # Stupid way to stack input...
            # for improvement, we can use hstack or vstack to create a batch of data
            state_vec = np.expand_dims(transition.state, axis=0)
            action = np.expand_dims(transition.action, axis=0)
            total_return = np.expand_dims(total_return, axis=0)
            
            value_func.update(sess, state_vec, total_return)

            baseline = value_func.predict(sess, state_vec)

            # A = Q - V
            advantage = total_return - baseline

            policy_func.update(sess, state_vec, advantage, action)
        
        print ('Episode {}: Total Reward {}'.format(epi, episode_reward))


with tf.Session() as sess:
    with tf.variable_scope('REINFORCE'):
        policy_estimator = PolicyEstimator(action_dim=action_dim, obs_dim=obs_dim)
        policy_estimator.build()

        value_estimator = ValueEstimator(action_dim=action_dim, obs_dim=obs_dim)
        value_estimator.build()

        if FLAGS.save_graph:
            # Save the graph for debug
            tf.summary.FileWriter('logs', sess.graph)

        # initialize weights
        sess.run(tf.global_variables_initializer())
        # Run.
        reinforce(env, sess, policy_estimator, value_estimator, FLAGS.num_episodes, discount=1.0)
