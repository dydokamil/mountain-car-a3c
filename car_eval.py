import gym
import numpy as np
import tensorflow as tf

from A3CNetwork import A3CNetwork
from common import a_size, s_size, action_low, action_high

env = gym.make("MountainCarContinuous-v0")

tf.reset_default_graph()
model = A3CNetwork(s_size, a_size, 'global', None)

saver = tf.train.Saver()
model_dir = './model'

with tf.Session() as sess:
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, latest_checkpoint)

    while True:
        s = env.reset()
        rnn_state = model.state_init
        d = False
        total_reward = 0
        while not d:
            a_dist, rnn_state = sess.run(
                [model.normal_dist,
                 model.state_out],
                feed_dict={model.inputs: [s],
                           model.state_in[0]: rnn_state[0],
                           model.state_in[1]: rnn_state[1]})
            a = np.squeeze(a_dist)
            a = np.clip(a, a_min=action_low,
                        a_max=action_high)
            s, r, d, _ = env.step(a)
            env.render()

            total_reward += r
        print("Reward:", total_reward)
