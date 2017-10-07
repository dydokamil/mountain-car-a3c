import gym
import numpy as np
import tensorflow as tf

from A3CNetwork import A3CNetwork

env = gym.make("CartPole-v1")

tf.reset_default_graph()
model = A3CNetwork(4, 2, 'global', None)

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
        total_steps = 0
        while not d:
            feed_dict = {model.inputs: [s],
                         model.state_in[0]: rnn_state[0],
                         model.state_in[1]: rnn_state[1]}
            a_dist, rnn_state = sess.run([model.policy, model.state_out],
                                         feed_dict=feed_dict)
            a = np.argmax(a_dist)
            s, _, d, _ = env.step(a)
            env.render()

            total_steps += 1
        print("Reward:", total_steps)
