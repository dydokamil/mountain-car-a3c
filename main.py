# coding: utf-8

# Credit: https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb

import os
import threading
from time import sleep

import gym
import numpy as np
import scipy.signal
import tensorflow as tf

from A3CNetwork import A3CNetwork
# ### Helper Functions
from common import s_size, a_size

main_lock = threading.Lock()
action_high = [1]
action_low = [-1]


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
# TODO normalize rewards?
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# ### Worker Agent
class Worker:
    def __init__(self, game, name, s_size, a_size, trainer, model_path,
                 global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(
            "train_" + str(self.number))

        # Create the local copy of the network and the TensorFlow op
        # to copy global parameters to local network
        self.local_AC = A3CNetwork(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = self.actions = np.identity(a_size, dtype=bool).tolist()
        self.env = game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = \
            rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     # self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: self.batch_rnn_state[0],
                     self.local_AC.state_in[1]: self.batch_rnn_state[1]}
        # with tf.device('/gpu:0'):
        v_l, l_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run(
            [self.local_AC.value_loss,
             self.local_AC.loss,
             self.local_AC.entropy,
             self.local_AC.grad_norms,
             self.local_AC.var_norms,
             self.local_AC.state_out,
             self.local_AC.apply_grads],
            feed_dict=feed_dict)
        print("Loss:", np.mean(l_l))
        # print("Trained.")
        # return (v_l / len(rollout), p_l / len(rollout),
        #         e_l / len(rollout), g_n, v_n)

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0

                with main_lock:
                    s = self.env.reset()
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                d = False
                while not d:
                    # Take an action using probabilities
                    # from policy network output.
                    a_dist, v, rnn_state = sess.run(
                        [self.local_AC.normal_dist, self.local_AC.value,
                         self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs: [s],
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})

                    a = a_dist.squeeze()
                    a = np.clip(a, a_min=action_low,
                                a_max=action_high)

                    with main_lock:
                        s1, r, d, _ = self.env.step(a)
                        if self.name == 'worker_0':
                            self.env.render()

                    r /= 100.

                    if d:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended,
                    # but the experience buffer is full, then we make an update
                    # step using that experience rollout.
                    # TODO remove batch training and value bootstraping
                    # if len(episode_buffer) == 30 \
                    #         and not d \
                    #         and episode_step_count != max_episode_length - 1:
                    #     # Since we don't know what the true final return is,
                    #     # we "bootstrap" from our current
                    #     # value estimation.
                    #     feed_dict = {self.local_AC.inputs: [s],
                    #                  self.local_AC.state_in[0]: rnn_state[0],
                    #                  self.local_AC.state_in[1]: rnn_state[1]}
                    #     v1 = sess.run(self.local_AC.value,
                    #                   feed_dict=feed_dict)[0, 0]
                    #     v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer,
                    #                                          sess, gamma, v1)
                    #     episode_buffer = []
                    #     sess.run(self.update_local_ops)
                    # if d:
                    #     break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer
                # at the end of the episode.
                if len(episode_buffer) != 0:
                    self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(
                            episode_count) + '.ckpt')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    if mean_reward >= MAX_REWARD * .99:
                        saver.save(sess, self.model_path + '/model-' + str(
                            episode_count) + '.ckpt')
                        print("Saved Model")

                        coord.request_stop()  # STOP
                    print("Mean reward:", mean_reward)
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1


if __name__ == '__main__':
    max_episode_length = 300
    gamma = .99  # discount rate for advantage estimation and reward
    model_path = './model'
    MAX_REWARD = 1e5

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32,
                                      name='global_episodes',
                                      trainable=False)
        trainer = tf.train.AdamOptimizer()
        # Generate global network
        master_network = A3CNetwork(s_size, a_size, 'global', None)
        # Set workers ot number of available CPU threads
        # num_workers = multiprocessing.cpu_count()
        num_workers = 4
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(
                Worker(gym.make("MountainCarContinuous-v0"), i, s_size, a_size, trainer,
                       model_path, global_episodes))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        worker_threads = []


        def worker_work(w): return w.work(max_episode_length,
                                          gamma,
                                          sess, coord, saver)


        for worker in workers:
            def worker_work(): worker.work(max_episode_length, gamma,
                                           sess, coord, saver)


            t = threading.Thread(target=worker_work)
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
