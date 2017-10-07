import numpy as np
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')


class A3CNetwork:
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, s_size],
                                         dtype=tf.float32, name='inputs')
            img_size = int(np.sqrt(s_size))
            self.imageIn = tf.reshape(self.inputs, shape=[-1,
                                                          img_size,
                                                          img_size,
                                                          1])

            # convolutional layers
            # TODO try without elu
            W_conv1 = weight_variable([3, 3, 1, 64])
            b_conv1 = weight_variable([64])
            h_conv1 = conv2d(self.imageIn, W_conv1) + b_conv1

            W_conv2 = weight_variable([3, 3, 64, 128])
            b_conv2 = weight_variable([128])
            h_conv2 = conv2d(h_conv1, W_conv2) + b_conv2

            W_fc1 = weight_variable([23 * 23 * 128, 200])
            b_fc1 = weight_variable([200])

            h_conv2_flat = tf.reshape(h_conv2, [-1, 23 * 23 * 128])
            h_fc1 = tf.nn.elu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(128, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c],
                                  name='c_in')
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h],
                                  name='h_in')
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(h_fc1, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in,
                sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 128])

            # W_policy = tf.Variable(tf.random_normal([128, a_size]))
            # b_policy = tf.Variable(tf.random_normal([a_size]))
            # self.policy = tf.matmul(rnn_out, W_policy) + b_policy

            W_mu = tf.Variable(tf.truncated_normal([128, a_size]))
            b_mu = tf.Variable(tf.truncated_normal([a_size]))
            self.mu = tf.matmul(rnn_out, W_mu) + b_mu

            W_sigma = tf.Variable(tf.truncated_normal([128, a_size]))
            b_sigma = tf.Variable(tf.truncated_normal([a_size]))
            self.sigma = tf.nn.softplus(
                tf.matmul(rnn_out, W_sigma) + b_sigma)

            self.normal_dist = tf.random_normal(shape=[a_size],
                                                mean=self.mu,
                                                stddev=self.sigma)

            W_value = tf.Variable(tf.random_normal([128, 1]))
            self.value = tf.matmul(rnn_out, W_value)

            # Only the worker network need ops for loss functions
            # and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size,
                                                 dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],
                                                 dtype=tf.float32)

                # self.responsible_outputs = tf.reduce_sum(
                #     self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(
                    tf.square(self.target_v - tf.reshape(self.value, [-1])))

                # self.entropy = - tf.reduce_sum(
                #     self.policy * tf.log(self.policy))

                self.log_prob = (tf.log(tf.pow(tf.sqrt(2.
                                                       * self.sigma
                                                       * np.pi), -1))
                                 - (self.normal_dist - self.mu)
                                 * (self.normal_dist - self.mu)
                                 * tf.pow((2. * self.sigma), -1))

                self.entropy = -.5 * (tf.log(2. * np.pi * self.sigma) + 1.)

                # self.policy_loss = -tf.reduce_sum(
                #     tf.log(self.responsible_outputs) * self.advantages)

                self.advantages_tiled = tf.tile(self.advantages, [a_size])
                self.advantages_tiled = tf.reshape(self.advantages_tiled, [-1, a_size])
                self.policy_loss = -self.log_prob * self.advantages_tiled

                self.loss = (0.5
                             * self.value_loss
                             + self.policy_loss
                             - self.entropy
                             * 0.01)

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = \
                    tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(
                    zip(grads, global_vars))
