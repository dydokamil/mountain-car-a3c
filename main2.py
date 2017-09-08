import glob
import itertools
import os
import random
from collections import deque

import gym
import numpy as np
from cv2 import cv2
from keras.layers import Convolution2D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

BATCH_SIZE = 64
GAMMA = .99

epsilon = .05
D = deque(maxlen=5000)
render = False
best = -20.
learning_rate = 1.
LEARNING_RATE_DECAY = .999
MIN_LEARNING_RATE = .1

DISCRETIZATION_LEVELS = (3, 2, 2)
as1 = np.linspace(-1, 1, DISCRETIZATION_LEVELS[0])
as2 = np.linspace(0, 1, DISCRETIZATION_LEVELS[1])
as3 = np.linspace(0, 1, DISCRETIZATION_LEVELS[2])
POSSIBLE_ACTIONS = np.asarray([x for x in itertools.product(as1, as2, as3)])
print("Possible actions:", len(POSSIBLE_ACTIONS))


def get_model():
    # saved = glob.glob('*.h5')
    # if len(saved) == 1:
    #     global best
    #     best = float(saved[0].split('_')[-1].split('.h5')[0])
    #     print("Returning a saved model.")
    #     return load_model(saved[0])

    model = Sequential()
    model.add(Convolution2D(64, 8, strides=(4, 4), padding='same',
                            activation='relu',
                            input_shape=(96, 96, 4)))
    model.add(Convolution2D(64, 4, strides=(2, 2), padding='same',
                            activation='relu'))
    model.add(Convolution2D(64, 3, padding='same',
                            activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(len(POSSIBLE_ACTIONS)))

    optimizer = Adam(learning_rate)
    model.compile(loss='mse', optimizer=optimizer)

    return model


def save_exp_replay(s, a, r, s_prime, t):
    D.append([s, a, r, s_prime, t])


def decay_learning_rate():
    global learning_rate
    learning_rate *= LEARNING_RATE_DECAY
    if learning_rate < MIN_LEARNING_RATE:
        learning_rate = MIN_LEARNING_RATE


def get_exp_replay():
    try:
        return random.sample(D, BATCH_SIZE)
    except ValueError:
        return None


def choose_action(s, model):
    s = np.expand_dims(s, axis=0)
    if np.random.rand() < epsilon:
        return random.choice(POSSIBLE_ACTIONS)
    else:
        return POSSIBLE_ACTIONS[np.argmax(model.predict(s))]


def img_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    env = gym.make("CarRacing-v0")
    env.reset()

    model = get_model()
    episode = 0
    i = 0
    while True:
        print("Episode", episode)
        s = env.reset()
        total_reward = 0
        s = img_to_gray(s)
        s = np.stack((s, s, s, s), axis=2)
        terminated = False
        while not terminated:
            if render:
                env.render()
            a = choose_action(s, model)
            s_prime, r, terminated, _ = env.step(a)

            s_prime = img_to_gray(s_prime)
            s_prime = np.expand_dims(s_prime, axis=2)

            s_prime = np.dstack((s[:, :, 1:], s_prime))
            save_exp_replay(s, a, r, s_prime, terminated)
            s = s_prime
            total_reward += r

        print("Total reward:", total_reward, '. Learning rate:', learning_rate)
        i += 1
        render = i % 10 == 0  # render?

        if total_reward >= best:
            best = total_reward

            # delete the previous model
            saved_models = glob.glob('*.h5')
            for m in saved_models:
                os.remove(m)

            # save the model
            print("Saving the model with a score of", best)
            model.save(f'model_{best}.h5')

        episode += 1
        # training
        batch = get_exp_replay()
        if batch is None:
            continue

        X = np.zeros((len(batch), 96, 96, 4))
        y = np.zeros((len(batch), len(POSSIBLE_ACTIONS)))

        for idx, (ss, aa, rr, ss_prime, terminated) in enumerate(batch):
            """
            1. Do a feedforward pass for the current state s to get predicted
                Q-values for all actions.
            2. Do a feedforward pass for the next state s’ and calculate
                maximum overall network outputs max a’ Q(s’, a’).
            3. Set Q-value target for action to r + γmax a’ Q(s’, a’)
                For all other actions, set the Q-value target to the same as
                originally returned from step 1.
            4. Update the weights using backpropagation.
            """
            X[idx] = ss
            Q_sa = model.predict(np.expand_dims(ss, axis=0))
            Q_sa_prime = model.predict(np.expand_dims(ss_prime, axis=0))

            if terminated:
                tt = r
            else:
                tt = r + GAMMA * np.max(Q_sa_prime[0])

            action_index = np.where(POSSIBLE_ACTIONS == aa)[0][0]
            Q_sa[0][action_index] = tt
            y[idx] = Q_sa

        model.train_on_batch(X, y)
        decay_learning_rate()
        print("Trained on a batch.")
