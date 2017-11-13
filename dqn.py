# -*- coding: utf-8 -*-
# https://keon.io/deep-q-learning/
import random
import gym
import numpy as np
import time
import sys
import xxhash

import curses
import numpy as np
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint
from snake import Snake

from collections import deque
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, Lambda
from keras.optimizers import Adam

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

import tensorflow as tf
import hashlib

import logging

EPISODES = 1000

# Deep Q Network:
# https://github.com/keon/deep-q-learning
# And the blog
# https://keon.io/deep-q-learning/

# https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html
# https://www.intelnervana.com/demystifying-deep-reinforcement-learning/

def dueling_shape(x):
    shape = list(x)
    return tuple([None, 4])

def dueling_layer(x):
    v_duel = x[:, 0]
    A_duel = x[:, 1:]
    return v_duel + (A_duel - K.max(A_duel))

class DQNAgent:
    """

    Blog: https://keon.io/deep-q-learning/
    Github page: https://github.com/keon/deep-q-learning

    """
    def __init__(self, state_size, action_size):
        """Init class
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.buffer = deque(maxlen=10000)

    def _build_model(self):
        """Build model
        """
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        """Store move
        """
        # We want to store ~relevant data
        # In the game we have lots of moves, where we just move
        # to a new location, without any extra infromation,
        # so we need a way to filter some data.
        val = (state, action, reward, next_state, done)
        h = str(val).encode()
        hash_ = xxhash.xxh32(h).intdigest()

        # We have collided with out self, so we want to remember these
        if done:
            self.memory.append(val)
            self.buffer.append(hash_)
            return

        # Filter using hash
        if hash_ not in self.buffer:
            self.memory.append(val)
            self.buffer.append(hash_)
            return

        # We'll add some data sometimes
        if np.random.uniform() > 0.9:
            self.memory.append(val)
            self.buffer.append(hash_)

    def act(self, state):
        """Predict, what we should do
        """
        if np.random.uniform() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        """Fit to data
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # DQN
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model = load_model(name + '.h5')
        self.epsilon = 0.01

    def save(self, name):
        try:
            self.model.save(name + '.h5', overwrite=True)
        except OSError:
            pass

def write_state(state, reshape=None):
    map_ = ""
    if reshape is not None:
        state = np.reshape(state, reshape)
    for each in state:
        line = ""
        for sym in each:
            if sym == 0.2:
                line += " #"
            elif sym == 0:
                line += " ."
            elif sym == 0.5:
                line += " o"
            elif sym == 2.0:
                line += " O"
            elif sym == 1:
                line += " $"
        print(line)
    print(" ")

if __name__ == '__main__':
    from snake import Snake

    make_images = True
    if make_images:
        import matplotlib.pyplot as plt

    args = sys.argv

    if args[1].lower() == 'play':
        PLAY = True
    elif args[1].lower() == 'train':
        PLAY = False
    else:
        print("Could not get the command. Please supply if we train or play, f.ex:")
        print(">> python dqn.py play")
        sys.exit()

    actions = [KEY_DOWN, KEY_UP, KEY_LEFT, KEY_RIGHT]

    snake = Snake(render=False)
    state_size = snake.get_input_size()
    action_size = len(actions)

    # Initialize Model
    agent = DQNAgent(state_size, action_size)

    # Patch size for fitting model
    batch_size = 32

    # Number of times the game is played
    EPISODES = int(1e6)
    if PLAY:
        agent.load("snake_model")
        EPISODES = 1

    n_steps = 3000
    moving_average = [0]

    all_time_high = 0
    for e in range(EPISODES):
        if PLAY:
            print("--------------------------------")
        done = False
        state, original = snake.reset()
        time_i = 0
        for time_i in range(n_steps):
            action = agent.act(state)
            if PLAY:
                if make_images:
                    fig = plt.figure(frameon=False)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.matshow(original)
                    fig.savefig('./images/snake_{}.png'.format(str(time_i).zfill(3)), bbox_inches='tight')
                else:
                    write_state(original)
                    time.sleep(0.05)
            if done:
                break
            snake.step(actions[action])
            reward, done = snake.play()
            next_state, original = snake.get_state()
            reward = reward if not done else -1

            # Record the step if we're training
            if not PLAY:
                agent.remember(state, action, reward, next_state, done)

            state = next_state

        # Add snake length
        moving_average.append(len(snake.snake))
        if len(moving_average) > 10:
            moving_average.pop(0)

        mean_ = np.mean(moving_average)

        if not PLAY:
            print("episode: {0}/{1}, played steps: {2}, e: {3:.2}, snake length: {4}, snake length moving average: {5}"
                        .format(e, EPISODES, str(time_i).rjust(3), agent.epsilon, len(snake.snake), mean_))
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if e % 100 == 0:
                agent.save("snake_model")
    if not PLAY:
        agent.save("snake_model")
