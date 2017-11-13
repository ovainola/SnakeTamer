# -*- coding: utf-8 -*-
# https://keon.io/deep-q-learning/
import random
import gym
import numpy as np
import time
import sys

import curses
import numpy as np
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint

from collections import deque
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import logging

EPISODES = 1000

# Deep Q Network:
# https://github.com/keon/deep-q-learning
# And the blog
# https://keon.io/deep-q-learning/
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model = load_model(name + '.h5')
        self.epsilon = 0.01

    def save(self, name):
        self.model.save(name + '.h5', overwrite=True)

def write_state(state):
    map_ = ""
    for each in state:
        line = ""
        for sym in each:
            if sym == 0.2:
                line += "#"
            elif sym == 0:
                line += "."
            elif sym == 0.5:
                line += "o"
            elif sym == 1:
                line += "$"
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
    board_size = snake.size
    state_size = board_size[0] * board_size[1]
    action_size = len(actions)
    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 32

    if PLAY:
        EPISODES = 1
        agent.load("snake_model")

    for e in range(EPISODES):
        done = False
        state, original = snake.reset()
        state = np.reshape(state, [1, state_size])
        for time_i in range(500):
            if done:
                break
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
                    time.sleep(0.1)
            snake.step(actions[action])
            reward, done = snake.play()
            next_state, original = snake.get_state()
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            if not PLAY:
                agent.remember(state, action, reward, next_state, done)
            state = next_state
        if not PLAY:
            print("episode: {0}/{1}, score: {2}, e: {3:.2}, snake length: {4}"
                        .format(e, EPISODES, time_i, agent.epsilon, len(snake.snake)))
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if e % 100 == 0:
                agent.save("snake_model")
    if not PLAY:
        agent.save("snake_model")
