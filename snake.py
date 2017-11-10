# SNAKES GAME
# Use ARROW KEYS to play, SPACE BAR for pausing/resuming and Esc Key for exiting

import curses
import numpy as np
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint

class Snake(object):

    def __init__(self, render=True):
        self.size = (21, 61)
        self.mid_axis_0 = int(np.floor(self.size[0] / 2))
        self.mid_axis_1 = int(np.floor(self.size[1] / 2))
        self.render = render
        if self.render:
            curses.initscr()
            self.win = curses.newwin(self.size[0], self.size[1], 0, 0)
            self.win.keypad(1)

            curses.noecho()
            curses.curs_set(0)
            self.win.border(0)
            self.win.nodelay(1)

        self.key = KEY_RIGHT
        self.score = 0

        self.snake = [[4, 10], [4, 9], [4, 8]]
        self.food = [10, 20]

        if self.render:
            self.win.addch(self.food[0], self.food[1], '*')
        self.prev_key = None
        self.move_penalty = 0

    def reset(self):
        if self.render:
            self.win = curses.newwin(self.size[0], self.size[1], 0, 0)
            self.win.keypad(1)
            curses.noecho()
            curses.curs_set(0)
            self.win.border(0)
            self.win.nodelay(1)

        self.key = KEY_RIGHT
        self.score = 0

        self.snake = [[4, 10], [4, 9], [4, 8]]
        self.food = [10, 20]

        if self.render:
            self.win.addch(self.food[0], self.food[1], '*')
        self.prev_key = None
        return self.get_state()

    def get_state(self):
        table = np.zeros(self.size)
        for each in self.snake:
            table[each[0], each[1]] = 0.2
        table[self.snake[0][0], self.snake[0][1]] = 0.5
        table[self.food[0], self.food[1]] = 1

        shift_1 = self.mid_axis_0 - self.snake[0][0]
        shift_2 = self.mid_axis_1 - self.snake[0][1]

        shifted_table = np.roll(table, shift_1, axis=0)
        shifted_table = np.roll(shifted_table, shift_2, axis=1)

        return shifted_table, table

    def step(self, key=None):
        # self.move_penalty = 0
        self.prev_key = self.key
        if key is None:
            event = self.win.getch()
        else:
            event = key
        self.key = self.key if event == -1 else event

        if self.key == KEY_LEFT and self.prev_key == KEY_RIGHT:
            self.key = self.prev_key
        elif self.key == KEY_RIGHT and self.prev_key == KEY_LEFT:
            self.key = self.prev_key
        elif self.key == KEY_DOWN and self.prev_key == KEY_UP:
            self.key = self.prev_key
        elif self.key == KEY_UP and self.prev_key == KEY_DOWN:
            self.key = self.prev_key

    def play(self):

        if self.render:
            self.win.border(0)
            self.win.addstr(0, 2, 'Score : ' + str(self.score) + ' ')
            self.win.addstr(0, 27, ' SNAKE ')
            self.win.timeout(int(150 - (len(self.snake)/5 + len(self.snake)/10)%120))

        # Calculates the new coordinates of the head of the snake. NOTE: len(snake) increases.
        # This is taken care of later at [1].
        self.snake.insert(0, [self.snake[0][0] + (self.key == KEY_DOWN and 1) + (self.key == KEY_UP and -1), self.snake[0][1] + (self.key == KEY_LEFT and -1) + (self.key == KEY_RIGHT and 1)])

        # If snake crosses the boundaries, make it enter from the other side
        if self.snake[0][0] == 0: self.snake[0][0] = 18
        if self.snake[0][1] == 0: self.snake[0][1] = 58
        if self.snake[0][0] == 19: self.snake[0][0] = 1
        if self.snake[0][1] == 59: self.snake[0][1] = 1

        # Exit if snake crosses the boundaries (Uncomment to enable)
        #if snake[0][0] == 0 or snake[0][0] == 19 or snake[0][1] == 0 or snake[0][1] == 59: break

        # If snake runs over itself
        if self.snake[0] in self.snake[1:]:
            self.snake.pop()
            return -10, True

        if self.snake[0] == self.food: # When snake eats the food
            if self.render:
                self.win.addch(self.food[0], self.food[1], '#')
            self.food = []
            self.score += 1
            while self.food == []:
                self.food = [randint(1, 18), randint(1, 58)] # Calculating next food's coordinates
                if self.food in self.snake: self.food = []
            if self.render:
                self.win.addch(self.food[0], self.food[1], '*')
            return 10, False
        else:
            last = self.snake.pop() # [1] If it does not eat the food, length decreases
            if self.render:
                self.win.addch(last[0], last[1], ' ')
        if self.render:
            self.win.addch(self.snake[0][0], self.snake[0][1], '#')
        head = self.snake[0]
        neck = self.snake[1]
        dist_1 = np.sqrt( (head[0] - self.food[0])**2 + (head[1] - self.food[1])**2 )
        dist_2 = np.sqrt( (neck[0] - self.food[0])**2 + (neck[1] - self.food[1])**2 )
        if dist_1 < dist_2:
            return 1, False
        else:
            return -1, False

        print("\nScore - " + str(self.score))

if __name__ == '__main__':
    snake = Snake()
    while True:
        snake.step()
        reward, done = snake.play()
        if done:
            break
    curses.endwin()