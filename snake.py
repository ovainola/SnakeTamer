"""
Snake game

Base for the snake game got from here: https://gist.github.com/sanchitgangwar/2158089
"""


# https://gist.github.com/sanchitgangwar/2158089
import curses
import numpy as np
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint


KEY_DICT = {KEY_RIGHT: 0, KEY_LEFT: 0.3, KEY_UP: 0.6, KEY_DOWN:1}

class Snake(object):
    """Snake game
    """
    def __init__(self, render=True):
        self.size = (15, 15)
        self.input_size = 5
        self.n_moves = 5
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

        self.initial_conditions()
        if self.render:
            self.win.addch(self.food[0], self.food[1], '*')
        self.prev_key = None
        self.move_penalty = 0

    def get_input_size(self):
        return self.input_size

    def initial_conditions(self):
        self.score = 0
        snake_head_x = np.random.randint(2, self.size[0]-2)
        snake_head_y = np.random.randint(2, self.size[1]-2)
        self.snake = [[snake_head_x, snake_head_y]]
        snake_dir = np.random.choice([KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN])
        if snake_dir == KEY_RIGHT:
            self.key = KEY_LEFT
            body_1 = [snake_head_x, snake_head_y+1]
            body_2 = [snake_head_x, snake_head_y+2]
        elif snake_dir == KEY_LEFT:
            self.key = KEY_RIGHT
            body_1 = [snake_head_x, snake_head_y-1]
            body_2 = [snake_head_x, snake_head_y-2]
        elif snake_dir == KEY_UP:
            self.key = KEY_DOWN
            body_1 = [snake_head_x-1, snake_head_y]
            body_2 = [snake_head_x-2, snake_head_y]
        elif snake_dir == KEY_DOWN:
            self.key = KEY_UP
            body_1 = [snake_head_x+1, snake_head_y]
            body_2 = [snake_head_x+2, snake_head_y]
        self.snake.append(body_1)
        self.snake.append(body_2)
        while True:
            food_x = np.random.randint(1, self.size[0]-2)
            food_y = np.random.randint(1, self.size[1]-2)
            food = [food_x, food_y]
            if food not in self.snake:
                self.food = food
                break

    def reset(self):
        """Reset the board and fruit location
        """
        if self.render:
            self.win = curses.newwin(self.size[0], self.size[1], 0, 0)
            self.win.keypad(1)
            curses.noecho()
            curses.curs_set(0)
            self.win.border(0)
            self.win.nodelay(1)

        self.initial_conditions()
        if self.render:
            self.win.addch(self.food[0], self.food[1], '*')
        self.prev_key = None
        return self.get_state()

    def get_state(self):
        """Get the current game map
        """
        table = np.zeros(self.size)
        print_table = np.zeros(self.size)

        # Add snake to table
        for each in self.snake:
            table[each[0], each[1]] = 0.5
        table[self.snake[0][0], self.snake[0][1]] = 2.0
        table[self.food[0], self.food[1]] = 1

        # Shift table so that head is in the middle
        shift_1 = self.mid_axis_0 - self.snake[0][0]
        shift_2 = self.mid_axis_1 - self.snake[0][1]

        shifted_table = np.roll(table, shift_1, axis=0)
        shifted_table = np.roll(shifted_table, shift_2, axis=1)

        i, j = np.where(shifted_table == 1)

        n1 = i[0] - self.mid_axis_0
        n2 = j[0] - self.mid_axis_1

        shifted_table[i[0], j[0]] = 0

        food_idx = 0
        if abs(n1) < abs(n2):
            if np.sign(n1) == 1:
                food_idx = 0
            else:
                food_idx = 1
        else:
            if np.sign(n2) == 1:
                food_idx = 2
            else:
                food_idx = 3

        msize = self.get_input_size()
        final_input = np.zeros((1, msize))

        # Take surrounding elements around the head
        final_input[0, 0] = shifted_table[self.mid_axis_0 + 1, self.mid_axis_1    ]
        final_input[0, 1] = shifted_table[self.mid_axis_0    , self.mid_axis_1 + 1]
        final_input[0, 2] = shifted_table[self.mid_axis_0 - 1, self.mid_axis_1    ]
        final_input[0, 3] = shifted_table[self.mid_axis_0    , self.mid_axis_1  -1]
        final_input[0, 4] = food_idx

        return final_input, table

    def step(self, key=None):
        """Add direction for the next step
        """
        self.prev_key = self.key
        if key is None:
            event = self.win.getch()
        else:
            event = key
        self.key = self.key if event == -1 else event

        # Add an precaution, so that we don't collide to the previous element
        if self.key == KEY_LEFT and self.prev_key == KEY_RIGHT:
            self.key = self.prev_key
        elif self.key == KEY_RIGHT and self.prev_key == KEY_LEFT:
            self.key = self.prev_key
        elif self.key == KEY_DOWN and self.prev_key == KEY_UP:
            self.key = self.prev_key
        elif self.key == KEY_UP and self.prev_key == KEY_DOWN:
            self.key = self.prev_key

    def play(self):
        """Apply move
        """

        if self.render:
            self.win.border(0)
            self.win.timeout(int(150 - (len(self.snake)/5 + len(self.snake)/10)%120))

        # Calculates the new coordinates of the head of the snake. NOTE: len(snake) increases.
        # This is taken care of later at [1].
        self.snake.insert(0, [self.snake[0][0] + (self.key == KEY_DOWN and 1) + (self.key == KEY_UP and -1),
                              self.snake[0][1] + (self.key == KEY_LEFT and -1) + (self.key == KEY_RIGHT and 1)])

        # If snake crosses the boundaries, make it enter from the other side
        self.snake[0][0] = (self.snake[0][0] + self.size[0]) % (self.size[0])
        self.snake[0][1] = (self.snake[0][1] + self.size[1]) % (self.size[1])


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
                self.food = [randint(1, self.size[0]-1), randint(1, self.size[1]-1)] # Calculating next food's coordinates
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

        shift_x = self.mid_axis_0 - head[0]
        shift_y = self.mid_axis_1 - head[1]

        food_x_shifted = self.food[0] + shift_x
        food_y_shifted = self.food[1] + shift_y

        neck_x_shifted = neck[0] + shift_x
        neck_y_shifted = neck[1] + shift_y

        dist_1 = np.sqrt( (self.mid_axis_0 - food_x_shifted)**2 + (self.mid_axis_1 - food_y_shifted)**2 )
        dist_2 = np.sqrt( (neck_x_shifted - food_x_shifted)**2 + (neck_y_shifted - food_y_shifted)**2 )

        if dist_1 < dist_2:
            return 0.1, False

        return -0.3, False

if __name__ == '__main__':
    snake = Snake()
    while True:
        snake.step()
        reward, done = snake.play()
        if done:
            break
    curses.endwin()