import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import time

pygame.init()
font = pygame.font.SysFont("arial", 24)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple("Point", "x, y")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
RED = (200, 0, 0)

BLOCK_SIZE = 20
SPEED = 80

class SnakeGameAI:

    def __init__(self, w = 640, h = 480) -> None:
        self.w = w
        self.h = h
        self.time = time.time()
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # initialize game state
        self.direction = Direction.RIGHT

        # snake head
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (BLOCK_SIZE * 2), self.head.y)]
        
        self.score = 0
        self.food = None
        self.place_food()
        self.iterations = 0

    
    def place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        self.food = Point(x, y)

        if self.food in self.snake:
            self.place_food()


    def play_step(self, action):
        self.iterations += 1
        # user input:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()     

        # move to new point with give direction
        self.move(action)
        self.snake.insert(0, self.head)

        # is game over:
        game_over = False
        reward = 0
        if self.is_collision() or self.iterations > len(self.snake) * 100:
            reward = -10
            game_over = True
            return reward, game_over, self.score
        
        # snake eats the food
        if self.head == self.food:
            reward = 10
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()
        
        # update UI with the moves
        self.update_ui()
        self.clock.tick(SPEED)    

        # return o/p: status and score
        return reward, game_over, self.score



    def is_collision(self, point = None):
        if point is None:
            point = self.head

        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y < 0 or point.y > self.h - BLOCK_SIZE:
            return True
        
        if point in self.snake[1:]:
            return True
        
        return False

    def update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.circle(self.display, BLUE1, (pt.x + BLOCK_SIZE / 2, pt.y + BLOCK_SIZE / 2), BLOCK_SIZE/2)
            pygame.draw.circle(self.display, BLUE2, (pt.x + BLOCK_SIZE / 2, pt.y + BLOCK_SIZE / 2), BLOCK_SIZE/4)

        pygame.draw.circle(self.display, RED, (self.food.x + BLOCK_SIZE / 2, self.food.y + BLOCK_SIZE / 2), BLOCK_SIZE/2)

        text = font.render("Score:: "+ str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def move(self, action):

        # straight, right , left
        direction_flow = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = direction_flow.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = direction_flow[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_direction = direction_flow[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_direction = direction_flow[next_idx]

        self.direction = new_direction
        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
 
    