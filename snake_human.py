import pygame
import random
from enum import Enum
from collections import namedtuple

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
SPEED = 5

class SnakeGame:

    def __init__(self, w = 640, h = 440) -> None:
        self.w = w
        self.h = h
        
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

        # initial game state
        self.direction = Direction.RIGHT

        # snake head
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (BLOCK_SIZE * 2), self.head.y)]
        
        self.score = 0
        self.food = None
        self.place_food()
    
    def place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()

    def is_colliding(self):
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y < 0 or self.head.y > self.h - BLOCK_SIZE:
            return True
        
        if self.head in self.snake[1:]:
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

    def move(self, direction):
        x = self.head.x
        y = self.head.y
        
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
 
    def play_step(self):
        
        # user input:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP

        # move to new point with give direction
        self.move(self.direction)
        self.snake.insert(0, self.head)
        
        # is game over:
        game_over = False
        if self.is_colliding():
            game_over = True
            return game_over, self.score
        
        # snake eats the food
        if self.head == self.food:
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()
        
        # update UI with the moves
        self.update_ui()
        self.clock.tick(SPEED)    

        # return status and score
        return game_over, self.score


if __name__ == "__main__":
    game = SnakeGame()

    while True:
        game_over, score = game.play_step()

        if game_over:
            break

    print('Final score: ', score)

    pygame.quit()


