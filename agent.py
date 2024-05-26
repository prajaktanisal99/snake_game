import torch
import random
import numpy as np
from snake_ai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from collections import deque
from model import Linear_QNet, QTrainer
import matplotlib.pyplot as plt
from IPython import display


plt.ion()

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    
    def __init__(self) -> None:
        self.num_game = 0
        self.epsilon = 0 # control randomness
        self.discount_rate = 0.9 #gamma
        self.memory = deque(maxlen=MAX_MEMORY) # if exceeds given limit, removes element from left : popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, LEARNING_RATE, self.discount_rate)
        

    def get_state(self, game):
        head = game.snake[0]

        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        point_u = Point(head.x, head.y - BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # state:
        # [danger_s, danger_r, danger_l, d_l, d_r, d_u, d_d, f_l, f_r, f_u, f_d]

        state = [
            # danger_s : straight
            (dir_l and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_d)) or 
            (dir_u and game.is_collision(point_u)), 

            # danger_r : right
            (dir_u and game.is_collision(point_r)) or 
            (dir_r and game.is_collision(point_d)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)),

            # danger_l : left
            (dir_l and game.is_collision(point_d)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)), 

            dir_l,
            dir_r, 
            dir_u,
            dir_d,

            # food: f_l, f_r, f_u, f_d
            game.food.x < game.head.x, # left
            game.food.x > game.head.x, # right
            game.food.y < game.head.y, # up
            game.food.y > game.head.y, #down
        ]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, status):
        self.memory.append((state, action, reward, next_state, status))

    def train_on_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, next_states, statuses = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, statuses)

    def train_on_short_memory(self, state, action, reward, next_state, status):
        self.trainer.train_step(state, action, reward, next_state, status)

    def get_action(self, state):
        # exploration (random moves) vs exploitation (after learning)
        self.epsilon = 80 - self.num_game
        
        # random move logic
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            first_state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(first_state)
            move = torch.argmax(prediction).item()
            action[move] = 1
        
        return action

def train_agent():
    plot_scores = []
    plot_avg_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state Qi
        old_state = agent.get_state(game)
        # get appropriate action for given state
        move = agent.get_action(old_state)

        # perform move and get new state
        reward, status, score = game.play_step(move)
        new_state = agent.get_state(game)
        
        # train short memory
        agent.train_on_short_memory(old_state, move, reward, new_state, status)

        # remember
        agent.remember(old_state, move, reward, new_state, status)

        if status:
            # train long memory : experience replay; trains on all previous moves, plot results
            game.reset()
            agent.num_game += 1
            agent.train_on_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
            
            print(f"Game: {agent.num_game} {score} {record}")
            plot_scores.append(score)
            total_score += score
            avg_score = total_score / agent.num_game
            plot_avg_scores.append(avg_score)
            plot(plot_scores, plot_avg_scores)
        

def plot(scores, avg_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("No. of Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(avg_scores)
    plt.ylim(ymin = 0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(avg_scores) - 1, avg_scores[-1], str(avg_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


if __name__ == "__main__":
    train_agent()