import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module): 
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # feed-forward neural network
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, tensor_x):
        # activation function
        tensor_x = F.relu(self.linear1(tensor_x))
        tensor_x = self.linear2(tensor_x)
        return tensor_x
    
    def save(self, file = "mode.pth"):
        folder = "./model"
        if not os.path.exists(folder):
            os.makedirs(folder)

        file = os.path.join(folder, file)
        torch.save(self.state_dict(), file)

class QTrainer:

    def __init__(self, model, learning_rate, discount_rate) -> None:
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.model = model

        # optimiser
        self.optimiser = optim.Adam(model.parameters(), lr = self.learning_rate)
        
        # loss function
        self.loss_function = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, status):
        state = torch.tensor(state, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        action = torch.tensor(action, dtype = torch.long)
        reward = torch.tensor(reward, dtype = torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            action = torch.unsqueeze(action, 0)
            status = (status, )
        
        # Bellman Equation
        # Get predicted Q value with current state
        print("state:", state)
        prediction = self.model(state)
        print(prediction)

        
        # Q(i+1) : reward + discount_rate * max(next predicted Q value)
        target = prediction.clone()
        for i in range(len(status)):
            Q_new = reward[i]
            if not status[i]:
                Q_new = reward[i] + self.discount_rate * torch.max(self.model(next_state[i]))
                print('Q_new', Q_new)

            target[i][torch.argmax(action[i]).item()] = Q_new
            print("target:", target)
        print("target.shape",target.shape)

        # reset gradient
        self.optimiser.zero_grad()
        loss = self.loss_function(target, prediction)
        loss.backward()

        self.optimiser.step()