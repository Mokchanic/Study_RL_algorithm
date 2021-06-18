import gym
import collections
import random

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

# Q_value Network
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else:
            return out.argmax().item()
# main
def main():
    env = gym.make('CartPole-v1')

    dirPath = '/home/blue/reinforcement_alrorithm/Train_data/DQN/DQN_GPU.pt' 

    q = Qnet().to(device)
    checkpoint = torch.load(dirPath)
    q.load_state_dict(checkpoint)


    q.eval()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr = learning_rate)

    for n_epi in range(10000):
        env.render()

        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))

        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s). float ().to(device), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            #memory.put((s, a, r/100.0, s_prime, done_mask))
            s = s_prime
            score += r


            if done:                
                break
            
        # print(score)
        # score = 0.0

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main()