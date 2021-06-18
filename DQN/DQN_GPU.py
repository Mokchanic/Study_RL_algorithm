import os
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

# Replay Buffer
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)
            
        return torch.tensor(s_lst, dtype=torch. float).to(device),\
            torch.tensor(a_lst).to(device), torch.tensor(r_lst).to(device),\
            torch.tensor(s_prime_lst, dtype = torch. float).to(device),\
            torch.tensor(done_mask_lst).to(device)

    def size(self):
        return len(self.buffer)

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

# train
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime). max (1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# main
def main():
    env = gym.make('CartPole-v1')
    q = Qnet().to(device)
    q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    dirPath = os.path.dirname(os.path.realpath(__file__))
    dirPath = dirPath.replace('/home/blue/reinforcement_algorithm/DQN', '/home/blue/reinforcement_algorithm/Train_data/DQN')
    

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr = learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))

        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s). float ().to(device), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r/100.0, s_prime, done_mask))
            s = s_prime
            score += r
            if done:
                break

        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval ==0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%"\
            .format (n_epi, score/print_interval, memory.size(), epsilon*100))

            torch.save(q, dirPath + 'DQN.pt')
            torch.save(q.state_dict(), dirPath + 'DQN_state_dict.pt')
            torch.save({
                'DQN': q.state_dict(),
                'optimizer': optimizer.state_dict()
            }, dirPath + 'all.tar')

            score = 0.0

    
    env.close()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main()