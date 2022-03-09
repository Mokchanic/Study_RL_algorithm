import gym
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Common.ReplayBuffer import ReplayBuffer

#parameter
LEARNING_RATE = 5E-4
GAMMA = 0.98
BATCH_SIZE = 32
MAX_EPI = int(3e4)
PRINT_INTERVAL = 20
BUFFER_SIZE = 32

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class Qnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, observation: torch.Tensor):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, state: torch.Tensor, epsilon):
        out = self.forward(state)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def typeChanger(state, action, reward, next_state, done_mask):
    state = state.clone().to(device)
    action = torch.tensor(action, dtype=torch.float32).to(device)
    reward = torch.tensor(reward, dtype=torch.float32).to(device)
    next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
    done_mask = torch.tensor(done_mask, dtype=torch.float32).to(device)
    return state, action, reward, next_state, done_mask


def train(q, q_target, buffer, optimizer):
    for i in range(20):
        state, action, reward, next_state, done_mask = buffer.sample(BUFFER_SIZE)

        q_out = q(state)
        action = torch.tensor(action, dtype=torch.int64)
        q_a = q_out.gather(1,action)
        max_q_prime = q_target(next_state).max(1)[0].unsqueeze(1)
        target = reward + GAMMA + max_q_prime * done_mask

        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    dirPath = './Train_data/DQN.pt'
    env = gym.make('CartPole-v1')
    q = Qnet(state_dim=4, action_dim=2)
    # q_target = Qnet(state_dim=4, action_dim=2)
    # q_target.load_state_dict(q.state_dict())
    check_point = torch.load(dirPath)
    q.load_state_dict(check_point['q_state'])
    buffer = ReplayBuffer(4, 2)

    score = 0.0
    n_step = 0
    optimizer = optim.Adam(q.parameters(), lr=LEARNING_RATE)

    for n_epi in range(MAX_EPI):
        env.render()
        epsilon = max(0.01, 0.1 - 0.01 * (n_epi/200))
        state = env.reset() # state는 카트의 위치, 카트의 속도, 막대의 각도, 막대의 각속도
        state = torch.tensor(state, dtype=torch.float32).to(device)
        done = False

        while not done:
            n_step += 1
            action = q.sample_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            done_mask = 0.0 if done else 1.0

            state, action, reward, next_state, done_mask = typeChanger(state, action, reward, next_state, done_mask)
            buffer.add(state, action, reward/100.0, next_state, done_mask)

            state = next_state
            score += reward

            if done:
                break

        # if n_step > 5000: # ==buffer_size
        #     train(q, q_target, buffer, optimizer)

        if n_epi % PRINT_INTERVAL == 0 and n_epi != 0:
            # q_target.load_state_dict(q.state_dict())
            print("n_epi :{}, score :{:.1f}, n_step :{}, eps :{:.1f}%"
                  .format(n_epi, score/PRINT_INTERVAL, n_step, epsilon * 100))
            score = 0.0

    env.close()

if __name__=='__main__':
    main()

