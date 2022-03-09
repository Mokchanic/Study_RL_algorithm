import torch
import torch.nn as nn
import torch.nn.functional as F

from Common.define import PPO_define

device = 'cuda:0' if torch.cuda.is_available() else "cpu"

def mlp(input_dim, mlp_dims, last_relu=False):
    nn_layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        nn_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            nn_layers.append(nn.ReLU6())
    net = nn.Sequential(*nn_layers)
    return net

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.replay_buffer=[]
        self.define = PPO_define()
        self.pi_nn = mlp(self.define.INPUT, self.define.MLP_DIMS_PI)
        self.v_nn  = mlp(self.define.INPUT, self.define.MLP_DIMS_V)

    def pi(self, state, softmax_dim = 0):
        pi_nn = self.pi_nn(state)
        prob = F.softmax(pi_nn, dim = softmax_dim)
        return prob

    def v (self, state, softmax_dim = 0):
        v_nn = self.v_nn(state)
        v = F.softmax(v_nn, dim = softmax_dim)
        return v

    def put_data(self, transition):
        self.replay_buffer.append(transition)

    def make_batch(self):
        state_batch = torch.stack([s for (s, a, r, ns, d, prob) in self.replay_buffer])
        action_batch = torch.Tensor([a for (s, a, r, ns, d, prob) in self.replay_buffer])
        reward_batch = torch.Tensor([r for (s, a, r, ns, d, prob) in self.replay_buffer])
        next_state_batch = torch.stack([ns for (s, a, r, ns, d, prob) in self.replay_buffer])
        done_batch = torch.Tensor([prob for (s, a, r, ns, d, prob) in self.replay_buffer])
        prob_batch = torch.Tensor([d for (s, a, r, ns, d, prob) in self.replay_buffer])

        print("state: ", state_batch)
        print("action: ", action_batch)

        self.replay_buffer.clear()
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, prob_batch

    def train(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_mask_batch, prob_batch\
            = self.make_batch()

        for i in range(self.define.K_epoch):
            td_target = reward_batch + self.define.gamma * self.v(next_state_batch) * done_mask_batch
            delta = td_target - self.v(state_batch)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.define.gamma * self.define.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])

            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float32)

            pi   = self.pi(state_batch, softmax_dim=1)
            pi_a = pi.gather(1, action_batch)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_batch))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.define.eps_clip, 1+self.define.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(state_batch) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()






