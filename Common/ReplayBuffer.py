import torch
import numpy as np

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

#parameter
BUFFER_SIZE = int(5e3)

class ReplayBuffer(object):
    def __init__(self, state_dim: int, action_dim: int, max_size = BUFFER_SIZE):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state      = torch.zeros((max_size, state_dim), device=device)
        self.action     = torch.zeros((max_size, 1), device=device)
        self.reward     = torch.zeros((max_size, 1), device=device)
        self.next_state = torch.zeros((max_size, state_dim), device=device)
        self.done_mask  = torch.zeros((max_size, 1), device=device)

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done_mask: torch.Tensor):
        self.state[self.ptr]      = state
        self.action[self.ptr]     = action
        self.reward[self.ptr]     = reward
        self.next_state[self.ptr] = next_state
        self.done_mask[self.ptr]  = done_mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, batch_size)

        return (
            self.state[index].clone(), self.action[index].clone(),
            self.reward[index].clone(), self.next_state[index].clone(), self.done_mask[index].clone()
        )
