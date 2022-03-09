import gym
import torch
from torch.distributions import Categorical

from PPO import PPO
from Common.define import PPO_define

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    env = gym.make("CartPole-v1")
    ppo = PPO()
    define = PPO_define()
    score = 0.0
    print_interval = 20

    for n_epi in range(3000):
        state = env.reset()
        done = False

        while not done:
            for t in range(define.T_horizon):
                state = torch.tensor(state, dtype=torch.float32).to(device)
                prob = ppo.pi(state)
                m = Categorical(prob)
                action = m.sample().item()
                next_state, reward, done, info = env.step(action)

                ppo.put_data((state, action, reward, next_state, prob[action].item(), done))
                state = next_state

                score += reward

                if done:
                    break

            ppo.train()

        if n_epi%print_interval==0 and n_epi!=0:
            print("Epi :{}, Avg_score :{}".format((n_epi, score/print_interval)))
            score = 0.0

    env.close()




if __name__ == '__main__':
    main()