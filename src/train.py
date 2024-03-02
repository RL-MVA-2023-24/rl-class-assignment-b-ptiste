# Standard library imports
from copy import deepcopy
import random

# Third-party library imports
import numpy as np
import torch
import torch.nn as nn

# Local application imports
from env_hiv import HIVPatient
from gymnasium.wrappers import TimeLimit
    
device = "cpu"

from evaluate import evaluate_HIV, evaluate_HIV_population
env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)      
    
class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space, n_hid):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(observation_space, n_hid)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.bn3 = nn.BatchNorm1d(n_hid)
        self.fc4 = nn.Linear(n_hid, action_space)
        self.activation = torch.nn.SiLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x)) + x
        x = self.activation(self.fc3(x)) + x
        x = self.fc4(x)
        return x
    
def greedy_action(network, state):
    network.eval()
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0))
        return torch.argmax(Q).item()
    
class ProjectAgent:
    def __init__(self):
        self.nb_actions = 4
        self.nb_obs = 6
        self.model = QNetwork(self.nb_obs, self.nb_actions, 256)
    
    def act(self, observation, use_random=False):
        return greedy_action(self.model, observation)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self):
        self.model.load_state_dict(torch.load("best_policy.pt"))
        print("Model loaded")
        self.model.eval()
    
class dqn_agent:
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model.to(device) 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0

    def act(self, observation, use_random=False):
        return greedy_action(self.model, observation)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self):
        pass
    
    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_return = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
           
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau * model_state_dict[key] + (1-tau) * target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                
                validation_score = evaluate_HIV(agent=self, nb_episode=1) if episode >= 75 else -1
                
                episode_return.append(episode_cum_reward)
                print("Episode ", '{:2d}'.format(episode), 
                        ", epsilon ", '{:6.2f}'.format(epsilon), 
                        ", batch size ", '{:4d}'.format(len(self.memory)), 
                        ", ep return ", '{:4.2f} M'.format(episode_cum_reward / 1e6), 
                        ", validation score ", '{:4.2f} M'.format(validation_score / 1e6),
                        sep='')

                if validation_score > best_return:
                    print('New best policy found')
                    best_return = validation_score
                    self.save(r"C:\Users\baptc\Documents\Etudes\MVA\S2\RL\rl-class-assignment-b-ptiste\outputs\best_policy.pt")
    
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
    

if __name__ == "__main__":
    
    config = {'nb_actions': 4,
            'observation_space':6, 
          'learning_rate': 0.001,
          'gamma': 0.90,
          'buffer_size': 100000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 20000,
          'epsilon_delay_decay': 500,
          'batch_size': 512,
          'gradient_steps': 3,
          'update_target_strategy': 'ema', # or 'ema'
          'update_target_freq': 400,
          'update_target_tau': 0.001,
          'criterion': torch.nn.SmoothL1Loss(),
          }
    
    agent = dqn_agent(config, QNetwork(6, 4, 256))
 
    returns = agent.train(env, 499)
