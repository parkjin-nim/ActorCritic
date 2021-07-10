# for MAC os
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    # gives access to parameters of our network, among other things
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork,self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128) # * means unpack a list
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # tensor in GPU and tensor in CPU is different
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device) # Send network to Cuda tensors

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class PolicyGradientAgent():
    def __init__(self,lr, input_dims, gamma=0.99, n_action=4):
        self.gamma = gamma
        self.lr = lr
        # keep track of reward memory
        # keep track of log_prob
        self.reward_memory = []
        self.action_memory = []

        self.policy = PolicyNetwork(self.lr, input_dims, n_action)

    def choose_action(self, observation):
        # numpy to pytorch tensor
        # observation have no batch dimension
        # pytorch expects a batch dimension fed in
        state = T.tensor([observation]).to(self.policy.device)

        probabilities = F.softmax(self.policy.forward(state))
        action_probs  = T.distributions.Categorical(probabilities)
        action        = action_probs.sample()
        log_probs     = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        # action is pytorch tensor.
        # openaigym does not take tensor as input
        return action.item()

    def store_reward(self,reward):
        self.reward_memory.append(reward)

    def learn(self):
        # zero grad. up of our optimizer
        # pytorch specific.
        # pytorch does not keep track of one update and next
        self.policy.optimizer.zero_grad()

        # G_t = R_t+1 + gamma * R_t+2 + gamma**2 * R_t+3
        # G_t = sum from k=0 to k=T {gamma**k * R_t+k+1}
        G = np.zeros_like(self.reward_memory)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        # covert to device specified for our network.
        G = T.tensor(G, dtype=T.float).to(self.policy.device)

        # using framework to get gradient
        # Derv.(Gt * log_prob_policy_a) == Gt * Derv.(log_prob_policy_a)
        loss = 0
        for g, log_prob in zip(G, self.action_memory):
            loss += -g * log_prob
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []

import gym
import matplotlib.pyplot as plt

def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0,i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running avg of prev. 100 scores')
    plt.savefig(figure_file)


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    n_games = 3000
    agent = PolicyGradientAgent(gamma=0.99, lr=0.0005,input_dims=[8],
                                n_action=4)
    fname = 'REINFORCE_'+'lunar_lunar_lr' + str(agent.lr) + '_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_reward(reward)
            observation = observation_
        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ',i, 'score %.2f' % score,
              'avg score %.2f' % avg_score)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(scores,x,figure_file)

    # for i in range(n_games):
    #     obs = env.reset()
    #     score = 0
    #     done=False
    #     while not done:
    #         action = env.action_space.sample()
    #         obs_, reward, done, info = env.step(action)
    #         score += reward
    #         #env.render()
    #     print('episode ',i, 'score %.1f'%score)

