import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class GeneticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(GeneticNetwork,self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # send entire network to device
        self.to(self.device)

    def forward(self, observation):
        # to put obs of env(numpy) to state of pytorch.nn(tensor)
        # . then cuda float tensor is different from just float tensor
        state = T.tensor(observation, dtype=T.float).to(self.device)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Here, n_actions = 2. net. outputs mean, diagonal cov of Gaussian in Continuous action space
        x = self.fc3(x)

        # return only score, handle activation later in choose_action()
        return x # in actor net 2 output, mu & sigma.  in critic 1 output, state value

class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=2,
                 layer1_size=64, layer2_size=64, n_output=1):

        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_output
        self.actor = GeneticNetwork(alpha, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions)    # 2 output: mean, sigma
        self.critic = GeneticNetwork(beta, input_dims, layer1_size,
                                    layer2_size, n_actions=1) # estimating value function for the state

    def choose_action(self, observation):
        mu, sigma = self.actor.forward(observation)
        # ensure sigma is a real non-negative
        sigma = T.exp(sigma)
        # we seek to learn mu, sigma s.t. to maximize expected return over time.
        action_probs = T.distributions.Normal(mu, sigma)
        # action sampling function should be z_a = mu + sigma*I, I~N(0,1) for back-propagation. to work
        z_prob = action_probs.sample(sample_shape=T.Size([self.n_outputs]))
        # log_probs to calc. loss
        self.log_probs = action_probs.log_prob(z_prob).to(self.actor.device)
        # continuous action value betw. -1 ~ +1
        action = T.tanh(z_prob)

        return action.item() # change tensor to value bcoz OpenAI gym does not take tensor.

    """Here AC train NN every time step off of a single observation  """
    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value_ = self.critic.forward(new_state) # V_new_state
        critic_value = self.critic.forward(state)      # V_state

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)

        # critic estimates temporal difference error
        delta = reward + self.gamma * critic_value_ * (1-int(done)) - critic_value

        # critic loss is MSE
        critic_loss = delta**2

        # actor loss is weighted sum of policy gradient
        actor_loss = -self.log_probs * delta

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()

import matplotlib.pyplot as plt
import gym
def plotLearning(scores, filename, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)

if __name__ =="__main__":
    agent = Agent(alpha=0.000005, beta=0.00001, input_dims=[2], gamma=0.99,
                  layer1_size=254, layer2_size=256)

    env = gym.make("MountainCarContinuous-v0")
    env.seed(101)
    np.random.seed(101)
    score_history = []
    num_episodes = 500

    for i in range(num_episodes):
        done = False
        score =0
        observation = env.reset()
        while not done:
            # action here should be numpy array. reshape it a single element
            action = np.array(agent.choose_action(observation)).reshape((1,))
            observation_, reward, done, info = env.step(action)
            if reward > 0:
                print(reward)
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward
        score_history.append(score)
        print('episode', i, 'score %.2f' % score)

    filename = 'plots/' + 'moutaincar-continuous.png'
    plotLearning(score_history, filename, window=20)
