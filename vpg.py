from timeit import timeit
from turtle import circle
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

from torch import nn
from torch import optim
from torch import Tensor

import agent_network
from environment import Environment
from graph_world import World
import utils
import time

class PolicyEstimator(nn.Module):
    def __init__(self, env):
        super(PolicyEstimator, self).__init__()
        self.num_observations = env.observation_space['target'].shape[0] + env.observation_space['curr_loc'].shape[0]
        # print(f"num_observation={self.num_observations}")
        self.num_actions = env.action_space.n

        self.network = nn.Sequential(
            nn.Linear(self.num_observations, 16),
            nn.Sigmoid(),
            nn.Linear(16, self.num_actions),
            nn.Softmax(dim=-1)
        )
    def forward(self, observation):
        return self.network(observation)
    def predict(self, observation):
        return self.forward(observation)

def vanilla_policy_gradient(env, estimator, num_episodes=1, batch_size=10, discount_factor=0.01, render=False,early_exit_reward_amount=None, device=None):

    if device!=None:
        print(device)
        estimator.to(device)
    estimator.cuda()
    total_rewards, batch_rewards, batch_observations, batch_actions = [], [], [], []
    batch_counter = 1
    epsilon = 0.9
    ct=0
    optimizer = optim.Adam(estimator.network.parameters(), lr=0.01)
    action_space = np.arange(env.action_space.n) 
    losses = []
    for current_episode in range(num_episodes):
        print(f"current_episode: {current_episode}")
        observation = env.reset()
        rewards, actions, observations = [], [], []

        while True:
            if render:
                env.render()
            print(f"observation = {observation}")
            # use policy to make predictions and run an action
            t = torch.cat((observation['cur_loc'], observation['target']))
            if device!=None:
                print(device)
                t.to(device)
            
            print(f"action_probs: {estimator.predict(t.cuda())}")
            action_probs = estimator.predict(t.cuda()).cpu().detach().numpy()
            
            # epsilon greedy technique 
            if np.random.rand() < epsilon and ct<2000:
                action = np.random.choice(action_space)    
                ct+=1

            else:
                action = np.random.choice(action_space, p=action_probs) # randomly select an action weighted by its probability


            print(f"action={action}")
            # push all episodic data, move to next observation
            observations.append(t.tolist())
            observation, reward, done = env.step(action)
            rewards.append(reward)
            actions.append(action)
            print(reward)

            if done:
                # apply discount to rewards
                r = np.full(len(rewards), discount_factor) ** np.arange(len(rewards)) * np.array(rewards)
                r = r[::-1].cumsum()[::-1]
                discounted_rewards = r - r.mean()

                # collect the per-batch rewards, observations, actions
                batch_rewards.extend(discounted_rewards)
                batch_observations.extend(observations)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                if batch_counter >= batch_size:
                    # reset gradient
                    optimizer.zero_grad()

                    # tensorify things
                    batch_rewards = torch.FloatTensor(batch_rewards)
                    # print(batch_observations)
                    batch_observations = torch.FloatTensor(batch_observations)
                    batch_actions = torch.LongTensor(batch_actions)

                    # calculate loss
                    if device!=None:
                        batch_observations.to(device)
                    logprob = torch.log(estimator.predict(batch_observations.cuda()))
                    # print(f"logprob = {logprob}")
                    # print(f"batch_reward = {batch_rewards}")
                    for param in estimator.network.parameters():
                        print(param.data)
                    batch_actions = batch_actions.reshape(len(batch_actions), 1).cuda()
                    selected_logprobs = batch_rewards.cuda() * torch.gather(logprob, 1, batch_actions).squeeze()
                    # print(f"selected_logprob = {selected_logprobs}")
                    loss = -selected_logprobs.mean()
                    print(loss.item()*1000)
                    losses.append(loss.item()*1000)
                    # backprop/optimize
                    loss.backward()
                    optimizer.step()

                    # reset the batch
                    batch_rewards, batch_observations, batch_actions = [], [], []
                    batch_counter = 1

                # get running average of last 100 rewards, print every 100 episodes
                average_reward = np.mean(total_rewards[-1000:])
                # if current_episode % 1000 == 0:
                    # print(f"average of last 1000 rewards as of episode {current_episode}: {average_reward:.2f}")

                # quit early if average_reward is high enough
                if early_exit_reward_amount and average_reward > early_exit_reward_amount:
                    return total_rewards

                break

    return total_rewards,losses


if __name__ == '__main__':
    st = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create environment
    X_,Y_, vocabNet, conceptNet = agent_network.initialise(epochs = 10000, device=device)
    env = Environment(vocabNet=vocabNet,conceptNet=conceptNet,X_=X_, Y_=Y_)

    # creating graph for all the vertices as agent location 
    # and storing locations in  file
    utils.saveGraph(env.locations, env.radiuses)


    # actually run the algorithm
    policy = PolicyEstimator(env)
    rewards,losses = vanilla_policy_gradient(env, policy, num_episodes=10000, device=device)

    # moving average
    moving_average_num = 100
    def moving_average(x, n=moving_average_num):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    # plotting

    '''
    plt.scatter(np.arange(len(rewards)), rewards, label='individual episodes')
    plt.plot(moving_average(rewards), label=f'moving average of last {moving_average_num} episodes')
    plt.title(f'Vanilla Policy Gradient')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

    '''
    
    
    # print(losses)
    plt.plot(losses)
    plt.ylabel('loss')
    plt.savefig('data/loss_plot.png')
    plt.close()

    # take random target and agent location
    for l1 in env.locations:
        for l2 in env.locations:
            if l1[0] ==l2[0] and l1[1] == l2[1]:
                continue
            else:
                agent_loc = l1
                target_loc = l2
                t = torch.cat((agent_loc, target_loc))
                action_probs = policy.predict(t.cuda())
                print(f"agent_loc: {agent_loc} target_loc: {target_loc}")
                octant,segment = World.quadrant_circle_pair(target_loc,agent_loc)
                print(f"octant={octant}, segment={segment}")

                print(action_probs)
                print("\n")
                action = np.random.choice(np.arange(env.action_space.n), p=action_probs.detach().numpy())
                #print(action)

    # graphplot.plot_graph(env.locations)

    et = time.time()
    res = et - st
    final_res = res / 60
    print('Execution time:', final_res, 'minutes')

