from asyncio import constants
import torch
import torch.nn as nn
import torch.nn.functional as F

# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import numpy as np
# from copy import deepcopy
from scipy.stats import truncnorm
import os


from communication import Communication
import constants



'''
Both of the agent will learn their own mapping because each agent 
is a separate neural network
'''


'''
Neural network architecture:
    Input Layer: 28 [one_hot encoding]
    Hidden Layer1: 10
    Output Layer : size of vocabulary = 28 [one_hot encoding]
'''
class MapNet(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(MapNet,self).__init__()

        self.L1 = nn.Linear(input_size,10)
        self.L2 = nn.Linear(10,output_size)

    def forward(self, input):
        x = self.L1(input)
        # print(x)
        x = F.relu(x)
        x = self.L2(x)
        # print(x)
        x = F.softmax(x, dim=0)
        return x




def print_loss(losses, learning_rate = 0.01):
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Learning rate %f"%(learning_rate))
    plt.show()

def train_agent(net, X,Y, learning_rate = 0.01, loss_fn = nn.MSELoss(), epochs = 1000):
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    losses = []
    input_count=[]
    for epoch in range(epochs):
        # t = torch.FloatTensor(1).uniform_(-1, 1) #number of concepts = 28
        t2 = truncnorm.rvs(-10, 10, size=1) 
        index = np.random.choice(28)
        # input_count.append(index)
        X_ = X[index] +\
        torch.tensor(t2, dtype=torch.float)
        Y_ = Y[index]
        # print(f"X_ = {X_}")
        # print(f"Y_ = {Y_}")
        pred_y = net(X_)
        # print(f"pred = {pred_y}")
        # print(f"pred sum = {pred_y.sum()}")
        loss = loss_fn(pred_y, Y_)
        losses.append(loss.item())
        
        net.zero_grad()
        loss.backward()
        optimiser.step()
    # print_loss(losses)
    # print()

def one_hot_encoded(data):
    dim = len(data)
    temp = np.eye(dim)
    return temp

def train2(net, X,Y,learning_rate = 0.01, loss_fn = nn.MSELoss()):
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    for i in range(1):
        pred_y = net(X)
        loss = loss_fn(pred_y, Y)
        # print(f"loss = {loss}")
        net.zero_grad()
        loss.backward()
        optimiser.step()


# if __name__ == '__main__':
def initialise(epochs = 10000):
    # Getting the mappings
    vocab_map = Communication.generate_vocabulary(constants.n_octants,constants.n_segments)
    # vocab_map = np.array(vocab_map)
    
    X_ = [i[0] for i in vocab_map]
    Y_ = [i[1] for i in vocab_map]
    
    # Creating one hot encoding of each of the vector
    X = torch.tensor(one_hot_encoded(X_), dtype=torch.float)
    Y = torch.tensor(one_hot_encoded(Y_), dtype=torch.float)
    # t = torch.FloatTensor(28).uniform_(-1, 1)


    # VocabNet : concept -> vocab
    # input size = output size = 8+20  = constants.n_octants+ constants.n_segments
    vocabNet = MapNet(28,28)
    
    # print(vocabNet)

    train_agent(vocabNet,X,Y, epochs=epochs)
    # print(X[2])
    # print(vocabNet(X[2]))
    '''
    for i in range(11):
        pred = vocabNet(X[i])
        print(torch.sum(pred))
        print(pred)
        print(torch.argmax(pred))
        print(torch.argmax(Y[i]))
        print("*"*50)
        # os.system('clear')
    # print(torch.sum(pred, 1))
 
    # print(torch.argmax(Y,1))
    # print(torch.argmax(pred, 1))
    # print(vocabNet(X[:3]))

    # train_agent(vocabNet,)
    '''

    # Vocab to concepts 
    # this will be used by listener agent
    conceptNet = MapNet(28, 28)
    train_agent(conceptNet, X,Y, epochs=epochs)

    return X_, Y_, conceptNet, vocabNet


'''
if __name__ == '__main__':
    # Getting the mappings
    vocab_map = Communication.generate_vocabulary(constants.n_octants,constants.n_segments)
    # vocab_map = np.array(vocab_map)
    
    X_ = [i[0] for i in vocab_map]
    Y_ = [i[1] for i in vocab_map]
    
    # Creating one hot encoding of each of the vector
    X = torch.tensor(one_hot_encoded(X_), dtype=torch.float)
    Y = torch.tensor(one_hot_encoded(Y_), dtype=torch.float)
    # t = torch.FloatTensor(28).uniform_(-1, 1)


    # VocabNet tries to learn the mapping from concept to vocab
    # input size = output size = 8+20  = constants.n_octants+ constants.n_segments
    vocabNet = MapNet(28,28)
    
    # print(vocabNet)

    train_agent(vocabNet,X,Y, epochs=10000)
    # print(vocabNet(X[5]))
    train2(vocabNet, X[5],Y[5])
    # print(vocabNet(X[5]))
    # print(X[2])
    # print(vocabNet(X[2]))
    
    for i in range(11):
        pred = vocabNet(X[i])
        print(torch.sum(pred))
        print(pred)
        print(torch.argmax(pred))
        print(torch.argmax(Y[i]))
        print("*"*50)
        # os.system('clear')
    # print(torch.sum(pred, 1))
 
    # print(torch.argmax(Y,1))
    # print(torch.argmax(pred, 1))
    # print(vocabNet(X[:3]))

    # train_agent(vocabNet,)
    

    # Vocab to concepts 
    # this will be used by listener agent
    # conceptNet = MapNet(28, 28)
    # train_agent(conceptNet, X,Y, epochs=10000)
    # print(X[5].view(1,28))

    
'''