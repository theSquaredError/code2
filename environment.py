# from turtle import done
import constants
from graph_world import World
import agent_network

import torch
import numpy as np
import gym
from gym import spaces

time = 1

class Environment(gym.Env):
    def __init__(self, vocabNet, conceptNet,X_,Y_):
        self.X_,self.Y_ = X_,Y_
        self.vocabNet = vocabNet
        self.conceptNet = conceptNet
        self.n_quadrants = 8
        self.n_circles = 20
        self.num_vertices = 10
        self.locations = (constants.max_DIMENSIONALITY - constants.min_DIMENSIONALITY)*torch.rand(self.num_vertices, 2) + constants.min_DIMENSIONALITY
        
        self.location_quad_map = {}
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict(
            {
                "curr_loc": spaces.Box(-10, 10, shape=(2,), dtype=float),
                "target": spaces.Box(-10, 10, shape=(2,), dtype=float),
            }
        )
        
        # self.observation_space = spaces.Box(shape=(2,2))
    def _get_obs(self):
        return {"cur_loc": self._agent_location, "target": self._target_location}
    
    def step(self, action):
        if action == constants.ACTION_GUIDE:
            dist = np.linalg.norm(self._agent_location-self._target_location)
            reward = 10
        if action == constants.ACTION_POINT:
            pointed_segment = 21
            while pointed_segment >=21:
                pointed_segment = np.random.poisson(lam=1.5)

            pointed_segment+=101
            # choosing nearby segment
            octant,segment = World.quadrant_circle_pair(self._target_location,self._agent_location)
            if pointed_segment >103:
                reward =  2

            elif pointed_segment == 101:
                # checking if only 2 locations are in that segment
                count = 0
                for location in self.locations:
                    if location[0] == self._agent_location[0] and location[1] == self._agent_location[1]:
                        continue
                    elif location[0] == self._target_location[0] and location[1] == self._target_location[1]:
                        continue
                    else:
                        quad,seg = World.quadrant_circle_pair(location,self._agent_location)
                        if quad == octant and segment == seg:
                            count +=1
                
                if count >=1:
                    reward = 2
                else:
                    reward = 10
            elif  pointed_segment == 102:
                count = 0
                for location in self.locations:
                    if location[0] == self._agent_location[0] and location[1] == self._agent_location[1]:
                        continue
                    elif location[0] == self._target_location[0] and location[1] == self._target_location[1]:
                        continue
                    else:
                        quad,seg = World.quadrant_circle_pair(location,self._agent_location)
                        if quad == octant and segment == seg+1:
                            count +=1
                
                if count >=1:
                    reward = 2
                else:
                    reward = 10
            
            else:
                reward = 2

        if action == constants.ACTION_UTTER:
            # now we perform the utterance and listener has to interpret
            # or a single neural network taking two inputs as one
            
            octant, segment = World.quadrant_circle_pair(self._target_location, self._agent_location)
            # finding vocab for octant and segment
            input_oct = torch.zeros(28)
            input_oct[self.X_.index(octant)] = 1

            input_seg = torch.zeros(28)
            # print(f"X_={self.X_}")
            # print(f"segment={segment}")
            input_seg[self.X_.index(segment)] = 1
            

            # This has to be done 4 times
            for i in range(4):
                oct_vocab_probs,seg_vocab_probs = self.vocabNet(input_oct), self.vocabNet(input_seg)

                # above 2 contains probabilities for each word 
                # we will sample the vocab and send to listener

                # sampling vocabs
                oct_utter = np.random.choice(
                    self.Y_, p=oct_vocab_probs.detach().numpy())
                seg_utter = np.random.choice(
                    self.Y_, p=seg_vocab_probs.detach().numpy())
                
                vocab_oct = torch.zeros(28)
                vocab_oct[self.Y_.index(oct_utter)] = 1

                vocab_seg = torch.zeros(28)
                vocab_seg[self.Y_.index(seg_utter)] =1

                # print(f"oct_utter={oct_utter}")
                ####### Listener ############
                oct_probs,seg_probs = self.conceptNet(vocab_oct), self.conceptNet(vocab_seg)

                pred_oct = np.random.choice(self.X_, p = oct_probs.detach().numpy())
                pred_seg = np.random.choice(self.X_, p = seg_probs.detach().numpy())

                if pred_oct == octant and pred_seg == segment:
                    # Listener understood correctly
                    reward = 15  # some higher reward
                    #Tweaking the vocabNet and conceptNet
                    agent_network.train2(self.vocabNet,input_oct,vocab_oct)
                    agent_network.train2(self.vocabNet,input_seg,vocab_seg)

                    agent_network.train2(self.conceptNet,vocab_oct,input_oct)
                    agent_network.train2(self.conceptNet,vocab_seg,input_seg)

                    break
                else:
                    reward = -5
        global time
        if time%10 == 0: 
            done = True
        else:
            done = False
        time+=1
        observation = self.reset()
        return observation, reward, done
        

    def reset(self):
        ini_index = np.random.choice(10)
        self._agent_location = self.locations[ini_index]
        # choosing target other the agent location
        nonsrc_indices = np.where(self.locations != self._agent_location)[0]
        target = np.random.choice(nonsrc_indices)
        self._target_location = self.locations[target]
        
        observation = self._get_obs()

        return observation




def main():
    X_,Y_, vocabNet, conceptNet = agent_network.initialise(epochs=10)
    env = Environment(vocabNet=vocabNet,conceptNet=conceptNet,X_=X_, Y_=Y_)
    # for _,loc in enumerate(env.locations):
    #     env.location_quad_map[loc] = World.quadrant_circle_pair(loc)

    observation = env.reset()
    # print(observation)
    # print(observation['cur_loc'])
    t = torch.cat((observation['cur_loc'], observation['target']))
    print(t)


if __name__ == '__main__':
    main()