"""

"""
from unicodedata import name
import torch
from torch import Tensor
import constants
# import graphvisualisation

class World: 
    def __init__(self, n_concepts) -> None:
        self.n_quadrants = 8
        self.n_circles = 20
        self.num_vertices = n_concepts
        torch.manual_seed(10)
        self.locations = (constants.max_DIMENSIONALITY - constants.min_DIMENSIONALITY)*torch.rand(n_concepts, 2) + constants.min_DIMENSIONALITY
        # creating the radiuses of the concentric circles
        self.radiuses = torch.linspace(0, 100, steps=self.n_circles)

    @staticmethod
    def quadrant_circle_pair(pairs, source):
        co1 = pairs[0] - source[0]
        co2 = pairs[1] - source[1]
        point1, point2 = source, pairs
        length = torch.sqrt(torch.square(point1[0]-point2[0])+torch.square(point1[1]-point2[1]))
        angle = torch.rad2deg(torch.acos((point2[0]-point1[0])/length))
        if point1[1]>point2[1]:
            angle = 360 - angle
        
    
        octant = 0
        segment = 0
        if angle>=0 and angle<=45:
            octant = 1
        elif angle>45 and angle<=90:
            octant = 2
        elif angle>90 and angle<=135:
            octant = 3
        elif angle>135 and angle<=180:
            octant = 4
        elif angle>180 and angle<=225:
            octant = 5
        elif angle>225 and angle<=270:
            octant = 6
        elif angle>270 and angle<=315:
            octant = 7
        elif angle>315:
            octant = 8
        # finding the circle
        # c_x,c_y =source[0], source[1] #coordinates of the origin
        distance = torch.sqrt(torch.square(co1) + torch.square(co2))
        radiuses = torch.load('data/radiuses.pt')
        for i, s in enumerate(radiuses):
            if distance<=s.item():
                segment = i
                break

        return octant,segment+101
    


if __name__ == '__main__':
    world = World(10)
    print(world.locations.numpy())

    