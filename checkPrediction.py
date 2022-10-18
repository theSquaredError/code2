import constants
from graph_world import World
 
def checkPrediction(action, env, agent_location, target_location):
    # checking if the action is correct

    # utterance: 0
    #guiding: 2
    #pointing: 1

    # finding the appropriate action for the given agent location
    # and target location

    quad, seg = World.quadrant_circle_pair(target_location, agent_location)

    # checking whether only target is present in that quadrant-segment
    count = 0 
    for l1 in env.locations:
        if l1[0] == agent_location[0] and l1[1] == agent_location[1]:
            continue
        elif l1[0] == target_location[0] and l1[1] == target_location[1]:
            continue
        else:
            quadrant, segment = World.quadrant_circle_pair(l1, agent_location)
            # if the quad and seg same as target
            if quadrant == quad and seg == segment:
                count+=1
     
    if count == 0:
        if seg == 102 or seg == 103:
            correct_action = constants.ACTION_POINT
        else:
            correct_action = constants.ACTION_UTTER
    else:
        correct_action = constants.ACTION_GUIDE

    
    return correct_action
