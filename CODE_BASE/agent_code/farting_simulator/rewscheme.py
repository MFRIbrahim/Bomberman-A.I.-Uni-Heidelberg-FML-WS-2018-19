from agent_code.clean_bomb.params import GRAYSCALE_VALUES
from settings import events

"""

            HIER KOMMEN UNSERE REWARD IDEEN HIN DIE WIR IMPLEMENTIEREN WOLLEN


Reward schemes:
        MOVEMENT: -.1                           covered
        MOVING BACK TO A ALREADY
            VISITED FIELD IN FRAMESTACK:-.15
        WALKING BY A COIN:
        INVALID_ACTION: -1                      covered
        BOMB_DROPPED : 0.1                      covered

        CRATE_DESTROYED: 0.5
        COIN_FOUND: 0.9
        COIN_COLLECTED: 1                       covered

        KILLED_OPPONENT: 1,                     covered
        KILLED_SELF: -2                         covered

        GOT_KILLED: -2,                         covered
        OPPONENT_ELIMINATED': 0                 covered
        SURVIVED_ROUND': 1                      covered
"""

# mapping events to reward values
EVT_REW_MAPPING = {
    'MOVED_LEFT': 0,
    'MOVED_RIGHT': 0,
    'MOVED_UP': 0,
    'MOVED_DOWN': 0,
    'WAITED': 0,
    'INTERRUPTED': 0,
    'INVALID_ACTION': 0,

    'BOMB_DROPPED': 0,
    'BOMB_EXPLODED': 0,

    'CRATE_DESTROYED': 0.5,
    'COIN_FOUND': 0,
    'COIN_COLLECTED': 1,

    'KILLED_OPPONENT': 2,
    'KILLED_SELF': -1,

    'GOT_KILLED': -1,
    'OPPONENT_ELIMINATED': 0,
    'SURVIVED_ROUND': 0,
}



class RewardScheme():
    def __init__(self, agent):
        self.agent = agent

    def getReward(self):
        reward = 0
        for evt in self.agent.events:
            reward += EVT_REW_MAPPING[events[evt]]
        stack = self.agent.frame_stack.getStack()
        indices = (stack == GRAYSCALE_VALUES["self"]).nonzero()

        #print(stack, )

        return reward