from torch import nn as nn
from torch.nn import functional as F


class BombNet(nn.Module):
    def __init__(self, number_of_actions):
        super(BombNet, self).__init__()
        
        self.number_of_actions = number_of_actions
        
        self.conv1 = nn.Conv2d(4, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(1024, 170)
        self.fc2 = nn.Linear(170, self.number_of_actions)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.size()[0], -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class Dueling_BombNet(nn.Module):
    def __init__(self, number_of_actions):
        super(Dueling_BombNet, self).__init__()

        self.number_of_actions = number_of_actions
    
        self.conv1 = nn.Conv2d(4, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        self.fc_adv = nn.Linear(1024, 170)
        self.fc_val = nn.Linear(1024, 170)

        self.fc_adv2 = nn.Linear(170, self.number_of_actions)
        self.fc_val2 = nn.Linear(170, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        # x = F.relu(self.fc_1(x))
        # out = self.fc_2(x)

        adv = F.relu(self.fc_adv(x))
        val = F.relu(self.fc_val(x))

        adv = self.fc_adv2(adv)
        val = self.fc_val2(val).expand(x.size(0), self.number_of_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.number_of_actions)

        # print("done")
        return x
