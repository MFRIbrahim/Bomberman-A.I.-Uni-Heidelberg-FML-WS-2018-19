from agent_code.clean_bomb.params import params
import os
import torch
import numpy as np
from settings import s
import random
from torch.autograd import Variable


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

ACTION_MAPPING = {
    0: "LEFT",
    1: "RIGHT",
    2: "UP",
    3: "DOWN",
    4: "BOMB",
    5: "WAIT"
}


class AlphabombAgent():
    def __init__(self, NetworkClass, network_data_path):
        self.number_of_actions = len(ACTION_MAPPING)
        self.epsilon = np.linspace(params["initial_eps"], params["final_eps"], s.n_rounds)

        self.Q_network = NetworkClass(self.number_of_actions)
        self.target_network = NetworkClass(self.number_of_actions)

        try:
            pth = os.path.join(__location__, network_data_path)
            self.Q_network.load_state_dict(torch.load(pth))
            self.Q_network.eval()
            print("successfully loaded", network_data_path)

        except FileNotFoundError:
            print("no brain found-- let's create one")

        if params["CUDA"]:
            self.Q_network = self.Q_network.cuda()
            self.target_network = self.target_network.cuda()

        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), lr=params["learning_rate"])
        self.loss_fn = torch.nn.MSELoss(reduction='sum')


    def update_target_network(self):
        """
        We update the the target-network with the weights of the Q-network.
        This usually happens after a high number of iterations.
        """
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def update_Q_network(self, state_batch, action_batch, reward_batch, state_new_batch, terminal_batch):
        """
        Implementation of the Double Deep Q-Learning algorithm to update the Q-network

        :param state_batch:
        :param action_batch:
        :param reward_batch:
        :param state_new_batch:
        :param terminal_batch:
        :return:
        """

        # First we calculate the action a* = argmax_a' Q_net(s, a')
        output = self.Q_network(state_new_batch).detach()
        _, a_max = output.max(dim=1)

        if params["CUDA"]:
            a_max = a_max.cuda()

        # Now we calculate y = r + gamma * Q_target(s, a*)
        target_output = self.target_network(state_new_batch).detach()
        target_q = target_output.gather(1, a_max.unsqueeze(1))

        if params["CUDA"]:
            target_q = target_q.cuda()

        y = reward_batch + params["gamma"] * (target_q * (1 - terminal_batch))

        output_old = self.Q_network(state_batch)
        q = (output_old * action_batch).sum(dim=1).view(-1, 1)

        loss = self.loss_fn(input=q, target=y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_action(self, output, episode):
        """
        Returns the action we should take according to the neural net
        :param output: The Q-values for the given state, produced by our Q-network
        :param episode: The current episode
        :return: state tensor and the action string we need to make the next step
        """

        # we represent our action by one-hot encoding, by setting the desited
        # action to one. The rest is zero, which means we do not choose that action
        action = torch.zeros(self.number_of_actions)

        if params["CUDA"]:
            action = action.cuda()

        # epsilon greedy policy
        action_index = torch.argmax(output)
        if random.random() <= self.epsilon[episode]:
            action_index = torch.randint(self.number_of_actions, torch.Size([]))

        if params["CUDA"]:
            action_index = action_index.cuda()

        action[action_index] = 1
        action_str = ACTION_MAPPING[int(action_index.cpu().numpy())]

        return action.unsqueeze(0), action_str

    def save_model(self, pth):
        torch.save(self.Q_network.state_dict(), pth)
