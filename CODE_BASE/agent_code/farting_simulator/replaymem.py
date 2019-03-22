import random
import torch

from agent_code.clean_bomb.params import params


class StdReplayMemory():
    """
    A standard implementation of replay memory used in DQN
    """

    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self._memory = []

    def add(self, transition):
        """
        Stores a transition of the form (s,a,r,s',terminal) in the replay memory
        :param transition:
        :return:
        """
        self._memory.append(transition)

        if len(self._memory) > self.capacity:
            self._memory.pop(0)

    def sample_batch(self):
        """
        Uniformly samples a batch from the replay memory of the form (s,a,r,s')
        :return:
        """

        minibatch = random.sample(self._memory, min(len(self._memory), self.batch_size))

        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_new_batch = torch.cat(tuple(d[3] for d in minibatch))
        terminal_batch = torch.cat(tuple(d[4] for d in minibatch))

        if params["CUDA"]:
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_new_batch = state_new_batch.cuda()
            terminal_batch = terminal_batch.cuda()

        return state_batch, action_batch, reward_batch, state_new_batch, terminal_batch


class PrioritizedExperienceMemory():
    def __init__(self):
        raise NotImplementedError("todo")
