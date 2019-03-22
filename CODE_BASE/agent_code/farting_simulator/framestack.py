import numpy as np
import torch

from agent_code.clean_bomb.params import params


class FrameStack():
    """
    This class implements a fixed-size stack of size k for frames and manipulates them
        ________________                       ___________________      _________________
        |               |_                     |                  |    |                 |
        |               | |                    |                  |... |                 |
        |   k Frames    | |      getStack()    |       k concatenated frames             |
        |               | |     |---------->   |                  |... |                 |
        |               | |                    |                  |    |                 |
        |               | |                    |                  |    |                 |
        ----------------  |                     ------------------      -----------------
          ----------------

    The k concatenated frames later represent a state of the game
    """
    def __init__(self, maxsize=4, frame0=None):
        """
        :param maxsize: The maximum number of elements in the stack
        :param frame0: An initial frame to fill the entire stack
        """
        self.maxsize = maxsize
        self.stack = []

        if frame0 is not None:
            self.init(frame0)

    def __preprocess_frame(self, frame):
        """
        We preprocess the given frame, transpose it and turn it into a torch.Tensor
        :param frame: the frame we wish to preprocess
        :return:
        """
        array = frame.T
        array = np.reshape(array, (array.shape[0], array.shape[1], 1))
        array = array.transpose(2, 0, 1)
        array = array.astype(np.float32)
        array = torch.Tensor(array)

        if params["CUDA"]:
            array = array.cuda()

        return array

    def add(self, frame):
        """
        Reads a numpy.array frame, transforms it into a torch.Tensor and adds it to the stack.
        Depending whether or not CUDA is enabled, it is pushed to the GPU
        :param frame:
        :return:
        """
        array = self.__preprocess_frame(frame)

        if len(self.stack) == self.maxsize:
            self.stack.pop()
        self.stack.insert(0, array)

    def __len__(self):
        return len(self.stack)

    def reset(self):
        self.stack = []

    def init(self, frame0):
        for i in range(0, 4):
            self.add(frame0)

    def getStack(self, additional_frame = None):
        """
        We concatenate the previous frames stored in the stack and concatenate them to obtain the current
        state of the arena.
        :param additional_frame: If an additional frame is passed, it is concatenated with the last k-1 elements
        of the stack, so the oldest element is ignored.
        :return:
        """
        # TODO: variable size
        if additional_frame is None:
            return torch.cat((self.stack[0], self.stack[1], self.stack[2], self.stack[3])).unsqueeze(0)

        else:
            processed_frame = self.__preprocess_frame(additional_frame)
            return torch.cat((processed_frame, self.stack[0], self.stack[1], self.stack[2])).unsqueeze(0)