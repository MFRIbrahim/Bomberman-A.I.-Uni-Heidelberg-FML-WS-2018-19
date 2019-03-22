import numpy as np

from agent_code.clean_bomb.framestack import FrameStack
from agent_code.clean_bomb.replaymem import StdReplayMemory
from agent_code.clean_bomb.model import BombNet, Dueling_BombNet
from agent_code.clean_bomb.params import params, GRAYSCALE_VALUES
from agent_code.clean_bomb.abagent import AlphabombAgent
from agent_code.clean_bomb.rewscheme import RewardScheme

from settings import s

import torch
import os
import json

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def setup(self):
    """Called once before a set of games to initialize data structures etc.
    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """

    self.deppmindagent = AlphabombAgent(
        NetworkClass=BombNet,
        network_data_path="brain0.pth",
    )

    self.replay_mem = StdReplayMemory(
        capacity=params["replay_mem_capacity"],
        batch_size=params["replay_mem_batchsize"]
    )

    self.frame_stack = FrameStack(maxsize=params["framestack_capacity"])
    self.reward_scheme = RewardScheme(self)

    # stores the terminal events, i.e.
    # killing oneself, getting killed and surviving the round
    self.terminal_events = {13, 14, 16}

    self.episode = 0
    self.prev_state = None
    self.prev_action = None

    # useful stuff for tracking
    self.loss_list = []
    self.q_max = 0
    self.collected_coins = 0
    self.score_points = 0
    self.collected_rewards = 0
    self.cumulative_steps = 0

    self.loss_list_data = []
    self.q_max_data = []
    self.collected_coins_data = []
    self.score_points_data = []
    self.collected_rewards_data = []

    print("the following parameters are used:")
    print(json.dumps(params, indent=1))
    print("------ initialization successful ------")


def get_current_frame(self):
    """
    Returns the grayscale representaion of the arena. The grayscale mapping used to do so is defined in the
    function itelf. We will later concatenate this frame and the previuos 3 frames to generate a state of the game.
    :param self: persistent agent object
    :return:
    """
    # We introduce the following mapping for all possible grid values

    arena = self.game_state["arena"]
    frame = np.copy(arena)

    # First we gather all the relevant information we need to describe
    # a state
    self_x, self_y, self_name, self_bombs_left, self_score = self.game_state["self"]

    coins = self.game_state["coins"]
    bombs = self.game_state["bombs"]  # (bx, by, bt)
    bombs = {(b[0], b[1]) for b in bombs}
    explosions = self.game_state["explosions"]
    opponents = self.game_state["others"]  # (ox, oy, oname, obombs, oscore)
    opponents = {(opp[0], opp[1]) for opp in opponents}

    # Now we save the walls in the current frame
    wall_indeces = frame == -1
    frame[wall_indeces] = GRAYSCALE_VALUES["wall"]

    # Then we save the crates in the current frame
    crate_indeces = frame == 1
    frame[crate_indeces] = GRAYSCALE_VALUES["crate"]

    # We also save the bombs in the current frame
    for bomb in bombs:
        b_x, b_y = bomb
        frame[b_x, b_y] = GRAYSCALE_VALUES["bomb"]

    # We save the coins in the current frame
    for coin in coins:
        c_x, c_y = coin
        frame[c_x, c_y] = GRAYSCALE_VALUES["coin"]

    if (self_x, self_y) not in bombs:
        frame[self_x, self_y] = GRAYSCALE_VALUES["self"]
    else:
        frame[self_x, self_y] = GRAYSCALE_VALUES["self_and_bomb"]

    # Next we save the explosions in the current frame
    explosion_indeces = explosions != 0
    frame[explosion_indeces] = GRAYSCALE_VALUES["explosion"]

    # Finally we save the agents in the current frame
    for opponent in opponents:
        o_x, o_y = opponent
        if opponent in bombs:
            frame[o_x, o_y] = GRAYSCALE_VALUES["enemy_and_bomb"]
        else:
            frame[o_x, o_y] = GRAYSCALE_VALUES["enemy"]

    return frame


def act(self):
    """Called each game step to determine the agent's next action.
    You can find out about the state of the game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of self.next_action will be used. The default value is 'WAIT'.
    """

    frame = get_current_frame(self)

    if len(self.frame_stack) == 0:
        self.frame_stack.init(frame)
    else:
        self.frame_stack.add(frame)

    state = self.frame_stack.getStack()
    output = self.deppmindagent.Q_network(state)[0]

    action, action_str = self.deppmindagent.get_action(output, self.episode)

    self.prev_state = state
    self.prev_action = action

    # And execute our action
    self.next_action = action_str
    self.cumulative_steps += 1


def train_depp_agent(self, terminal=0.0):
    """
    Called one we want to train our agent, which is right after reward_update(...) and
    end_of_episode(...). Depending on the outcome we can specify if the reached state is
    terminal or not
    """

    cur_frame = get_current_frame(self)
    cur_state = self.frame_stack.getStack(additional_frame=cur_frame)

    for evt in self.events:
        if evt == 11:
            self.collected_coins += 1

    reward = self.reward_scheme.getReward()
    self.collected_rewards += reward

    self.q_max = max(self.q_max, self.deppmindagent.Q_network(cur_state).max().item())

    reward = torch.Tensor(np.array([reward], dtype=np.float32)).unsqueeze(0)
    terminal = torch.Tensor(np.array([terminal], dtype=np.float32)).unsqueeze(0)
    if params["CUDA"]:
        reward = reward.cuda()
        terminal = terminal.cuda()

    self.replay_mem.add((self.prev_state, self.prev_action, reward, cur_state, terminal))
    state_batch, action_batch, reward_batch, state_new_batch, terminal_batch = self.replay_mem.sample_batch()

    loss = self.deppmindagent.update_Q_network(state_batch, action_batch, reward_batch, state_new_batch, terminal_batch)
    self.loss_list.append(loss)


def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.
    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """

    train_depp_agent(self, terminal=0.0)

    # We periodically update the Q-target weights with the Q-network weights
    if self.cumulative_steps % params["target_newtork_update_freq"] == 0:
        self.deppmindagent.update_target_network()
        print("updating target net after ", self.cumulative_steps, " steps")

    # print("end of reward update")


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.
    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """

    train_depp_agent(self, terminal=1.0)
    self.frame_stack.reset()

    loss_avg = sum(self.loss_list) / float(len(self.loss_list))
    dummy_1, dummy_2, dummy_3, dummy_4, self.score_points = self.game_state["self"]

    # finally we save the model depending on the frequency
    if self.episode % params["save_network_freq"] == 0:
        print(" == SAVING MODEL == ")
        pth = os.path.join(__location__, "brain0.pth")
        self.deppmindagent.save_model(pth)

    print(f"> finished ep {self.episode} | eps {self.deppmindagent.epsilon[self.episode]} | loss: {loss_avg}"
          f" | qmax: {self.q_max} | coins: {self.collected_coins} | score: {self.score_points} | rew: {self.collected_rewards}")

    self.loss_list_data.append(loss_avg)
    self.q_max_data.append(self.q_max)
    self.collected_coins_data.append(self.collected_coins)
    self.score_points_data.append(self.score_points)
    self.collected_rewards_data.append(self.collected_rewards)

    self.loss_list = []
    self.q_max = 0
    self.collected_coins = 0
    self.score_points = 0
    self.collected_rewards = 0
    self.cumulative_steps = 0

    self.episode += 1
    
    if self.episode % params["save_network_freq"] == 0:
        print("SAVING TRACKED DATA")
        for fname, dataset in [
            ["1loss.json", self.loss_list_data],
            ["1qmax.json", self.q_max_data],
            ["1coins.json", self.collected_coins_data],
            ["1score.json", self.score_points_data],
            ["1rewards.json", self.collected_rewards_data]
        ]:
            pth = os.path.join(__location__, fname)
            with open(pth, "w") as outfile:
                outfile.write(json.dumps(dataset))

    print("done")