import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from supervised_learning import Net
from MCTS import MCTS
from collections import namedtuple

_TTTB = namedtuple("Connect4Board","env board turn winner terminal")

class ConnectNode:

    def __init__(self,env,trainer,state,steps,terminal,reward):

        self.env = env
        self.epsilon = 0.25
        self.state = state
        self.steps = steps
        self.trainer = trainer
        #self.player = player
        #self.winner = winner

        self.__terminal = terminal
        self.__reward = reward


    def default_policy_search(self,NN,children):
        '''
        This function will serve as a replacement for
        acting randomly during the simulation phase
        of rollout. It will take the form of an epsilon greedy
        algorithm
        :return:
        The policy will use the NN
        for guidance
        '''

        if np.random.random() < self.epsilon:
            return self.find_random_child()
        else:
            if not children:
                children = self.find_children()

            best_reward = -100
            best_node = None
            for i,child in enumerate(children):
                board = np.array(child.state[0].observation.board)
                x = torch.from_numpy(board).type(dtype=torch.float)
                reward = NN(x).detach().numpy()[0]
                if reward > best_reward:
                    best_node = child

            return best_node

    def find_children(self):

        self.env.state = self.state.copy()
        self.env.steps = self.steps.copy()

        if self.__terminal:
            return set()

        # get all possible actions from the current node
        return {
            self.make_move(i) for i in range(self.env.configuration.columns) if
                   self.env.state[0].observation.board[i] == 0
        }

    def find_random_child(self):

        self.env.state = self.state.copy()
        self.env.steps = self.steps.copy()

        possible_actions = [i for i in range(self.env.configuration.columns) if
                   self.env.state[0].observation.board[i] == 0]

        return self.make_move(np.random.choice(possible_actions))

    def get_reward(self,r):

        reward_conv = {
            0.0:-1.0,
            0.5:0.0,
            1.0:1.0
        }

        return reward_conv[r]

    def reward(self):
        return self.__reward

    def is_terminal(self):
        return self.__terminal

    def make_move(self, index):

        self.env.state = self.state.copy()
        self.env.steps = self.steps.copy()

        observation, reward, done, info = self.trainer.step(int(index))

        reward = self.get_reward(reward)

        new_state = self.env.state.copy()
        new_steps = self.env.steps.copy()

        return ConnectNode(self.env,self.trainer,new_state,new_steps,done,reward)

    def to_pretty_string(self):

        self.env.state = self.state.copy()
        self.env.steps = self.steps.copy()

        self.env.render()


def play_game():

    inputDim = 42  # takes variable 'x'
    outputDim = 1  # takes variable 'y'
    hidden_layer_size = 100

    ###################
    # Load in NN
    ###################

    model = Net(inputDim, hidden_layer_size, outputDim)

    model.load_state_dict(torch.load('data/model_weights'))

    model.eval()

    tree = MCTS(NN=model)

    from kaggle_environments import evaluate, make, utils

    env = make("connectx", debug=True)

    env.render()

    # Play as first position against random agent.
    trainer = env.train([None, "negamax"])

    state = env.state.copy()
    steps = env.steps.copy()
    terminal = env.done
    reward = 0.0

    root_node = ConnectNode(env,trainer,state,steps,terminal,reward)

    tree.do_rollout(root_node)

    rewards = []
    for episode in range(50):
        print('Running episode {}'.format(episode))
        reward_c = []
        for episode2 in range(10):
            trainer.reset()
            node = root_node

            while True:

                for _ in range(5):
                    tree.do_rollout(node)

                node = tree.choose(node)

                state = node.state.copy()
                steps = node.steps.copy()

                env.state = state
                env.steps = steps

                if node.is_terminal():
                    reward_c.append(node.reward())
                    break

        rewards.append(sum(reward_c) / float(len(reward_c)))


    # Few a few games after training is done
    for _ in range(5):

        print('SHOWING NEW GAME')
        print('---------------------------------------')
        node = root_node
        env.reset()
        trainer.reset()
        np.random.seed(np.random.randint(0,10000))
        while True:
            print('  ')
            print('  ')
            node = tree.choose(node)

            state = node.state.copy()
            steps = node.steps.copy()

            env.state = state
            env.steps = steps

            env.render()

            if node.is_terminal():
                break

        print('---------------------------------------')



    plt.title('Avg win vs Iters')
    plt.plot(rewards)
    plt.show()

    print('Avg reward = {}'.format(np.mean(rewards)))


if __name__ == '__main__':

    play_game()


