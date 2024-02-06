from abstractagent import AbstractAgent
from blackjack import BlackjackEnv, BlackjackObservation
from carddeck import *
from collections import defaultdict
from utils import player_has_ace
from enum import Enum
import random

"""
* SARSA *
Algorithm parameters: step size alpha is (0, 1], small epsilon > 0
Initialize Q(s, a), for all s in S+, a is A(s), arbitrarily except that Q(terminal, ·)=0
Loop for each episode:
    Initialize S
    Choose A from S using policy derived from Q (e.g., "-greedy)
    Loop for each step of episode:
        Take action A, observe R, S'
            Choose A' from S' using policy derived from Q (e.g., -greedy)
            Q(S, A) <- Q(S, A) + alpha*R + gamma*Q(S',A') - Q(S,A)
            S <- S'
            A <- A'
    until S is terminal
"""

"""
Possible actions
STAND - Take no more cards.
HIT - Take another card.
"""
class ACTIONS(Enum):
    STAND = 0
    HIT = 1

class SarsaAgent(AbstractAgent):
    """
    Here you will provide your implementation of SARSA method.
    You are supposed to implement train() method. If you want
    to, you can split the code in two phases - training and
    testing, but it is not a requirement.

    For SARSA explanation check AIMA book or Sutton and Burton
    book. You can choose any strategy and/or step-size function
    (learning rate) as long as you fulfil convergence criteria.
    """
    def __init__(self, env: BlackjackEnv, number_of_episodes: int):
        self.env = env
        self.number_of_episodes = number_of_episodes
        self.Q = defaultdict(lambda: [0.0, 0.0])
        self.Ns = defaultdict(lambda: 0)
        self.gamma = 1  # We are not discountng gamma = 1
        self.alpha = 0  # Just initial value for learning rate, will be seted later in train method, alpha function should be decreasing with number of state visits

    """
    Epsilon greedy policy
    """
    def player_greedy_decision(self, state):
        epsilon = (1 / (10 + self.Ns[state]))
        if random.uniform(0,1) < epsilon:
            # Choose random action
            action = self.env.action_space.sample()
            return action
        else:
            # Choose action of a greedy policy
            action = 0 if self.Q[state][ACTIONS.STAND.value] > self.Q[state][ACTIONS.HIT.value] else 1
            return action

    def train(self):
        for i in range(self.number_of_episodes):
            print(i)
            observation = self.env.reset()
            terminal = False
            reward = 0

            # Representation of state is 3-tuple (player_hand, dealer_hand, ace_info)
            s = (observation.player_hand.value(), observation.dealer_hand.value(), False)
            self.Ns[s] += 1
            # Action
            action = self.player_greedy_decision(s)

            while not terminal:
                observation, reward, terminal, _ = self.env.step(action)
                s_next = (observation.player_hand.value(), observation.dealer_hand.value(), player_has_ace(observation.player_hand.cards))
                self.Ns[s_next] += 1
                '''
                Choose A' from S' using policy derived from Q (e.g., epsilon-greedy)
                '''
                action_next = self.player_greedy_decision(s_next)
                '''
                Q(S, A) <- Q(S, A) + alpha*(R + gamma*Q(S',A') - Q(S,A))
                '''
                self.alpha = 10 / (9 + self.Ns[s])
                # print(self.alpha)
                self.Q[s][action] = self.Q[s][action] + self.alpha*(reward + self.gamma*self.Q[s_next][action_next]-self.Q[s][action])
                '''
                S <- S'
                A <- A'
                '''
                s = s_next
                action = action_next

        return self.Q


    def get_hypothesis(self, observation: BlackjackObservation, terminal: bool, action: int) -> float:
        """
        Implement this method so that I can test your code. This method is supposed to return your learned Q value for
        particular observation and action.

        :param observation: The observation as in the game. Contains information about what the player sees - player's
        hand and dealer's hand.
        :param terminal: Whether the hands were seen after the end of the game, i.e. whether the state is terminal.
        :param action: Action for Q-value.
        :return: The learned Q-value for the given observation and action.
        """
        s = (observation.player_hand.value(), observation.dealer_hand.value(), player_has_ace(observation.player_hand.cards))
        return self.Q[s][action]
