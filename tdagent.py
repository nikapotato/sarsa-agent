from abstractagent import AbstractAgent
from blackjack import BlackjackObservation, BlackjackEnv, BlackjackAction
from carddeck import *
from collections import defaultdict
from utils import player_has_ace

class TDAgent(AbstractAgent):
    """
    Implementation of an agent that plays the same strategy as the dealer.
    This means that the agent draws a card when sum of cards in his hand
    is less than 17.

    Your goal is to modify train() method to learn the state utility function
    and the get_hypothesis() method that returns the state utility function.
    I.e. you need to change this agent to a passive reinforcement learning
    agent that learns utility estimates using temporal difference method.
    """

    def __init__(self, env: BlackjackEnv, number_of_episodes: int):
        self.env = env
        self.number_of_episodes = number_of_episodes
        self.U = defaultdict(lambda :0)
        self.Ns = defaultdict(lambda :0)
        self.gamma = 1  # We are not discountng gamma = 1
        self.alpha = 0 # Just initial value for learning rate, will be seted later in train method, alpha function should be decreasing with number of state visits

    def train(self):
        for i in range(self.number_of_episodes):
            print(i)
            observation = self.env.reset()
            terminal = False
            reward = 0

            # Representation of state is 3-tuple (player_hand, dealer_hand, ace_info)
            s = (observation.player_hand.value(), observation.dealer_hand.value(), False)
            self.Ns[s] += 1
            while not terminal:
                # render method will print you the situation in the terminal
                self.env.render()
                '''
                                Repetat (for each step of episode):
                                    a = action given by policy for s
                                    Take action a, observe reward and next state
                                '''
                # Action given by policy for state s
                action = self.receive_observation_and_get_action(observation, terminal)
                observation, reward, terminal, _ = self.env.step(action)
                s_next = (observation.player_hand.value(), observation.dealer_hand.value(), player_has_ace(observation.player_hand.cards))
                self.Ns[s_next] += 1
                self.alpha = 10 / (9 + self.Ns[s])
                '''
                                   U(s) ← U(s) + αlpha(reward_s_next + γ*U(s_next) − U(s))
                                   Note: reward_s_next == reward
                               '''
                self.U[s] = self.U[s] + self.alpha * (reward + self.gamma * self.U[s_next] - self.U[s])
                '''
                    s ← s_next 
                '''
                s = s_next
            self.env.render()

    def receive_observation_and_get_action(self, observation: BlackjackObservation, terminal: bool) -> int:
        return BlackjackAction.HIT.value if observation.player_hand.value() < 17 else BlackjackAction.STAND.value

    def get_hypothesis(self, observation: BlackjackObservation, terminal: bool) -> float:
        """
        Implement this method so that I can test your code. This method is supposed to return your learned U value for
        particular observation.

        :param observation: The observation as in the game. Contains information about what the player sees - player's
        hand and dealer's hand.
        :param terminal: Whether the hands were seen after the end of the game, i.e. whether the state is terminal.
        :return: The learned U-value for the given observation.
        """
        s = [(observation.player_hand.value(), observation.dealer_hand.value(), player_has_ace(observation.player_hand.cards))]
        return self.U[s]
