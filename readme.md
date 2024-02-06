# Overview

Implementation of the SARSA algorithm.

## Representation of State

We know many ways how we can represent the states of a problem. The simplest representation of a problem state is information about the total value of the cards that we and the dealer hold.

The state is represented as a pair of values (player_hand, dealer_hand), which indicate the total sum of the card values in the dealer's and the player's hands.
