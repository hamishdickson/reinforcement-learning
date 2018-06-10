"""
Easy 21

Easy 21 is a game similar to Blackjack:

- the game is applied to an infinite deck of cards (ie cards are supplied by replacement)
- each draw from the deck results in a value between 1 and 10 (uniformly distributed) with a colour of red (prob 1/3) or black (prob 2/3)
- there are no aces or picture (face) cards in this game
- at the start of the game both the player and dealer draw one black card (fully observed)
- each turn the player may either stick or hit
- if the player hits then she draws another card from the deck
- if the player sticks she recieves no further cards
- the values of the player's cards are added (black cards) or subtracted (red cards)
- if the player's sum exceeds 21 or becomes less than 1, then she "goes bust" and loses the game (reward -1)
- if the player sticks then the dealer starts taking turns. The dealer always sticks on any sum of 17 or greater and hits otherwise. If the dealer goes bust, then the player wins; otherwise the outcome - win (reward + 1). lose (reward -1), or draw (reward 0) - is the player with the largest sum



1. Write an environment that implements the game Easy21. Specifically, write a function named `step` which takes as input a state `s` (dealers first card 1-10 and the player's sum 1-21) and an action `a` (hit or stick) and returns a sample of the next state s' (which may be terminal if the game is finished) and reward r.

We will be using this environment for model-free reinforcement
learning, and you should not explicitly represent the transition matrix for the
MDP. There is no discounting (γ = 1). You should treat the dealer’s moves as
part of the environment, i.e. calling step with a stick action will play out the
dealer’s cards and return the final reward and terminal state.


Resources: http://incompleteideas.net/book/bookdraft2017nov5.pdf

"""

