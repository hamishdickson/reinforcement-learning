# playing with openai gym

# random thing I didn't know: if you use a v0 env, 25% of the time your action
# is ignored and the previous action is repeated
# use v4 to get rid of this randomness

import gym
import numpy as np

# `make` creates and enviornment
env = gym.make("CartPole-v0")

# initialise it
# obs = env.reset()
# env.render()

# you can render as a numpy array for deep learning reasons I guess
# img = env.render(mode='rgb_array')
# img.shape
# h, w, channels (=3 for rgb)

# what actions are possible?
# env.action_space
# Discrete(2) - which means there are 2 actions (0 and 1)

# action = 1
# obs, reward, done, info = env.step(action)

# really basic policy:
# basically accelerate left when the pole is tilting left and
# accelerate right when we're tilting right:

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(1000): # ie 1k max
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break

    totals.append(episode_rewards)

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))