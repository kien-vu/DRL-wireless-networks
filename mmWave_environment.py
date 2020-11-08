
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class mmWenvironment:

    def __init__(self, initial_value, target_value, predict_value, random_state, state_size=4, action_size=4):
        np.random.seed(random_state)
        self.initial_value = initial_value
        self.target_value = target_value
        self.state_size = state_size
        self.action_size = action_size
        self.observation_space = np.arange(action_size)
        self.action_space = np.arange(action_size)
        self.predict_value = predict_value
        self.reset()

    def reset(self):
        self.score = 0.
        self.current_state = 0
        self.last_action = self.current_state
        self.last_reward = 0.
        self.beamforming_on = False

        return np.arange(self.state_size)

    def __str__(self):
        out = '<environment.Rate_environment>'  + '\n'
        out += 'SINR state: {} dB.'.format(str(self.current_state)) + '\n'
        out += 'SINR score: {} dB.'.format(str(self.score)) + '\n'

        return out

    def step(self, action):
        """
          Performs the given action in the current
          environment state and updates the environment.

          Returns a (reward, nextState) pair
        """
        self.last_action = action
        # done = False
        reward = 0.

        valid_move = (self.predict_value[action] >= 0)
        actions = self.action_space

        self.last_reward = reward

        # Pull a valid move at random
        action = actions[valid_move == True]
        # action = action[0] if (len(action) > 1) else action
        if (any(valid_move)):
            np.random.shuffle(action)
            action = action[0]
            reward = self.predict_value[self.current_state, action]

            if (action == 2): # the beamforming can be switched on once
                if not self.beamforming_on:
                    self.beamforming_on = True
                else:
                    reward = 0.

        else:
            reward = 0.

        next_state = actions
        #  self.score += reward
        #  done = (self.score >= self.target_value)

        return ([next_state, reward]  )  # , done, self.score])
