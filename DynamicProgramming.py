#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        # TO DO: your own code
        greedy_action = argmax(self.Q_sa[s])
        # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        return greedy_action
        
    def update(self,s,a,p_sas,r_sas): 
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # TO DO: Add own code
        self.Q_sa[s, a] = np.sum(p_sas * (r_sas + self.gamma * np.max(self.Q_sa, axis=1)))
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''

    print("Running Q-value iteration")
    # Initialize an agent
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
    
    max_error = np.Infinity
    i = 0
    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
    while max_error > threshold:    
        # Initialize max_error to 0 for the current iteration
        max_error = 0

        # Sweep through the state space
        for s in range(env.n_states):
            for a in range(env.n_actions):
                old_Q_sa = QIagent.Q_sa[s, a]
                p_sas, r_sas = env.model(s, a)
                QIagent.update(s, a, p_sas, r_sas)
                max_error = max(max_error, abs(QIagent.Q_sa[s, a] - old_Q_sa))
        i += 1

        # Plot current Q-value estimates & print max error
        # first value of step_pause is 0.2
        # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)

        print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
 
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    # env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)

    # view optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next

    # # TO DO: Compute mean reward per timestep under the optimal policy
    v_star = np.max(QIagent.Q_sa[3, :])
    [mean_number_of_steps] = np.abs((v_star - env.goal_rewards)/env.reward_per_step)       
    
    mean_reward_per_timestep = v_star/mean_number_of_steps

    print("Mean reward per timestep under optimal policy: {}, time: {}".format(mean_reward_per_timestep, mean_number_of_steps))
    
if __name__ == '__main__':
    experiment()


