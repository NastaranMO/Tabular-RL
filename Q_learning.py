#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next):
        ''' 
        Update the Q-value(s,a) estimate based on the the maximum value of next state s_next and reward r
        
        Parameters:
        - s: the current state
        - a: the action taken in the current state
        - r: the reward observed after taking action a in state s
        - s_next: the next state observed after taking action a in state s

        '''
        # Compute back-up estimate/target G_t
        G_t = r + self.gamma * np.max(self.Q_sa[s_next,])
        error = G_t - self.Q_sa[s,a]
        # Q-learning update
        self.Q_sa[s,a] += self.learning_rate * error

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    
    s = env.reset() # reset the environment


    for t in range(n_timesteps):
        a = agent.select_action(s, policy, epsilon, temp) # sample action
        s_next, r, done = env.step(a) # get the knowledge from the environment
        agent.update(s,a,r,s_next) # update the Q-value estimate

        # Evaluate after every eval_interval timesteps
        if t % eval_interval == 0:
            mean_return = agent.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(t)

        if done: 
            # Episode terminates
            s = env.reset()
        else:    
            # Update the state
            s = s_next
        

    if plot:
       env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=.2) # Plot the Q-value estimates during Q-learning execution


    return np.array(eval_returns), np.array(eval_timesteps)   

def test():
    n_timesteps = 50000
    eval_interval=100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)


if __name__ == '__main__':
    test()
