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

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''

        T_ep = len(actions) # Length of the episode
        G = 0 # Start reward sum at 0
        
        # Calculate monte carlo backup at each step
        for t in reversed(range(T_ep)):
            G = self.gamma * rewards[t] + G
            # Update Q-value table             
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (G - self.Q_sa[states[t], actions[t]])
        

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    total_steps = 0 # Total number of steps to use for evaluation
    
    # Train the agent
    while n_timesteps > 0:
        s = env.reset() # Sample initial state

        states = [s]
        actions = []
        rewards = []
        done = False
        
        # Collect states, actions, and rewards
        for i in range(min(max_episode_length, n_timesteps)):
            a = pi.select_action(s, policy, epsilon, temp) # Sample action (e.g, epsilon-greedy)
            s_next, r, done = env.step(a) # Simulate environment

            states.append(s_next)
            actions.append(a)
            rewards.append(r)

            # Episode terminates
            if done:
                break

            # Update the state
            s = s_next
            
            n_timesteps -= 1 # Decrease available timesteps
            total_steps += 1

            # Evaluate the episode
            if total_steps % eval_interval == 0:
                mean_return = pi.evaluate(eval_env)
                eval_returns.append(mean_return)
                eval_timesteps.append(total_steps)
        # Update Q-values
        pi.update(states, actions, rewards)

    if plot:
       env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=5) # Plot the Q-value estimates during Monte Carlo RL execution

                 
    return np.array(eval_returns), np.array(eval_timesteps) 
    
def test():
    n_timesteps = 50000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    
            
if __name__ == '__main__':
    test()
