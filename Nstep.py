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
from Helper import linear_anneal

class NstepQLearningAgent(BaseAgent):
     
        def update(self, states, actions, rewards, done, n):
            ''' states: is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
            actions: is a list of actions observed in the episode, of length T_ep
            rewards: is a list of rewards observed in the episode, of length T_ep
            done: indicates whether the final s in states is was a terminal state 
            '''
            T_ep = len(actions) # Episode length
            for t in range(T_ep):
                G_t = 0 # Expected return
                m = min(n, T_ep - t) # Number of rewards left to sum

                if done and t + n >= T_ep:
                    # Calculate n-step target without bootstraping
                    for i in range(T_ep - t):
                        G_t += self.gamma ** i * rewards[t+i]
                else:
                    # Calculate n-step target with bootstraping
                    G_t = sum([self.gamma ** i * rewards[t+i] for i in range(m)]) + self.gamma ** (m) * np.max(self.Q_sa[states[t+m] if t+n < T_ep else states[-1],])
                

                error = G_t - self.Q_sa[states[t], actions[t]]
                # Update Q-value using the learning rate and the error
                self.Q_sa[states[t], actions[t]] += self.learning_rate * error


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    total_steps = 0 # Total number of steps to use for evaluation

    
    # Train the agent
    while n_timesteps > 0:
        s = env.reset() # Sample initial state
        states = []
        actions = []
        rewards = []

        done = False # whether the episode is done or not

        # Collect states, actions, and rewards
        for i in range(min(max_episode_length, n_timesteps)):
            a = pi.select_action(s, policy, epsilon, temp) # Sample action (e.g, epsilon-greedy)
            s_next, r, done = env.step(a) # Simulate environment
            states.append(s)
            actions.append(a)
            rewards.append(r)

            # Episode terminates
            if done:
                break
            
            s = s_next # Update the state
            n_timesteps -= 1 # Decrease available timesteps
            total_steps += 1

            # Evaluate the episode
            if total_steps % eval_interval == 0:
                mean_return = pi.evaluate(eval_env)
                eval_returns.append(mean_return)
                eval_timesteps.append(total_steps)
        
        # Update Q-values
        pi.update(states, actions, rewards, done, n)
            
    if plot:
       env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution

    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 50000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    
    
if __name__ == '__main__':
    test()
