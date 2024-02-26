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

class NstepQLearningAgent(BaseAgent):
        
    # def update(self, states, actions, rewards, done, n):
    #     ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
    #     actions is a list of actions observed in the episode, of length T_ep
    #     rewards is a list of rewards observed in the episode, of length T_ep
    #     done indicates whether the final s in states is was a terminal state '''
    #     # TO DO: Add own code
    #     T_ep = len(actions)
    #     G_t = 0
    #     # print("timesteps: ", T_ep)
    #     for t in range(T_ep):
    #         m = min(n, T_ep - t)
    #         if done and m < n:
    #             G_t = sum([self.gamma ** i * rewards[t+i] for i in range(m)])
    #         else:
    #             G_t = sum([self.gamma ** i * rewards[t+i] for i in range(m)]) + self.gamma ** m * np.max(self.Q_sa[states[t+n] if t+n < T_ep else states[-1],])
    #         error = G_t - self.Q_sa[states[t], actions[t]]
    #         self.Q_sa[states[t], actions[t]] += self.learning_rate * error

        def update(self, states, actions, rewards, done, n):
            ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
            actions is a list of actions observed in the episode, of length T_ep
            rewards is a list of rewards observed in the episode, of length T_ep
            done indicates whether the final s in states is was a terminal state '''
            # TO DO: Add own code
            T_ep = len(actions)
            # print("timesteps: ", T_ep)
            for t in range(T_ep):
                G_t = 0
                m = min(n, T_ep - t)
                # print("tp",T_ep - t)
                if done or t + n >= T_ep:
                    for i in range(T_ep - t):
                        G_t += self.gamma ** i * rewards[t+i]
                else:
                    # print("m: ", m)
                    G_t = sum([self.gamma ** i * rewards[t+i] for i in range(m)]) + self.gamma ** (m) * np.max(self.Q_sa[states[t+m] if t+n < T_ep else states[-1],])
                error = G_t - self.Q_sa[states[t], actions[t]]
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

    
    # TO DO: Write your n-step Q-learning algorithm here!
    # Collect episode
    for t in range(n_timesteps):
        s = env.reset()
        states = []
        actions = []
        rewards = []
        
        done = False
        for e in range(max_episode_length):
            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            # pi.update(states, actions, rewards, done, n)
            if done:
                # print("done")
                break
            s = s_next
            # print("episode: ", e)
        # Update Q-values
        pi.update(states, actions, rewards, done, n)
        # env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1)
            
        
        if t % eval_interval == 0:
            mean_return = pi.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(t)
    if plot:
       env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=10) # Plot the Q-value estimates during n-step Q-learning execution

    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 10000
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
