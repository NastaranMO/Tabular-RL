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
        #     T_ep = len(states)
        #     for t in range(T_ep):
        #         m = min(n, T_ep - t)
        #         if t + n >= T_ep:
        #             G_t = 0
        #             for i in range(T_ep - t):
        #                 G_t += self.gamma ** i * rewards[t+i]
        #         else:
        #             G_t = 0
        #             for i in range(m):
        #                 G_t += self.gamma ** i * rewards[t+i]
        #             G_t += (self.gamma ** m) * np.max(self.Q_sa[states[t+m]])
                
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
                if done and t + n >= T_ep:
                    for i in range(T_ep - t):
                        G_t += self.gamma ** i * rewards[t+i]
                else:
                    # print("m: ", m)
                    G_t = sum([self.gamma ** i * rewards[t+i] for i in range(m)]) + self.gamma ** (m) * np.max(self.Q_sa[states[t+m] if t+n < T_ep else states[-1],])
                error = G_t - self.Q_sa[states[t], actions[t]]
                self.Q_sa[states[t], actions[t]] += self.learning_rate * error

        # def update(self, states, actions, rewards, done, n):
        #     ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        #     actions is a list of actions observed in the episode, of length T_ep
        #     rewards is a list of rewards observed in the episode, of length T_ep
        #     done indicates whether the final state in states was a terminal state '''
        #     T_ep = len(actions)  # Assuming actions and rewards have the same length

        #     # Loop over each step in the episode
        #     for t in range(T_ep):
        #         # Calculate the n-step return G_t from state t
        #         G_t = sum(self.gamma**i * rewards[t + i] for i in range(min(n, T_ep - t)))
                
        #         # Add the value of the state at t+n, if it's not the terminal state
        #         if not (done and t + n >= T_ep):  # Check if state t+n is not terminal
        #             next_state = states[t + n] if (t + n) < T_ep else states[-1]
        #             G_t += self.gamma**n * np.max(self.Q_sa[next_state])

        #         # Update Q-value for state t and action taken at t
        #         current_q = self.Q_sa[states[t], actions[t]]
        #         error = G_t - current_q
        #         self.Q_sa[states[t], actions[t]] += self.learning_rate * error


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    total_steps = 0

    
    # TO DO: Write your n-step Q-learning algorithm here!
    # Collect episode
    while n_timesteps > 0:
        s = env.reset()
        states = []
        actions = []
        rewards = []

        done = False
        
        for i in range(min(max_episode_length, n_timesteps)):
            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)

            if done:
                # print("states", states, "actions", rewards)
                # print("next state", s_next)
                break
            s = s_next
            n_timesteps -= 1
            total_steps += 1

            if total_steps % eval_interval == 0:
                mean_return = pi.evaluate(eval_env)
                eval_returns.append(mean_return)
                eval_timesteps.append(total_steps)
        
        # Update Q-values
        pi.update(states, actions, rewards, done, n)
        # env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1)
            
    if plot:
       env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=10) # Plot the Q-value estimates during n-step Q-learning execution

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
