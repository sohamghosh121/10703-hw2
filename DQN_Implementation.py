#!/usr/bin/env python
import tensorflow as tf
import numpy as np, gym, sys, copy, argparse
import random
import os

random.seed(42)
np.random.seed(42)

import math

from keras.models import Model, load_model
from keras.initializers import RandomNormal, TruncatedNormal
from keras.layers import Input, Dense, Activation, Merge, Lambda
from keras.optimizers import Adam
import json

import keras.backend as K


from collections import deque


STATE_SPACE = {
	'CartPole-v0': 4,
	'MountainCar-v0': 2
}

ACTION_SPACE = {

	'CartPole-v0': 2,
	'MountainCar-v0': 3
}

def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

'''
 ' Same as above but returns the mean loss.
'''
def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
  return K.mean(huber_loss(y_true, y_pred, clip_delta))

import sys
def print_dot():
	sys.stdout.write('.')
	sys.stdout.flush() 

class Replay_Memory():
	memory = []
	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions.
		self.memory = deque([], memory_size)
		self.memory_size = memory_size
		self.burn_in = burn_in
		self.memory_full = False
		

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		sampled = random.sample(self.memory, batch_size)
		return sampled


	def append(self, transition):
		# Appends transition to the memory.
		self.memory.append(transition)
		# if len(self.memory) >= self.memory_size:
		# 	self.memory_full = True
		# 	self.memory = self.memory[len(self.memory) - self.memory_size + 1:] + [transition]
		# else:
		# 	self.memory.append(transition)


"""
	Q-Networks
"""
class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name, gamma=0.99, alpha=0.0001, num_past_states=10):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here. 
		self.environment_name = environment_name
		self.gamma = gamma
		self.alpha = alpha
		self.num_past_states = num_past_states
		self.init_network()

	def init_network(self):
		pass

	def get_Qvalues(self, state):
		state = np.array(state).reshape(self.input_size)
		return self.model.predict(state).reshape(-1)

	def train(self, observations):
		inputs = []
		targets = []
		for observation in observations:
			last_few_states, a, r, is_terminal, s_ = observation
			S = np.concatenate(last_few_states)
			target = self.get_Qvalues(S)
			S_ = np.concatenate(last_few_states[1:] + [s_])
			if not is_terminal:
				target_val = r + self.gamma * np.max(self.get_Qvalues(S_))
			else:
				target_val = r
			target[a] = target_val
			inputs.append(S)
			targets.append(target)
		targets = np.array(targets)
		inputs = np.array(inputs)
		self.model.fit(inputs, targets, epochs=1, verbose=0)

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights.
		self.model.save(suffix)

	def load_model_weights(self,filename):
		# Helper funciton to load model weights. 
		self.model = load_model(filename)


class LinearQNetwork(QNetwork):
	def __init__(self, environment_name, gamma=0.99, alpha=0.0001, num_past_states=10):
		self.name = 'LinearQNetwork'
		super(LinearQNetwork, self).__init__(environment_name, gamma, alpha, num_past_states)

	def init_network(self):
		self.input_size = (1, STATE_SPACE[self.environment_name] * self.num_past_states)
		self.inputs = Input(shape=(STATE_SPACE[self.environment_name] * self.num_past_states, )) # we take past 4 states
		self.outputs = Dense(ACTION_SPACE[self.environment_name],
			kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.5, seed=None),
			bias_initializer=TruncatedNormal(mean=0.0, stddev=0.5, seed=None)
		)(self.inputs)

		# self.outputs = Dense(ACTION_SPACE[self.environment_name],
			# kernel_initializer=RandomNormal(mean=0.0, stddev=0.05))(self.inputs)

		self.model = Model(inputs=self.inputs, outputs=self.outputs)
		opt = Adam(lr=self.alpha)
		self.model.compile(optimizer=opt, loss='mean_squared_error')


class DeepQNetwork(QNetwork): # MLP
	def __init__(self, environment_name, gamma=0.99, alpha=0.0001, num_past_states=4, hidden_sizes=[10], activation='relu'):
		self.name = 'DeepQNetwork'
		self.hidden_sizes = hidden_sizes
		self.activation = activation
		super(DeepQNetwork, self).__init__(environment_name, gamma, alpha, num_past_states)

	def init_network(self):
		self.input_size = (1, STATE_SPACE[self.environment_name] * self.num_past_states)
		self.inputs = Input(shape=(STATE_SPACE[self.environment_name] * self.num_past_states, )) # we take past 4 states
		self.hidden_layers = []
		for i, h_sz in enumerate(self.hidden_sizes):
			if i == 0:
				h_layer = Dense(h_sz)(self.inputs)
				h_layer = Activation(self.activation)(h_layer)
			else:
				h_layer = Dense(h_sz)(self.hidden_layers[-1])
				h_layer = Activation(self.activation)(h_layer)
			self.hidden_layers.append(h_layer)

		self.outputs = Dense(ACTION_SPACE[self.environment_name])(self.hidden_layers[-1])
		self.model = Model(inputs=self.inputs, outputs=self.outputs)
		opt = Adam(lr=self.alpha)
		self.model.compile(optimizer=opt, loss='mean_squared_error')


class DuelingDeepQNetwork(QNetwork):
	def __init__(self,
			environment_name,
			gamma=0.99,
			alpha=0.0001,
			num_past_states=4,
			shared_hidden_sizes=[10],
			value_advantage_hidden_sizes=[10],
			action_advantage_hidden_sizes=[10],
			activation='relu'):
		self.name = 'DuelingDeepQNetwork'
		self.shared_hidden_sizes = shared_hidden_sizes
		self.value_advantage_hidden_sizes = value_advantage_hidden_sizes
		self.action_advantage_hidden_sizes = action_advantage_hidden_sizes
		self.activation = activation
		super(DuelingDeepQNetwork, self).__init__(environment_name, gamma, alpha, num_past_states)

	def init_network(self):
		self.input_size = (1, STATE_SPACE[self.environment_name] * self.num_past_states)
		self.inputs = Input(shape=(STATE_SPACE[self.environment_name] * self.num_past_states, )) # we take past 4 states
		self.shared_hidden_layers = []
		for i, h_sz in enumerate(self.shared_hidden_sizes):
			if i == 0:
				h_layer = Dense(h_sz)(self.inputs)
				h_layer = Activation(self.activation)(h_layer)
			else:
				h_layer = Dense(h_sz)(self.shared_hidden_layers[-1])
				h_layer = Activation(self.activation)(h_layer)
			self.shared_hidden_layers.append(h_layer)

		self.value_adv_hidden_layers = []
		for i, h_sz in enumerate(self.value_advantage_hidden_sizes):
			if i == 0:
				h_layer = Dense(h_sz)(self.shared_hidden_layers[-1])
				h_layer = Activation(self.activation)(h_layer)
			else:
				h_layer = Dense(h_sz)(self.value_adv_hidden_layers[-1])
				h_layer = Activation(self.activation)(h_layer)
			self.value_adv_hidden_layers.append(h_layer)

		self.action_adv_hidden_layers = []
		for i, h_sz in enumerate(self.action_advantage_hidden_sizes):
			if i == 0:
				h_layer = Dense(h_sz)(self.shared_hidden_layers[-1])
				h_layer = Activation(self.activation)(h_layer)
			else:
				h_layer = Dense(h_sz)(self.action_adv_hidden_layers[-1])
				h_layer = Activation(self.activation)(h_layer)
			self.action_adv_hidden_layers.append(h_layer)

		self.value_advantage_out = Dense(1)(self.value_adv_hidden_layers[-1])

		def get_action_advantage(x):
			means = K.mean(x, axis=1, keepdims=True)
			advantage = x - means
			return advantage

		def add_advantages(inputs):
			val_adv, act_adv = inputs
			return val_adv + act_adv

		output_shape = (ACTION_SPACE[self.environment_name],)

		self.action_value = Dense(ACTION_SPACE[self.environment_name])(self.action_adv_hidden_layers[-1])
		self.action_advantage_out = Lambda(get_action_advantage, output_shape=output_shape)(self.action_value)

		self.final_output = Merge(mode=add_advantages, output_shape=output_shape)([self.value_advantage_out, self.action_advantage_out])

		self.model = Model(inputs=self.inputs, outputs=self.final_output)
		opt = Adam(lr=self.alpha)
		self.model.compile(optimizer=opt, loss='mean_squared_error')


class ConvDuelingDeepQNetwork(QNetwork):
	pass



class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, render=False, save_dir='.', num_past_states=10):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		self.num_past_states = num_past_states
		self.render = render
		self.replay_memory = Replay_Memory()
		self.env = gym.make(environment_name)
		self.save_dir = save_dir

	def Q_network(self, qnet):
		self.Q = qnet

	def epsilon_greedy_policy(self, epsilon, q_values):
		# Creating epsilon greedy probabilities to sample from.
		coin = np.random.random()
		greedy_action = self.greedy_policy(q_values)
		if coin > epsilon:
			return greedy_action 
		else:
			return np.random.choice([x for x in range(len(q_values))])

	def greedy_policy(self, q_values):
		return np.argmax(q_values)

	def train(self, num_episodes=100000, epsilon_0=0.5, epsilon_T=0.05, experience_replay=0):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		decay_factor = (epsilon_0 - epsilon_T)/float(num_episodes)

		if experience_replay:
			self.burn_in_memory()

		best_reward = float('-Inf')
		training_rewards = []
		for t in range(num_episodes):
			if t < 500:
				eps = 1.0
			else:
				eps = epsilon_0 - (t - 500) * decay_factor
			done = False
			s = self.env.reset()
			# self.env.render()
			last_few_states = [s]
			reward = 0
			steps = 0
			while not done:
				if len(last_few_states) == self.num_past_states:
					a = self.epsilon_greedy_policy(eps, self.Q.get_Qvalues(np.concatenate(last_few_states)))
				else:
					a = self.env.action_space.sample()
				s_, r, done, info = self.env.step(a)
				reward += r
				if not experience_replay:
					if len(last_few_states) >= self.num_past_states:
						self.Q.train(
							[(
								last_few_states, 
								a,
								r,
								done,
								s_
							)]
						)
				else:
					observations = self.replay_memory.sample_batch()
					self.Q.train(observations)
					if len(last_few_states) == self.num_past_states:
						self.replay_memory.append((last_few_states, a, r, done, s_))
				s = s_
				if len(last_few_states) >= self.num_past_states:
					last_few_states = last_few_states[1:] + [s]
				else:
					last_few_states.append(s)
				steps += 1
				# if done and steps < 200:
				# 	print('reached goal', steps)

			training_rewards.append(reward)
			
			print_dot()	
			if (t + 1) % 10 == 0:
				avg_reward = self.test(render=False)
				sys.stdout.write(str(len(training_rewards)) + 'Training Reward  ' + str(np.mean(training_rewards)))
				sys.stdout.write(' Testing epoch ' +  str(t + 1) + ':   ' + str(avg_reward) + '\n')
				self.Q.save_model_weights(self.save_dir + '/epoch-%d' % (t + 1))
				best_reward = avg_reward
				training_rewards = []
				

	def run_test(self, render):
		done = False
		cum_reward = 0.0
		s = self.env.reset()
		if render:
			self.env.render()
		last_few_states = [s]
		# print('------------NEW TEST-------------')
		while not done:
			if len(last_few_states) >= self.num_past_states:
				a = self.epsilon_greedy_policy(0.05, self.Q.get_Qvalues(np.concatenate(last_few_states)))
			else:
				a = self.env.action_space.sample()
			# print(a)
			s_, r, done, info = self.env.step(a)
			cum_reward += r
			s = s_
			
			if len(last_few_states) >= self.num_past_states:
				last_few_states = last_few_states[1:] + [s]
			else:
				last_few_states.append(s)

			if render:
				self.env.render()
		return cum_reward


	def test(self, model_file=None, num_episodes=20, render = False):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		if model_file:
			self.Q_network.load_model_weights(model_file)
		# do testing
		rewards = []
		for ep_num in range(num_episodes):
			cumulative_reward = self.run_test(render)
			rewards.append(cumulative_reward)
		return np.mean(rewards)
		

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		while True:
			done = False
			s = self.env.reset()
			last_few_states = [s]
			while not done:
				a = self.env.action_space.sample()
				s_, r, done, info = self.env.step(a)
				if len(last_few_states) >= self.num_past_states:
					self.replay_memory.append((last_few_states, a, r, done, s_))
					last_few_states = last_few_states[1:] + [s_]
				else:
					last_few_states.append(s)
				if len(self.replay_memory.memory) >= self.replay_memory.burn_in:
					return
				s = s_

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--test',dest='test',type=int,default=20)
	parser.add_argument('--replay',dest='experience_replay',type=int,default=0)
	parser.add_argument('--batch_size',dest='batch_size',type=int,default=32)
	parser.add_argument('--gamma',dest='gamma',type=float,default=0.99)
	parser.add_argument('--alpha',dest='alpha',type=float,default=0.0001)
	parser.add_argument('--model',dest='model',type=str,default='LinearQNetwork')
	parser.add_argument('--model_file', dest='model_file',type=str)
	parser.add_argument('--exp_folder',dest='exp_folder',type=str,default='.')
	parser.add_argument('--num_past_states',dest='num_past_states',type=int,default=4)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 


	if args.model == 'LinearQNetwork':
		Q = LinearQNetwork(environment_name,
			gamma=args.gamma,
			alpha=args.alpha,
			num_past_states=args.num_past_states)
	elif args.model == 'DeepQNetwork':
		hidden_sizes = [10, 10, 10]
		Q = DeepQNetwork(environment_name, 
			gamma=args.gamma,
			alpha=args.alpha,
			hidden_sizes=hidden_sizes,
			num_past_states=args.num_past_states)
	elif args.model == 'DuelingDeepQNetwork':
		shared_hidden_sizes = [10,10]
		value_advantage_hidden_sizes = [5,5]
		action_advantage_hidden_sizes = [5,5]
		Q = DuelingDeepQNetwork(environment_name, 
			gamma=args.gamma,
			alpha=args.alpha,
			shared_hidden_sizes=shared_hidden_sizes,
			value_advantage_hidden_sizes=value_advantage_hidden_sizes,
			action_advantage_hidden_sizes=action_advantage_hidden_sizes,
			num_past_states=args.num_past_states)
	else:
		print("Missing model name")
		exit()
	
	if args.model_file:
		Q.load_model_weights(args.model_file)

	agent = DQN_Agent(environment_name, 
			save_dir=args.exp_folder,
			num_past_states=args.num_past_states)
	agent.Q_network(Q)


	if args.train:
		os.system("mkdir -p %s " % args.exp_folder)
		agent.train(experience_replay=args.experience_replay,
			num_episodes=args.train,
			epsilon_0=0.5,
			epsilon_T=0.05)


	avg_reward = agent.test(render=args.render, num_episodes=args.test)
	print('Final average reward: ', avg_reward)

if __name__ == '__main__':
	main(sys.argv)
