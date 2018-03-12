#!/usr/bin/env python
import tensorflow as tf
import numpy as np, gym, sys, copy, argparse
import random
import os

random.seed(42)
np.random.seed(42)

import math

import keras
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
  cond  = K.abs(error) < clip_delta

  squared_loss = 0.5 * K.square(error)
  linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
	"""
		Implements Huber loss for Keras
	"""
  	return K.mean(huber_loss(y_true, y_pred, clip_delta))

keras.losses.huber_loss_mean = huber_loss_mean # add this to Keras's list of losses

import sys
def print_dot():
	sys.stdout.write('.')
	sys.stdout.flush() 

class Replay_Memory():
	"""
		Stores transitions from which samples are drawn from using random.sample
		Uses a deque object to maintain a fixed buffer size (discards old elements)
	"""
	def __init__(self, memory_size=50000, burn_in=10000):
		self.memory = deque([], memory_size)
		self.memory_size = memory_size
		self.burn_in = burn_in
		self.memory_full = False
		

	def sample_batch(self, batch_size=32):
		"""
			Use random.sample to sample a batch_size size of transitions from memory
		"""
		sampled = random.sample(self.memory, batch_size)
		return sampled


	def append(self, transition):
		"""
			Add transition to memory. Since we use deque, this takes care of the
			list length automatically
		"""
		self.memory.append(transition)

"""
	Q-Networks
"""
class QNetwork():
	"""
		This is the base class inherited by all our models.
		It calls init_network which should be overridden by each model that sets
		up the network architecture
		This also sets some of the hyperparameters (gamma, learning rate, how many states)
	"""
	def __init__(self, environment_name, gamma=0.99, alpha=0.0001, num_past_states=4):
		self.environment_name = environment_name
		self.gamma = gamma
		self.alpha = alpha
		self.num_past_states = num_past_states
		self.init_network()

	def init_network(self):
		"""
			Should be overridden by subclasses
			- create network architecture
			- set optimizer
			- compile model
		"""
		pass

	def get_Qvalues(self, state):
		"""
			Calls self.model to get the Q values for different actions
		"""
		state = np.array(state).reshape(self.input_size)
		return self.model.predict(state).reshape(-1)

	def train(self, observations):
		"""
			Implements the Q-learning training procedure
			- creates a batch of inputs, targets where targets are calculated
			using the Q network
			- calls self.model.fit for this batch
		"""
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

	def save_model_weights(self, filename):
		# Helper function to save model.
		self.model.save(filename)

	def load_model_weights(self,filename):
		# Helper funciton to load model. 
		self.model = load_model(filename)


class LinearQNetwork(QNetwork):
	"""
		Q(s, a) = w^T phi(s, a)
	"""
	def __init__(self, environment_name, gamma=0.99, alpha=0.0001, num_past_states=10):
		self.name = 'LinearQNetwork'
		super(LinearQNetwork, self).__init__(environment_name, gamma, alpha, num_past_states)

	def init_network(self):
		"""
			Creates a linear network with no activations and no hidden layers
		"""
		self.input_size = (1, STATE_SPACE[self.environment_name] * self.num_past_states)
		self.inputs = Input(shape=(STATE_SPACE[self.environment_name] * self.num_past_states, )) # we take past 4 states
		self.outputs = Dense(ACTION_SPACE[self.environment_name],
			kernel_initializer=RandomNormal(0, 1.0)
		)(self.inputs)
		self.model = Model(inputs=self.inputs, outputs=self.outputs)
		opt = Adam(lr=self.alpha)
		self.model.compile(optimizer=opt, loss='mean_squared_error')


class DeepQNetwork(QNetwork):
	"""
		This is a multi-layered perceptron with specified number of hidden layers
		and specified activation function
	"""
	def __init__(self, environment_name, gamma=0.99, alpha=0.0001, num_past_states=4, hidden_sizes=[10], activation='relu'):
		self.name = 'DeepQNetwork'
		self.hidden_sizes = hidden_sizes
		self.activation = activation
		super(DeepQNetwork, self).__init__(environment_name, gamma, alpha, num_past_states)

	def init_network(self):
		self.input_size = (1, STATE_SPACE[self.environment_name] * self.num_past_states)
		self.inputs = Input(shape=(STATE_SPACE[self.environment_name] * self.num_past_states, ))
		self.hidden_layers = []
		for i, h_sz in enumerate(self.hidden_sizes):
			if i == 0: # if the first hidden layer, take the input layer as input to the hidden layer
				h_layer = Dense(h_sz)(self.inputs)
				h_layer = Activation(self.activation)(h_layer)
			else: # else the hidden layer should take the previous hidden layer as input
				h_layer = Dense(h_sz)(self.hidden_layers[-1])
				h_layer = Activation(self.activation)(h_layer)
			self.hidden_layers.append(h_layer)

		self.outputs = Dense(ACTION_SPACE[self.environment_name])(self.hidden_layers[-1])
		self.model = Model(inputs=self.inputs, outputs=self.outputs)
		opt = Adam(lr=self.alpha)
		self.model.compile(optimizer=opt, loss=huber_loss_mean)


class DuelingDeepQNetwork(QNetwork):
	"""
		This is a multi-layered perceptron with specified number of hidden layers
		and specified activation function with a branching architecture.

		Inputs to the constructor shared_hidden_sizes,  value_advantage_hidden_sizes,
		action_advantage_hidden_sizes define how many hidden layers (and their sizes)
		in each part of the network.
	"""
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

		# create shared part
		self.shared_hidden_layers = []
		for i, h_sz in enumerate(self.shared_hidden_sizes):
			if i == 0:
				h_layer = Dense(h_sz)(self.inputs)
				h_layer = Activation(self.activation)(h_layer)
			else:
				h_layer = Dense(h_sz)(self.shared_hidden_layers[-1])
				h_layer = Activation(self.activation)(h_layer)
			self.shared_hidden_layers.append(h_layer)

		# create value advantage part
		self.value_adv_hidden_layers = []
		for i, h_sz in enumerate(self.value_advantage_hidden_sizes):
			if i == 0:
				h_layer = Dense(h_sz)(self.shared_hidden_layers[-1])
				h_layer = Activation(self.activation)(h_layer)
			else:
				h_layer = Dense(h_sz)(self.value_adv_hidden_layers[-1])
				h_layer = Activation(self.activation)(h_layer)
			self.value_adv_hidden_layers.append(h_layer)

		# create action advantage part
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



class DQN_Agent():
	"""
		This is the agent class.
		- initiates the environment by calling gym.make
		- chooses where to store models
		- other parameters include whether to repeatedly sample an action or not
	"""
	def __init__(self, environment_name, render=False, save_dir='.', num_past_states=10, repeated_sampling=0):
		self.num_past_states = num_past_states
		self.render = render
		self.replay_memory = Replay_Memory()
		self.env = gym.make(environment_name)
		self.save_dir = save_dir
		self.repeated_sampling = repeated_sampling

	def Q_network(self, qnet):
		"""
			Set Q-Network for this agent
			IMPORTANT: This should be called before training the agent.
		"""
		self.Q = qnet

	def epsilon_greedy_policy(self, epsilon, q_values):
		"""
			Creating epsilon greedy probabilities to sample from.
		"""
		coin = np.random.random()
		greedy_action = self.greedy_policy(q_values)
		if coin > epsilon:
			return greedy_action 
		else:
			return np.random.choice([x for x in range(len(q_values))])

	def greedy_policy(self, q_values):
		"""
			Get greedy policy (argmax_a Q(s, a))
		"""
		return np.argmax(q_values)

	def train(self, num_episodes=100000, epsilon_0=0.5, epsilon_T=0.05, decay_over=100000, experience_replay=0):

		# calculate how much to decay epsilon by every episode
		decay_factor = (epsilon_0 - epsilon_T)/float(decay_over)

		# if experience replay is chosen, first we need to burn in memory
		if experience_replay:
			self.burn_in_memory()

		training_rewards = []
		total_iterations = 0
		for t in range(num_episodes):
			eps = max(epsilon_T, epsilon_0 - t * decay_factor)
			done = False
			s = self.env.reset()
			if self.render:
				self.env.render()
			last_few_states = [s]
			reward = 0
			repeats_left = self.repeated_sampling
			a = None
			while not done:
				if len(last_few_states) == self.num_past_states and repeats_left == 0:
					# if no more repeats left, and need to sample new action, and already have enough past actions
					a = self.epsilon_greedy_policy(eps, self.Q.get_Qvalues(np.concatenate(last_few_states)))
					repeats_left = self.repeated_sampling
				elif len(last_few_states) < self.num_past_states:
					# this case is for the start when we don't have enough past states
					a = self.env.action_space.sample()
				elif a is None:
					# last corner case
					a = self.epsilon_greedy_policy(eps, self.Q.get_Qvalues(np.concatenate(last_few_states)))
				else:
					# if repeating action, don't sample again
					repeats_left -= 1
				s_, r, terminal, info = self.env.step(a) # take action
				if self.render: 
					self.env.render()
				reward += r
				if not experience_replay:
					# if not using experience replay, use this transition
					if len(last_few_states) >= self.num_past_states:
						self.Q.train(
							[(
								last_few_states, 
								a,
								r,
								terminal,
								s_
							)]
						)
				else:
					# using experience replay -> sample a batch, and train on that
					observations = self.replay_memory.sample_batch()
					self.Q.train(observations)
					if len(last_few_states) == self.num_past_states:
						self.replay_memory.append((last_few_states, a, r, terminal, s_))
				
				# update
				s = s_
				if len(last_few_states) >= self.num_past_states:
					last_few_states = last_few_states[1:] + [s]
				else:
					last_few_states.append(s)
				total_iterations += 1

			# this is used to calculate average training reward at the end of an episode
			training_rewards.append(reward)
			
			print_dot()	
			if (t + 1) % 100 == 0: # every 100 episodes, test agent
				avg_reward = self.test(render=self.render)
				sys.stdout.write(' Training Reward  ' + str(np.mean(training_rewards)))
				sys.stdout.write(' Testing Reward @ epoch ' +  str(t + 1) + ':   ' + str(avg_reward) + '\n')
				self.Q.save_model_weights(self.save_dir + '/epoch-%d' % (t + 1))
				training_rewards = []
				

	def run_test(self, render):
		"""
			Runs 1 episode of greedy policy with epsilon 0.05
		"""
		done = False
		cum_reward = 0.0
		s = self.env.reset()
		if render:
			self.env.render()
		last_few_states = [s]
		while not done:
			if len(last_few_states) >= self.num_past_states:
				a = self.epsilon_greedy_policy(0.05, self.Q.get_Qvalues(np.concatenate(last_few_states)))
			else:
				a = self.env.action_space.sample()
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
		"""
			Calls run_test num_episodes times to get testing reward
		"""
		if model_file:
			self.Q_network.load_model_weights(model_file)
		# do testing
		rewards = []
		for ep_num in range(num_episodes):
			cumulative_reward = self.run_test(render)
			rewards.append(cumulative_reward)
		return np.mean(rewards), np.std(rewards)

	def record_video(self, model_file=None, video_dir='.'):
		"""
			Wrap the environment in a monitor to record video to specified 
			directory
		"""
		self.env = gym.wrappers.Monitor(self.env, video_dir) # this should automatically generate videos
		self.run_test(False)
				
	def burn_in_memory(self):
		"""
			Randomly sample actions to create burn in memory for agent
		"""
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
	parser.add_argument('--train',dest='train',type=int,default=0)
	parser.add_argument('--test',dest='test',type=int,default=20)
	parser.add_argument('--replay',dest='experience_replay',type=int,default=0)
	parser.add_argument('--gamma',dest='gamma',type=float,default=0.99)
	parser.add_argument('--alpha',dest='alpha',type=float,default=0.0001)
	parser.add_argument('--model',dest='model',type=str,default='LinearQNetwork')
	parser.add_argument('--model_file', dest='model_file',type=str)
	parser.add_argument('--exp_folder',dest='exp_folder',type=str,default='.')
	parser.add_argument('--num_past_states',dest='num_past_states',type=int,default=4)
	parser.add_argument('--repeated_sampling',dest='repeated_sampling',type=int,default=0)
	parser.add_argument('--record_video',dest='record_video',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)


	if args.model == 'LinearQNetwork':
		Q = LinearQNetwork(environment_name,
			gamma=args.gamma,
			alpha=args.alpha,
			num_past_states=args.num_past_states)
	elif args.model == 'DeepQNetwork':
		if args.env == 'MountainCar-v0':
			hidden_sizes = [30, 30, 30]
		elif args.env == 'CartPole-v0':
			hidden_sizes = [10, 10, 10]
		Q = DeepQNetwork(environment_name, 
			gamma=args.gamma,
			alpha=args.alpha,
			hidden_sizes=hidden_sizes,
			num_past_states=args.num_past_states)
	elif args.model == 'DuelingDeepQNetwork':
		if args.env == 'MountainCar-v0':
			shared_hidden_sizes = [30,30]
			value_advantage_hidden_sizes = [30]
			action_advantage_hidden_sizes = [30]
		elif args.env == 'CartPole-v0':
			shared_hidden_sizes = [10,10]
			value_advantage_hidden_sizes = [10]
			action_advantage_hidden_sizes = [10]
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
			render=False, 
			save_dir=args.exp_folder,
			num_past_states=args.num_past_states,
			repeated_sampling=args.repeated_sampling)
	agent.Q_network(Q)


	if args.train:
		os.system("mkdir -p %s " % args.exp_folder)
		agent.train(experience_replay=args.experience_replay,
			num_episodes=args.train,
				decay_over=100000,
				epsilon_0=0.5,
				epsilon_T=0.05)


	if args.test:
		avg_reward, stddev_reward = agent.test(render=args.render, num_episodes=args.test)
		print('Final average reward: ', avg_reward, '+/-', stddev_reward)

	if args.record_video:
		agent.record_video(video_dir=args.record_video)

if __name__ == '__main__':
	main(sys.argv)


