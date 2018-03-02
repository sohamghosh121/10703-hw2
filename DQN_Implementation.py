#!/usr/bin/env python
import tensorflow as tf
import numpy as np, gym, sys, copy, argparse
import random
import math

from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.optimizers import SGD
import json

from keras.models import load_model
from collections import deque


STATE_SPACE = {
	'CartPole-v0': 4,
	'MountainCar-v0': 2
}

ACTION_SPACE = {

	'CartPole-v0': 2,
	'MountainCar-v0': 3
}

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

	def __init__(self, environment_name, gamma=0.99, alpha=0.0001, num_past_states=4):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here. 
		self.name = 'VanillaQNetwork'
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
		self.model.save(self.name + '-' + suffix)

	def load_model_weights(self,filename):
		# Helper funciton to load model weights. 
		self.model = load_model(filename)


class LinearQNetwork(QNetwork):
	def __init__(self, environment_name, gamma=0.99, alpha=0.0001, num_past_states=4):
		self.name = 'LinearQNetwork'
		super(LinearQNetwork, self).__init__(environment_name, gamma, alpha)

	def init_network(self):
		self.input_size = (1, STATE_SPACE[self.environment_name] * self.num_past_states)
		self.inputs = Input(shape=(STATE_SPACE[self.environment_name] * self.num_past_states, )) # we take past 4 states
		self.outputs = Dense(ACTION_SPACE[self.environment_name])(self.inputs)
		self.model = Model(inputs=self.inputs, outputs=self.outputs)
		opt = SGD(lr=self.alpha, momentum=0.0, decay=0.0, nesterov=False)
		self.model.compile(optimizer=opt, loss='mean_squared_error')


class DeepQNetwork(QNetwork): # MLP
	def __init__(self, environment_name, gamma=0.99, alpha=0.0001, num_past_states=4, hidden_sizes=[10], activation='tanh'):
		self.name = 'DeepQNetwork'
		self.hidden_sizes = hidden_sizes
		self.activation = activation
		super(DeepQNetwork, self).__init__(environment_name, gamma, alpha)

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
		opt = SGD(lr=self.alpha, momentum=0.0, decay=0.0, nesterov=False)
		self.model.compile(optimizer=opt, loss='mean_squared_error')


class DuelingDeepQNetwork(QNetwork):
	pass


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
	
	def __init__(self, environment_name, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		self.render = render
		self.replay_memory = Replay_Memory()
		self.env = gym.make(environment_name)

	def Q_network(self, qnet):
		self.Q = qnet

	def epsilon_greedy_policy(self, epsilon, q_values):
		# Creating epsilon greedy probabilities to sample from.
		coin = np.random.random()
		greedy_action = self.greedy_policy(q_values)
		if coin > epsilon:
			return greedy_action 
		else:
			return np.random.choice([x for x in range(len(q_values)) if x != greedy_action])

	def greedy_policy(self, q_values):
		return np.argmax(q_values)

	def train(self, num_episodes=100000, epsilon_0=0.5, epsilon_T=0.05, experience_replay=0):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		decay_factor = 10 ** (math.log10(epsilon_T/epsilon_0)/float(num_episodes))

		if experience_replay:
			self.burn_in_memory()

		for t in range(num_episodes):
			eps = epsilon_0 * (decay_factor ** t)
			done = False
			s = self.env.reset()
			last_few_states = [s]
			reward = 0
			while not done:
				if len(last_few_states) == 4:
					a = self.epsilon_greedy_policy(eps, self.Q.get_Qvalues(np.concatenate(last_few_states)))
				else:
					a = self.env.action_space.sample()
				s_, r, done, info = self.env.step(a)
				reward += r
				if not experience_replay:
					if len(last_few_states) >= 4:
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
				s = s_
				if len(last_few_states) >= 4:
					last_few_states = last_few_states[1:] + [s]
				else:
					last_few_states.append(s)
			if (t + 1) % 100 == 0:
				print('Average reward at epoch', t + 1, ':   ', self.test(render=False))

	def run_test(self, render):
		done = False
		cum_reward = 0.0
		s = self.env.reset()
		if render:
			self.env.render()

		last_few_states = [s]
		while not done:
			# print(s, self.Q.get_Qvalues(s))
			if len(last_few_states) >= 4:
				a = self.epsilon_greedy_policy(0.05, self.Q.get_Qvalues(np.concatenate(last_few_states)))
			else:
				a = self.env.action_space.sample()
			s_, r, done, info = self.env.step(a)
			cum_reward += r
			s = s_
			
			if len(last_few_states) >= 4:
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
			# print(cumulative_reward)
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
				if len(last_few_states) >= 4:
					self.replay_memory.append((last_few_states, a, r, done, s_))
					last_few_states = last_few_states[1:] + [s_]
				else:
					last_few_states.append(s)
				if len(self.replay_memory.memory) > self.replay_memory.burn_in:
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
	parser.add_argument('--model_params',dest='model_params',type=str,default='\{\}')
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 
	
	try:
		model_params = json.loads(args.model_params)
	except:
		model_params = {}


	if args.model == 'LinearQNetwork':
		Q = LinearQNetwork(environment_name,
			gamma=args.gamma,
			alpha=args.alpha)
	elif args.model == 'DeepQNetwork':
		hidden_sizes = model_params.get('hidden_sizes', [10])
		Q = DeepQNetwork(environment_name, 
			gamma=args.gamma,
			alpha=args.alpha,
			hidden_sizes=hidden_sizes)
	elif args.model == 'DuelingDeepNetwork':
		pass
	else:
		print("Missing model name")
		exit()
	
	if args.model_file:
		Q.load_model_weights(args.model_file)

	agent = DQN_Agent(environment_name)
	agent.Q_network(Q)


	if args.train:
		agent.train(experience_replay=args.experience_replay, num_episodes=args.train)


	avg_reward = agent.test(render=args.render, num_episodes=args.test)
	print(avg_reward)

if __name__ == '__main__':
	main(sys.argv)
