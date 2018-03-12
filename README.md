# 10-703 Homework 2 (model free learning)

To train and test a model on a specified environment, run the following command:

~~~~
>> python DQN_Implementation.py \
    --env <env-name: MountainCar-v0/CartPole-v0> \
    --train <number of episodes to train for> \
    --replay <0: no experience replay, 1: use experience replay> \
    --model <which model to train LinearQNetwork/DeepQNetwork/DuelingDeepQNetwork > \
    --gamma <value of gamma to use> \
    --exp_folder <which directory to save models in> \
    --num_past_states <number of past states to use as state representation, default 4> \
    --repeated_sampling <how many times to repeatedly sample an action (20 for MountainCar)>
    --test <number of times to test the model after finishing training>
    --model_file <path to model weights>
    --render <1: render the environment, 0: don't render>
    --alpha <learning rate>
    --record_video <path to directory where to save video after finishing training/testing>
~~~~

For example, to run Linear Q-Network for CartPole-v0, run the following command:
~~~~
>> python DQN_Implementation.py \
    --env CartPole-v0 \
    --train 8000 \
    --replay 0 \
    --model LinearQNetwork \
    --gamma 0.99 \
    --exp_folder "CartPole__LinearNoReplay" \
    --num_past_states 4 \
    --repeated_sampling 0
~~~~ 

To simply test a trained model and also record video of it,
~~~~
>> python DQN_Implementation.py \
    --env CartPole-v0 \
    --train 0 \
    --replay 0 \
    --model LinearQNetwork \
    --gamma 0.99 \
    --num_past_states 4 \
    --repeated_sampling 0 \
    --model_file "CartPole__LinearNoReplay/epoch-1000"
    --record_video "cartpole_lqn_epoch1000"
~~~~
