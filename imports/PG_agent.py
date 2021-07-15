import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass
#tf.compat.v1.disable_eager_execution()
from tensorflow import keras
import numpy as np
import math
from tensorflow.keras import layers
import copy
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import gc

import tensorflow_probability as tfp
tfd = tfp.distributions




class DRPG_agent():
    def __init__(self, layer_sizes):
        self.memory = []
        self.layer_sizes = layer_sizes
        self.gamma = 1.
        self.state_size = layer_sizes[0]
        self.n_actions = layer_sizes[-1]
        self.value_network = self.initialise_network(layer_sizes)
        self.actor_network = self.initialise_network(layer_sizes)
        self.opt = keras.optimizers.Adam(learning_rate=0.001)
        self.buffer = ExperienceBuffer()
        self.values = []
        self.actions = []

        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.sequences = []
        self.next_sequences = []
        self.all_values = []


    def initialise_network(self, layer_sizes, learning_rate = 0.01):

        '''
        Creates Q network for value function approximation
        '''
        input_size, sequence_size, rec_sizes, hidden_sizes, output_size = layer_sizes


        S_input = keras.Input(shape = (input_size,), name = "S_input")
        sequence_input = keras.Input(shape = (None,sequence_size), name = 'sequence_input')


        rec_out = sequence_input
        for i, rec_size in enumerate(rec_sizes):

            if i == len(rec_sizes) -1:
                rec_out = layers.GRU(rec_size)(rec_out)
            else:
                rec_out = layers.GRU(rec_size, input_shape = (None,sequence_size), return_sequences=True)(rec_out)



        concat = layers.concatenate([S_input, rec_out])

        hl = concat

        for i, hl_size in enumerate(hidden_sizes):
            hl = layers.Dense(hl_size,activation=tf.nn.relu, name = 'hidden_' + str(i))(hl)

        mu = layers.Dense(output_size, name = 'mu')(hl)
        log_std = layers.Dense(output_size, name = 'log_std')(hl)

        network = keras.Model(
            inputs = [S_input, sequence_input],
            outputs = [mu, log_std]
        )
        #keras.utils.plot_model(network, "multi_input_and_output_model.png", show_shapes=True)

        #opt = keras.optimizers.Adam() fitted methods
        #opt = keras.optimizers.Adam(learning_rate=learning_rate) #no nfitted methods
        #network.compile(optimizer=opt, loss='mean_squared_error')

        return network


    def get_actions(self, inputs):
        mu, log_std = self.actor_network.predict(inputs)
        actions = tfp.distributions.Normal(mu, tf.exp(log_std))

        return actions

    def loss(self, inputs, actions, returns):
        # Obtain mu and sigma from actor network
        mu, log_std = self.actor_network(inputs)

        # Compute log probability
        log_probability = self.log_probability(actions, mu, log_std)

        # Compute weighted loss
        loss_actor = - returns * log_probability

        return loss_actor


    def log_probability(self, actions, mu, log_std):


        EPS = 1e-8
        pre_sum = -0.5 * (((actions - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)


    def policy_update(self):
        inputs, actions, returns = self.get_inputs_targets()
        with tf.GradientTape() as tape:
            loss = self.loss(inputs, actions, returns)
            grads = tape.gradient(loss, self.actor_network.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.actor_network.trainable_variables))


    def get_inputs_targets(self):
        '''
        gets fitted Q inputs and calculates targets for training the Q-network for episodic training
        '''

        '''
                gets fitted Q inputs and calculates targets for training the Q-network for episodic training
                '''

        # iterate over all exprienc in memory and create fitted Q targets
        for i, trajectory in enumerate(self.memory):

            e_rewards = []
            sequence = [[0]*self.layer_sizes[1]]
            for j, transition in enumerate(trajectory):
                self.sequences.append(copy.deepcopy(sequence))
                state, action, reward, next_state, done, u = transition
                sequence.append(np.concatenate((state, u/10)))
                #one_hot_a = np.array([int(i == action) for i in range(self.layer_sizes[-1])])/10
                self.next_sequences.append(copy.deepcopy(sequence))
                self.states.append(state)
                self.next_states.append(next_state)
                self.actions.append(action)
                self.rewards.append(reward)
                e_rewards.append(reward)
                self.dones.append(done)


            e_values = [e_rewards[-1]]

            for i in range(2, len(e_rewards) + 1):
                e_values.insert(0, e_rewards[-i] + e_values[0] * self.gamma)
            self.all_values.extend(e_values)

        padded = pad_sequences(self.sequences, maxlen = 11, dtype='float64')
        states = np.array(self.states)
        actions = np.array(self.actions)
        all_values = np.array(self.all_values)

        self.sequences = []
        self.states = []
        self.actions = []
        self.all_values = []
        self.memory = []  # reset memory after this information has been extracted

        randomize = np.arange(len(states))
        np.random.shuffle(randomize)

        states = states[randomize]
        actions = actions[randomize]

        padded = padded[randomize]
        all_values = all_values[randomize]

        inputs = [states, padded]

        return inputs, actions, all_values