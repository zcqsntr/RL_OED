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
import tensorflow.keras.initializers as initializers


class DRPG_agent():
    def __init__(self, layer_sizes, learning_rate = 0.001, critic=True):
        self.memory = []
        self.layer_sizes = layer_sizes
        self.gamma = 1.

        self.critic = critic
        if critic:
            self.critic_network = self.initialise_network(layer_sizes, critic_nw=True)
            self.critic_network.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                                        loss='mean_squared_error')
        self.actor_network = self.initialise_network(layer_sizes)
        self.opt = keras.optimizers.Adam(learning_rate=learning_rate)

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


    def initialise_network(self, layer_sizes, critic_nw = False):

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


        if critic_nw:

            values = layers.Dense(1, name='mu')(hl)


            network = keras.Model(
                inputs=[S_input, sequence_input],
                outputs=[values]
            )
        else:
            mu = layers.Dense(output_size, name = 'mu', activation=tf.nn.sigmoid)(hl)
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

        states, sequences = inputs


        sequences = pad_sequences(sequences, maxlen=11, dtype='float64')


        mu, log_std = self.actor_network.predict([np.array(states), sequences])

        print('mu log_std',mu[0], log_std[0])


        actions = mu + tf.multiply(tf.random.normal(tf.shape(mu)), tf.exp(log_std))
        #print('actions',actions[0])

        return actions

    def loss(self, inputs, actions, returns):
        # Obtain mu and sigma from actor network
        mu, log_std = self.actor_network(inputs)

        # Compute log probability
        log_probability = self.log_probability(actions, mu, log_std)
        print('log probability', log_probability.shape)
        print('returns:', returns.shape)
        # Compute weighted loss
        loss_actor = - tf.reduce_mean(tf.multiply(returns, log_probability))
        print('loss actor', loss_actor.shape)
        return loss_actor


    def log_probability(self, actions, mu, log_std):


        EPS = 1e-8
        pre_sum = -0.5 * (((actions - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))

        print('pre sum', pre_sum.shape)
        return tf.reduce_sum(pre_sum, axis=1)


    def policy_update(self):

        inputs, actions, returns = self.get_inputs_targets()

        print(returns.shape)
        if self.critic:

            expected_returns = self.critic_network.predict(inputs)

            returns -= expected_returns.reshape(-1)
            print(expected_returns.reshape(-1).shape)
            self.critic_network.fit(inputs, returns, epochs = 1)

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
                sequence.append(np.concatenate((state, u/1)))
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
        print('inputs, actions, all_values', inputs[0].shape, inputs[1].shape, actions.shape, all_values.shape)
        return inputs, actions, all_values


    def save_network(self, save_path): # tested
        #print(self.network.layers[1].get_weights())
        self.actor_network.save(save_path + '/saved_network.h5')

    def load_network(self, load_path): #tested
        try:
            self.actor_network = keras.models.load_model(load_path + '/saved_network.h5') # sometimes this crashes, apparently a bug in keras

        except:
            print('EXCEPTION IN LOAD NETWORK')
            self.actor_network.load_weights(load_path+ '/saved_network.h5') # this requires model to be initialised exactly the same



class DDPG_agent():
    def __init__(self, val_layer_sizes, pol_layer_sizes, val_learning_rate = 0.01, pol_learning_rate = 0.001):
        self.layer_sizes = pol_layer_sizes
        self.val_layer_sizes = val_layer_sizes
        self.val_learning_rate = val_learning_rate
        self.memory = []
        self.gamma = 1.
        self.policy_network = self.initialise_network(pol_layer_sizes, out_act = tf.nn.tanh, scale = 1)
        self.policy_opt = keras.optimizers.Adam(learning_rate=pol_learning_rate)

        # Q(s,a) = value

        self.Q_network = self.initialise_network(val_layer_sizes)
        opt = keras.optimizers.Adam(learning_rate=val_learning_rate)  # no nfitted methods
        self.Q_network.compile(optimizer=opt, loss='mean_squared_error')

        self.values = []
        self.actions = []
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.sequences = []
        self.next_sequences = []
        self.all_returns = []

    def initialise_network(self, layer_sizes, out_act = tf.keras.activations.linear, scale = 1.):

        '''
        Creates Q network for value function approximation
        '''
        input_size, sequence_size, rec_sizes, hidden_sizes, output_size = layer_sizes

        S_input = keras.Input(shape = (input_size,), name = "S_input")

        if sequence_size != 0:
            sequence_input = keras.Input(shape = (None,sequence_size), name = 'sequence_input')

            rec_out = sequence_input
            for i, rec_size in enumerate(rec_sizes):

                if i == len(rec_sizes) -1:
                    rec_out = layers.GRU(rec_size)(rec_out)
                else:
                    rec_out = layers.GRU(rec_size, input_shape = (None,sequence_size), return_sequences=True)(rec_out)

            concat = layers.concatenate([S_input, rec_out])
        else:
            concat = S_input

        hl = concat

        for i, hl_size in enumerate(hidden_sizes):
            hl = layers.Dense(hl_size,activation=tf.nn.relu, name = 'hidden_' + str(i))(hl)

        out =  layers.Dense(layer_sizes[-1], name='mu', activation=out_act)(hl)
        out = layers.Lambda(lambda x: x*scale)(out)

        if sequence_size != 0:
            network = keras.Model(
                inputs=[S_input, sequence_input],
                outputs=[out]
            )
        else:
            network = keras.Model(
                inputs=[S_input],
                outputs=[out]
            )


        return network


    def get_actions(self, inputs, explore_rate, test_episode = False):
        '''
        PARALLEL version of get action
        Choses action based on enivormental state, explore rate and current value estimates

        Parameters:
            state: environmental state
            explore_rate
        Returns:
            action
        '''

        states, sequences = inputs

        sequences = pad_sequences(sequences, maxlen=11, dtype='float64')
        actions = self.policy_network([np.array(states), np.array(sequences)])
        actions = np.array(actions)
        if test_episode:
            actions[:-1] += np.random.normal(0, explore_rate*0.1,size = actions[:-1].shape)

        else:
            actions += np.random.normal(0, explore_rate*0.1, size=actions.shape)


        actions = np.clip(actions, 0, 0.1)
        self.actions.extend(actions)

        return actions

    def get_action(self, s, explore_rate):
        a = self.policy_network(s) \

        a += np.random.normal(0, explore_rate*1,size = a.shape)
        a = np.clip(a, -1, 1)
        return a



    def Q_update(self, recurrent = True, policy = True, fitted = True):

        inputs, actions, targets = self.get_inputs_targets(recurrent = recurrent)
        if recurrent:
            states, sequences = inputs
        else:
            states = inputs[0]


        if fitted:
            epochs = 500
            batch_size = 256

            self.reset_weights()
            callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience=1, restore_best_weights=True)
            callbacks = [callback]
        else:
            epochs = 1
            batch_size = 256

            callbacks = []

        if recurrent:
            history = self.Q_network.fit([tf.concat((states, actions), 1), sequences], targets, epochs = epochs, verbose = True, validation_split =0.1, batch_size=batch_size, callbacks = callbacks)
        else:
            history = self.Q_network.fit([tf.concat((states, actions), 1)], targets, epochs = epochs, verbose = False, validation_split =0.1, batch_size=batch_size, callbacks = callbacks)

        if policy:
            batches = math.ceil(states.shape[0]/batch_size)

            epoch_losses = []
            for epoch in range(epochs):
                batch_losses = []
                for batch in range(batches):

                    start = batch*batch_size
                    end = start + batch_size

                    with tf.GradientTape() as tape:
                        pred_actions = self.policy_network([states[start:end], sequences[start:end]]) if recurrent else self.policy_network([states[start:end]])


                        pred_values = self.Q_network([tf.concat((states[start:end], pred_actions), 1), sequences[start:end]]) if recurrent else self.Q_network([tf.concat((states[start:end], pred_actions), 1)])


                        # Used `-value` as we want to maximize the value given
                        # by the critic for our actions
                        loss = -tf.math.reduce_mean(pred_values)

                        policy_grad = tape.gradient(loss, self.policy_network.trainable_variables)

                        self.policy_opt.apply_gradients(zip(policy_grad, self.policy_network.trainable_variables))

                        batch_losses.append(loss)

                epoch_losses.append(np.mean(batch_losses))

                if len(epoch_losses) > 2 and epoch_losses[-1] - epoch_losses[-2] >=0:
                    print('epochs: ', len(epoch_losses), epoch_losses[0], epoch_losses[-1])
                    break

    def get_inputs_targets(self, recurrent = True, monte_carlo = False):
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
                state, action, reward, next_state, done = transition

                sequence.append(np.concatenate((state, action)))
                #one_hot_a = np.array([int(i == action) for i in range(self.layer_sizes[-1])])/10
                self.next_sequences.append(copy.deepcopy(sequence))
                self.states.append(state)
                self.next_states.append(next_state)
                self.actions.append(action)
                self.rewards.append(reward)
                e_rewards.append(reward)
                self.dones.append(done)

            if monte_carlo:
                e_values = [e_rewards[-1]]

                for i in range(2, len(e_rewards) + 1):
                    e_values.insert(0, e_rewards[-i] + e_values[0] * self.gamma)
                self.all_returns.extend(e_values)


        if recurrent:
            padded = pad_sequences(self.sequences, maxlen = 11, dtype='float64')
            next_padded = pad_sequences(self.next_sequences, maxlen=11, dtype='float64')


        next_states = np.array(self.next_states, dtype=np.float64)
        rewards = np.array(self.rewards).reshape(-1, 1)
        dones = np.array(self.dones).reshape(-1, 1)
        states = np.array(self.states)
        actions = np.array(self.actions)
        all_returns = np.array(self.all_returns)

        self.memory = []  # reset memory after this information has been extracted


        if monte_carlo : # only take last experiences
            batch_size = self.batch_size
            if states.shape[0] > batch_size:
                states = states[-batch_size:]
                padded = padded[-batch_size:]
                next_padded = next_padded[-batch_size:]
                next_states = next_states[-batch_size:]
                actions = actions[-batch_size:]
                rewards = rewards[-batch_size:]
                dones = dones[-batch_size:]
                all_returns = all_returns[-batch_size:]

        #values = self.predict([states, padded])

        if monte_carlo:
            targets = all_returns
        else:
            next_actions = self.policy_network([next_states, next_padded]) if recurrent else self.policy_network([next_states])

            print('next_actions:', next_actions.shape)
            next_values = self.Q_network.predict([tf.concat((states, next_actions), 1), next_padded]) if recurrent else self.Q_network.predict([tf.concat((states, next_actions), 1)])

            print('next values', next_values.shape)
            targets = rewards + self.gamma*(1-dones)*next_values
            print('rewards', rewards.shape)
            print('targets:', targets.shape)
            print((self.gamma*(1-dones)).shape)

        randomize = np.arange(len(states))
        np.random.shuffle(randomize)

        states = states[randomize]
        actions = actions[randomize]

        if recurrent: padded = padded[randomize]

        targets = targets[randomize]

        inputs = [states, padded] if recurrent else [states]
        print('inputs, actions, targets', inputs[0].shape, actions.shape, targets.shape)
        return inputs, actions, targets


    def save_network(self, save_path): # tested
        #print(self.network.layers[1].get_weights())
        self.actor_network.save(save_path + '/saved_network.h5')

    def load_network(self, load_path): #tested
        try:
            self.actor_network = keras.models.load_model(load_path + '/saved_network.h5') # sometimes this crashes, apparently a bug in keras

        except:
            print('EXCEPTION IN LOAD NETWORK')
            self.actor_network.load_weights(load_path+ '/saved_network.h5') # this requires model to be initialised exactly the same


    def reset_weights(self):
        '''
        Reinitialises weights to random values
        '''
        #sess = tf.keras.backend.get_session()
        #sess.run(tf.global_variables_initializer())
        del self.Q_network
        del self.policy_network
        gc.collect()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        self.Q_network = self.initialise_network(self.val_layer_sizes)
        opt = keras.optimizers.Adam(learning_rate=self.val_learning_rate)  # no nfitted methods
        self.Q_network.compile(optimizer=opt, loss='mean_squared_error')
        self.policy_network = self.initialise_network(self.layer_sizes, out_act=tf.nn.sigmoid, scale=0.1)