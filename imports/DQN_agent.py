import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
import numpy as np
import math
from tensorflow.keras import layers
import copy
from keras.preprocessing.sequence import pad_sequences
import time

def OneHot(input_dim=None, input_length=None):
    # Check if inputs were supplied correctly
    if input_dim is None or input_length is None:
        raise TypeError("input_dim or input_length is not set")

    # Helper method (not inlined for clarity)
    def _one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'),
                          num_classes=num_classes)

    # Final layer representation as a Lambda layer
    return Lambda(_one_hot,
                  arguments={'num_classes': input_dim},
                  input_shape=(input_length,))

class DQN_agent():

    def __init__(self,layer_sizes ):
        self.memory = []
        self.layer_sizes = layer_sizes
        self.gamma = 1.
        self.state_size = layer_sizes[0]
        self.n_actions = layer_sizes[-1]
        self.network = self.initialise_network(layer_sizes)
        self.target_network = self.initialise_network(layer_sizes)
        self.buffer = ExperienceBuffer()
        self.values = []
        self.actions = []


    def initialise_network(self, layer_sizes):

        '''
        Creates Q network for value function approximation
        '''


        regulariser = keras.regularizers.l1_l2(l1=1e-8, l2=1e-7)
        network = keras.Sequential()
        network.add(keras.layers.InputLayer([layer_sizes[0]]))

        for l in layer_sizes[1:-1]:
            network.add(keras.layers.Dense(l, activation=tf.nn.relu, kernel_regularizer=regulariser))
        network.add(keras.layers.Dense(layer_sizes[-1], kernel_regularizer=regulariser))  # linear output layer

        opt = keras.optimizers.Adam(learning_rate=1e-5)
        #opt = keras.optimizers.Adam()
        #opt = tf.keras.optimizers.SGD(learning_rate=0.1)
        #opt = tf.keras.optimizers.RMSprop()
        keras.utils.plot_model(network, "multi_input_and_output_model.png", show_shapes=True)
        network.compile(optimizer=opt, loss='mean_squared_error')  # TRY DIFFERENT OPTIMISERS
        # try clipnorm=1
        return network


    def predict(self, states): #YES

        return self.network.predict(states.reshape(-1,self.layer_sizes[0]))

    def target_predict(self, states): #YES

        return self.target_network.predict(states.reshape(-1,self.layer_sizes[0]))

    def get_inputs_targets(self):
        '''
        gets fitted Q inputs and calculates targets for training the Q-network for episodic training
        '''

        states = []

        next_states = []
        actions = []
        rewards = []
        dones = []
        sample = self.sample(320, 50000)

        for transition in sample:  # could make this faster

            state, action, reward, next_state, done= transition  # i fnext_state is none, then done

            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)




        states = np.array(states)
        next_states = np.array(next_states, dtype = np.float64)
        actions = np.array(actions)
        rewards = np.array(rewards)
        #print('ns:', next_states)


        # construct targe
        values = self.predict(states)
        next_values = self.target_predict(next_states)



        for i in range(len(next_states)):
            #print(actions[i], rewards[i])
            if dones[i]:

                values[i, actions[i]] = rewards[i]
            else:

                values[i, actions[i]] = rewards[i]  + self.gamma * np.max(next_values[i])

        return states, values

    def get_inputs_targets_MC(self, alpha=1):
        '''
        gets fitted Q inputs and calculates targets for training the Q-network for episodic training
        '''

        '''
                gets fitted Q inputs and calculates targets for training the Q-network for episodic training
                '''
        targets = []
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        all_values = []
        #sample = self.sample(32, 50000)
        sample = self.memory[-100:]
        # iterate over all exprienc in memory and create fitted Q targets


        for trajectory in sample:

            e_rewards = []
            for transition in trajectory:

                state, action, reward, next_state, done = transition
                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                e_rewards.append(reward)
                dones.append(done)

            e_values = [e_rewards[-1]]

            for i in range(2, len(e_rewards) + 1):
                e_values.insert(0, e_rewards[-i] + e_values[0] * self.gamma)
            all_values.extend(e_values)

        states = np.array(states)
        next_states = np.array(next_states, dtype=np.float64)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # construct target
        values = self.predict(states)



        # update the value for the taken action using cost function and current Q
        for i in range(len(next_states)):
            # print(actions[i], rewards[i])
            #print('-------------------')
            #print(values[i, actions[i]])
            #print(all_values[i])
            values[i, actions[i]] = (1-alpha )*values[i, actions[i]] + alpha * all_values[i]

            #values[i, actions[i]] = values[i, actions[i]] + alpha*(all_values[i] - values[i, actions[i]])

            #print(values[i, actions[i]])
            #print()
        # shuffle inputs and target for IID
        inputs, targets = np.array(states), np.array(values)

        randomize = np.arange(len(inputs))
        np.random.shuffle(randomize)
        inputs = inputs[randomize]
        targets = targets[randomize]

        if np.isnan(targets).any():
            print('NAN IN TARGETS!')

        return inputs, targets


    def get_inputs_targets_old(self):
        

        

        inputs = []
        targets = []


        #for transition in self.buffer.sample(): # could make this faster
        for transition in self.buffer.buffer[0:10]:  # could make this faster
            state, action, reward, next_state = transition # i fnext_state is none, then done
            print(action, reward)
            inputs.append(state)
            # construct targe
            values = self.predict(np.array(state))[0]



            assert len(values) == self.n_actions, 'neural network returning wrong number of values'


            #update the value for the taken action using cost function and current Q

            if next_state is None:
                values[action] = reward
            else:
                next_values = self.target_predict(np.array(next_state))[0]
                assert len(next_values) == self.n_actions, 'neural network returning wrong number of values'
                values[action] = reward + self.gamma * np.max( next_values)  # could introduce step size here, maybe not needed for neural agent

            targets.append(values)

        # shuffle inputs and target for IID

        inputs, targets  = np.array(inputs), np.array(targets)


        assert inputs.shape[1] == self.state_size, 'inputs to network wrong size'
        assert targets.shape[1] == self.n_actions, 'targets for network wrong size'
        return inputs, targets

    def Q_update(self, inputs = None, targets = None, alpha = 1):
        '''
        Uses a set of inputs and targets to update the Q network
        '''


        if inputs is None and targets is None:
            inputs, targets = self.get_inputs_targets_MC(alpha)
            #print(inputs, targets)

            #inputs_old, targets_old = self.get_inputs_targets_old()
            #print(inputs ==inputs_old)
            #print(np.isclose(targets, targets_old))
        #print('inputs: ', inputs)
        #print('target: ', targets)
        #print('target old: ', targets_old)


        history = self.network.fit(inputs, targets, epochs = 1, verbose = True)
        return history



    def update_target_network(self): # tested

        self.target_network.set_weights(self.network.get_weights())

    def save_network(self, save_path): # tested
        #print(self.network.layers[1].get_weights())
        self.network.save(save_path + '/saved_network.h5')
        self.target_network.save(save_path + '/saved_target_network.h5')

    def load_network(self, load_path): #tested
        try:
            self.network = keras.models.load_model(load_path + '/saved_network.h5') # sometimes this crashes, apparently a bug in keras
            self.target_network = keras.models.load_model(load_path + '/saved_target_network.h5')
        except:
            print('EXCEPTION IN LOAD NETWORK')
            self.network.load_weights(load_path+ '/saved_network.h5') # this requires model to be initialised exactly the same
            self.target_network.load_weights(load_path+ '/saved_target_network.h5')

    def get_action(self, state, explore_rate):


        if np.random.random() < explore_rate:
            action = np.random.choice(range(self.layer_sizes[-1]))


        else:
            values = self.predict(state)
            self.values.append(values)
            print('values:', values.shape)
            action = np.argmax(values)
            self.actions.append(action)
        return action

    def get_rate(self, episode, MIN_RATE,  MAX_RATE, denominator):
        '''
        Calculates the logarithmically decreasing explore or learning rate

        Parameters:
            episode: the current episode
            MIN_LEARNING_RATE: the minimum possible step size
            MAX_LEARNING_RATE: maximum step size
            denominator: controls the rate of decay of the step size
        Returns:
            step_size: the Q-learning step size
        '''

        # input validation
        if not 0 <= MIN_RATE <= 1:
            raise ValueError("MIN_LEARNING_RATE needs to be bewteen 0 and 1")

        if not 0 <= MAX_RATE <= 1:
            raise ValueError("MAX_LEARNING_RATE needs to be bewteen 0 and 1")

        if not 0 < denominator:
            raise ValueError("denominator needs to be above 0")

        rate = max(MIN_RATE, min(MAX_RATE, 1.0 - math.log10((episode+1)/denominator)))

        return rate

    def get_actions(self, states, explore_rate):
        '''
        PARALLEL version of get action
        Choses action based on enivormental state, explore rate and current value estimates

        Parameters:
            state: environmental state
            explore_rate
        Returns:
            action
        '''
        rng = np.random.random(len(states))

        explore_inds = np.where(rng < explore_rate)[0]

        exploit_inds = np.where(rng >= explore_rate)[0]

        explore_actions = np.random.choice(range(self.layer_sizes[-1]), len(explore_inds))
        actions = np.zeros((len(states)), dtype=np.int32)

        if len(exploit_inds) > 0:
            values = self.predict(np.array(states)[exploit_inds])
            print(values.shape)

            if np.isnan(values).any():
                print('NAN IN VALUES!')
                print('states that gave nan:', states)
            self.values.extend(values)


            exploit_actions = np.argmax(values, axis = 1)
            actions[exploit_inds] = exploit_actions


        actions[explore_inds] = explore_actions
        self.actions.extend(actions)
        return actions

    def sample(self, batch_size = 32, memory_size = 10000):
        '''
        Randomly samples the experience buffer

        Parameters:
            batch_size: the number of experience traces to sample
        Returns:
            sample: the sampled experience
        '''


        # start of experience traces
        indices = np.random.randint(max(0, len(self.memory) - memory_size), len(self.memory), size = (batch_size))

        sample = [self.memory[i] for i in indices]

        return np.array(sample)

class DRQN_agent(DQN_agent):
    def __init__(self,layer_sizes ):
        self.memory = []
        self.layer_sizes = layer_sizes
        self.gamma = 1.
        self.state_size = layer_sizes[0]
        self.n_actions = layer_sizes[-1]
        self.network = self.initialise_network(layer_sizes)
        self.target_network = self.initialise_network(layer_sizes)
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



    def initialise_network(self, layer_sizes):

        '''
        Creates Q network for value function approximation
        '''

        initialiser = keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)
        lstm_size = layer_sizes[2]

        S_input = keras.Input(shape = (self.layer_sizes[0],), name = "S_input")
        sequence_input = keras.Input(shape = (None,self.layer_sizes[1]), name = 'sequence_input')

        #lstm_out = layers.LSTM(lstm_size, input_shape = (None,self.layer_sizes[1]), return_sequences=True)(sequence_input)

        lstm_out = layers.GRU(lstm_size)(sequence_input)

        concat = layers.concatenate([S_input, lstm_out])

        hl = concat

        for i, hl_size in enumerate(layer_sizes[3:-1]):
            hl = layers.Dense(hl_size,activation=tf.nn.relu, name = 'hidden_' + str(i))(hl)

        out = layers.Dense(layer_sizes[-1], name = 'output')(hl)

        network = keras.Model(
            inputs = [S_input, sequence_input],
            outputs = [out]
        )
        #keras.utils.plot_model(network, "multi_input_and_output_model.png", show_shapes=True)

        opt = keras.optimizers.Adam()
        network.compile(optimizer=opt, loss='mean_squared_error')

        return network


    def get_inputs_targets(self, alpha=1, fitted_q = False, monte_carlo = False):
        '''
        gets fitted Q inputs and calculates targets for training the Q-network for episodic training
        '''

        '''
                gets fitted Q inputs and calculates targets for training the Q-network for episodic training
                '''


        all_values = []

        if fitted_q:
            sample = self.memory
        else:
            sample = self.sample(32, 50000)
        #sample = self.memory[-100:]
        # iterate over all exprienc in memory and create fitted Q targets



        t = time.time()
        for i, trajectory in enumerate(sample):

            e_rewards = []
            sequence = [[0]*self.layer_sizes[1]]
            next_sequence = [[0]*self.layer_sizes[1]]

            for j, transition in enumerate(trajectory):


                if j > 0: # this needs to be one behind

                    #one_hot_a = np.array([int(i == action) for i in range(self.layer_sizes[-1])])/10



                    sequence.append(np.concatenate((state, u/10)))



                self.sequences.append(copy.deepcopy(sequence))
                state, action, reward, next_state, done, u = transition

                #one_hot_a = np.array([int(i == action) for i in range(self.layer_sizes[-1])])/10


                next_sequence.append(np.concatenate((state, u/10)))


                self.next_sequences.append(copy.deepcopy(next_sequence))


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
                self.all_values.extend(e_values)
        print('sequence time', time.time() -t)

        self.memory = [] # reset memory after this information has been extracted

        padded = pad_sequences(self.sequences, maxlen = 11)
        next_padded = pad_sequences(self.next_sequences, maxlen = 11)
        states = np.array(self.states)

        next_states = np.array(self.next_states, dtype=np.float64)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = self.dones

        # construct target
        print(len(sample))
        print(states.shape, padded.shape)
        print(next_states.shape, next_padded.shape)
        t = time.time()
        values = self.predict([states, padded])
        print(values.shape)

        if monte_carlo:
            all_values = self.all_values
        else:

            all_values = self.predict([next_states, next_padded])

        print('values time', time.time() - t)


        t = time.time()
        # update the value for the taken action using cost function and current Q
        for i in range(len(next_states)):
            # print(actions[i], rewards[i])
            #print('-------------------')
            #print(values[i, actions[i]])
            #print(all_values[i])


            if monte_carlo:
                values[i, actions[i]] = (1-alpha )*values[i, actions[i]] + alpha * all_values[i]
            else:
                #print(rewards[i], self.gamma *all_values[i, actions[i]])



                if dones[i]:
                    values[i, actions[i]] = (1 - alpha) * values[i, actions[i]] + alpha*rewards[i]

                else:
                    values[i, actions[i]] = (1 - alpha) * values[i, actions[i]] + alpha *(rewards[i] + self.gamma * np.max(all_values[i])) #Q learning
                    #values[i, actions[i]] = (1 - alpha) * values[i, actions[i]] + alpha *(rewards[i] + self.gamma *all_values[i, actions[i]]) #SARSA


            #print(values[i, actions[i]])
            #print()
        # shuffle inputs and target for IID
        print('targets time:', time.time()-t)

        randomize = np.arange(len(states))
        np.random.shuffle(randomize)

        states = states[randomize]

        padded = padded[randomize]
        values = values[randomize]


        inputs = [states, padded]
        targets = values
        if np.isnan(targets).any():
            print('NAN IN TARGETS!')

        return inputs, targets



    def predict(self, inputs):

        return self.network.predict({'S_input': inputs[0], 'sequence_input':inputs[1]})

    def Q_update(self, inputs = None, targets = None, alpha = 1, fitted_q = False, verbose = True, monte_carlo = False):
        '''
        Uses a set of inputs and targets to update the Q network
        '''


        if inputs is None and targets is None:
            inputs, targets = self.get_inputs_targets(alpha =alpha, fitted_q=fitted_q, monte_carlo=monte_carlo)
            #print(inputs, targets)

            #inputs_old, targets_old = self.get_inputs_targets_old()
            #print(inputs ==inputs_old)
            #print(np.isclose(targets, targets_old))
        #print('inputs: ', inputs)
        #print('target: ', targets)
        #print('target old: ', targets_old)

        if fitted_q:
            epochs = 500
            batch_size = 256
            self.reset_weights()
            callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss')
            callbacks = [callback]
        else:
            epochs = 1
            batch_size = 32
            callbacks = []
        t = time.time()
        history = self.network.fit({'S_input': inputs[0], 'sequence_input':inputs[1]}, targets, epochs = epochs, verbose = verbose, validation_split =0.1, batch_size=batch_size, callbacks = callbacks)
        print('fit time:', time.time()-t)
        return history

    def get_actions(self, inputs, explore_rate):
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

        rng = np.random.random(len(states))

        explore_inds = np.where(rng < explore_rate)[0]

        exploit_inds = np.where(rng >= explore_rate)[0]

        explore_actions = np.random.choice(range(self.layer_sizes[-1]), len(explore_inds))
        actions = np.zeros((len(states)), dtype=np.int32)

        if len(exploit_inds) > 0:

            sequences = pad_sequences(sequences, maxlen=11)
            values = self.predict([np.array(states)[exploit_inds], np.array(sequences)[exploit_inds]])


            if np.isnan(values).any():
                print('NAN IN VALUES!')
                print('states that gave nan:', states)
            self.values.extend(values)


            exploit_actions = np.argmax(values, axis = 1)
            actions[exploit_inds] = exploit_actions


        actions[explore_inds] = explore_actions
        self.actions.extend(actions)
        return actions

    def reset_weights(self):
        '''
        Reinitialises weights to random values
        '''
        #sess = tf.keras.backend.get_session()
        #sess.run(tf.global_variables_initializer())

        self.network = self.initialise_network(self.layer_sizes)


class ExperienceBuffer():
    '''
    Class to handle the management of the QDN storage buffer, stores experience
    in the form [state, action, reward, next_state]
    '''
    def __init__(self, buffer_size = 12000):
        '''
        Parameters:

            buffer_size: number of experiences that can be stored
        '''

        # input validation
        if buffer_size <= 0 or not isinstance(buffer_size, int):
            raise ValueError("Buffer size must be a positive integer")

        # initialisation
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, transition):
        '''
        Adds a peice of experience to the buffer and removes the oldest experince if the buffer is full

        Parameters:
            experience: the new experience to be added, in the format [state, action, reward, state1]
        '''

        # input validation
        if len(transition) != 5:
            raise ValueError("Experience must be length 5, of the for [state, action, reward, state1, done]")
        if len(self.buffer) == self.buffer_size:
            self.buffer = self.buffer[1:, :]

        transition = np.array(transition).reshape(1,5)

        if self.buffer == []:
            self.buffer = transition
        else:
            self.buffer = np.append(self.buffer, transition, axis = 0)



    def sample(self, batch_size = 32):
        '''
        Randomly samples the experience buffer

        Parameters:
            batch_size: the number of experience traces to sample
        Returns:
            sample: the sampled experience
        '''


        # start of experience traces
        indices = np.random.randint(0, len(self.memory), size = (batch_size))

        sample = [self.buffer[i] for i in indices]

        return np.array(sample)
