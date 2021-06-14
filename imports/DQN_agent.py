import tensorflow as tf
from tensorflow import keras
import numpy as np
import math

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
