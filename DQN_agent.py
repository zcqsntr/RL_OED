import tensorflow as tf
from tensorflow import keras
import numpy as np
import math 

class DQN_agent():

    def __init__(self,layer_sizes = [6,20,20,10]):
        self.memory = []
        self.layer_sizes = layer_sizes
        self.gamma = 0.9
        self.state_size = layer_sizes[0]
        self.n_actions = layer_sizes[-1]
        self.network = self.initialise_network(layer_sizes)
        self.target_network = self.initialise_network(layer_sizes)
        self.buffer = ExperienceBuffer()


    def initialise_network(self, layer_sizes): #YES

        #tf.keras.backend.clear_session()
        initialiser = keras.initializers.RandomUniform(minval = -0.5, maxval = 0.5, seed = None)
        network = keras.Sequential([
            keras.layers.InputLayer([layer_sizes[0]]),
            keras.layers.Dense(layer_sizes[1], activation = tf.nn.relu),
            keras.layers.Dense(layer_sizes[2], activation = tf.nn.relu),
            keras.layers.Dense(layer_sizes[3]) # linear output layer
        ])

        network.compile(optimizer = 'adam', loss = 'mean_squared_error') # TRY DIFFERENT OPTIMISERS
        return network


    def predict(self, state): #YES

        return self.network.predict(state.reshape(1,-1))[0]

    def target_predict(self, state): #YES

        return self.target_network.predict(state.reshape(1,-1))[0]



    def get_inputs_targets(self):
        '''
        gets fitted Q inputs and calculates targets for training the Q-network for episodic training
        '''

        inputs = []
        targets = []



        for transition in self.buffer.sample():
            state, action, reward, next_state = transition # i fnext_state is none, then done
            inputs.append(state)
            # construct target
            values = self.predict(state)

            next_values = self.target_predict(next_state)

            assert len(values) == self.n_actions, 'neural network returning wrong number of values'
            assert len(next_values) == self.n_actions, 'neural network returning wrong number of values'

            #update the value for the taken action using cost function and current Q

            if next_state is None:
                values[action] = reward + self.gamma*np.max(next_values) # could introduce step size here, maybe not needed for neural agent
            else:
                values[action] = reward

            targets.append(values)

        # shuffle inputs and target for IID
        inputs, targets  = np.array(inputs), np.array(targets)


        assert inputs.shape[1] == self.state_size, 'inputs to network wrong size'
        assert targets.shape[1] == self.n_actions, 'targets for network wrong size'
        return inputs, targets

    def Q_update(self, inputs = None, targets = None):
        '''
        Uses a set of inputs and targets to update the Q network
        '''

        if inputs is None and targets is None:
            inputs, targets = self.get_inputs_targets()

        history = self.network.fit(inputs, targets, epochs = 300, verbose = False)
        return history

    def update_target_network(self): # test this
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
            self.network.load_weights(load_path+ '/saved_network.h5') # this requires model to be initialised exactly the same
            self.network.load_weights(load_path+ '/saved_target_network.h5')

    def get_action(self, state, explore_rate):


        if np.random.random() < explore_rate:
            action = np.random.choice(range(self.layer_sizes[-1]))
            # remove this when not debugging
            #values = self.predict(state)
            #self.values.append(values)
        else:
            values = self.predict(state)
            #self.values.append(values)
            action = np.argmax(values)

        return action

    def run_episode():
        return

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



class ExperienceBuffer():
    '''
    Class to handle the management of the QDN storage buffer, stores experience
    in the form [state, action, reward, next_state]
    '''
    def __init__(self, buffer_size = 1000):
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
        if len(transition) != 4:
            raise ValueError("Experience must be length 4, of the for [state, action, reward, state1]")
        if len(self.buffer) == self.buffer_size:
            self.buffer = self.buffer[1:, :]

        transition = np.array(transition).reshape(1,4)

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
        indices = np.random.randint(0, len(self.buffer), size = (batch_size))

        sample = [self.buffer[i] for i in indices]

        return np.array(sample)
