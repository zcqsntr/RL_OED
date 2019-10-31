class DQN_agent():

    def __init__(layer_sizes = [2,20,20,4]):
        self.memory = []
        self.layer_sizes = layer_sizes
        elf.gamma = 0.9
        self.state_size = layer_sizes[0]
        self.n_actions = layer_sizes[-1]
        self.network = self.initialise_network(layer_sizes)
        self.target_network = self.initialise_network(layer_sizes)


    def initialise_network(self, layer_sizes): #YES

        tf.keras.backend.clear_session()
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

    def fit(self, inputs, targets):
        history = self.network.fit(inputs, targets,  epochs = 300, verbose = 0) # TRY DIFFERENT EPOCHS
        return history

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
            self.values.append(values)
            action = np.argmax(values)

    def run_episode():
        return
