# package imports
from collections import deque
# package implements
from ..quantum.PQC import *
from ..utils.state_decoder import *

class DeepQLearning:
    def __init__(self, gamma:float = 0.99, n_episodes: int = 2000, max_memory_length: int = 10000, epsilon_start:float = 0.01, epsilon: float = 0.01, epsilon_min: float = 0.01, epsilon_max: float = 1.0, batch_size: int = 16, steps_per_update: int = 10, steps_per_target_update: int = 30, learning_rate_in: float = 0.001, learning_rate_var: float = 0.001, learning_rate_out: float = 0.1):
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.max_memory_length = max_memory_length
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.batch_size = batch_size
        self.steps_per_update = steps_per_update
        self.steps_per_target_update = steps_per_target_update
        self.learning_rate_in = learning_rate_in
        self.learning_rate_var = learning_rate_var
        self.learning_rate_out = learning_rate_out
        
        self.optimizer_in = tf.keras.optimizers.Adam(learning_rate=learning_rate_in, amsgrad=True)
        self.optimizer_var = tf.keras.optimizers.Adam(learning_rate=learning_rate_var, amsgrad=True)
        self.optimizer_out = tf.keras.optimizers.Adam(learning_rate=learning_rate_out, amsgrad=True)
        
        self.replay_memory = deque(maxlen=10000)
        # Assign the model parameters to each optimizer
        self.w_in, self.w_var, self.w_out = 1, 0, 2 
        
        
    # ## PARAMS ##
    # gamma = 0.99
    # n_episodes = 2000

    # # Define replay memory
    # max_memory_length = 10000 # Maximum replay length
    # replay_memory=deque(maxlen=max_memory_length)

    # epsilon_start = 0.0  # Initial epsilon greedy parameter
    # epsilon = 0.01  # Epsilon greedy parameter
    # epsilon_min = 0.01  # Minimum epsilon greedy parameter
    # epsilon_max = 1.0  # Maximum epsilon greedy parameter
    # batch_size = 16
    # steps_per_update = 10 # Train the model every x steps
    # steps_per_target_update = 30 # Update the target model every x steps
    
    # learning_rate_in = 0.001
    # learning_rate_var = 0.001
    # learning_rate_out = 0.1

    ## PARAMS ##

    class Rescaling(tf.keras.layers.Layer):
            """
            A custom layer for rescaling inputs.

            Args:
                input_dim (int): The dimension of the input.

            Attributes:
                input_dim (int): The dimension of the input.
                w (tf.Variable): The weight variable used for rescaling.

            Methods:
                call(inputs): Rescales the inputs using the weight variable.

            """

            def __init__(self, input_dim):
                super(DeepQLearning.Rescaling, self).__init__()
                self.input_dim = input_dim
                self.w = tf.Variable(
                    initial_value=tf.ones(shape=(1,input_dim)), dtype="float32",
                    trainable=True, name="obs-weights")

            def call(self, inputs):
                """
                Rescales the inputs using the weight variable.

                Args:
                    inputs (tf.Tensor): The input tensor.

                Returns:
                    tf.Tensor: The rescaled inputs.

                """
                return tf.math.multiply((inputs+1)/2, tf.repeat(self.w,repeats=tf.shape(inputs)[0],axis=0))

    def generate_model_Qlearning(self, qubits, n_layers: int, n_actions: int, observables, target):
        """Generates a Keras model for a data re-uploading PQC Q-function approximator."""

        input_tensor = tf.keras.Input(shape=(len(qubits), ), dtype=tf.dtypes.float32, name='input')
        re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables, activation='tanh')([input_tensor])
        process = tf.keras.Sequential([DeepQLearning.Rescaling(len(observables))], name=target*"Target"+"Q-values")
        Q_values = process(re_uploading_pqc)
        model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)

        return model

    # def interact_gym_env(self, state, model, epsilon, n_actions, env):

    #     state_array = []

    #     # Preprocess state
    #     # if atari:
    #     #     state = extract_state(state)
    #     # else:
    #     state_array = np.array(state) 
    #     state = tf.convert_to_tensor([state_array])

    #     # Sample action
    #     coin = np.random.random()
    #     if coin < epsilon:
    #         # random action
    #         action = np.random.choice(n_actions)
    #     else:
    #         # greedy action
    #         q_vals = model([state])
    #         action = int(tf.argmax(q_vals[0]).numpy())

    #     # Apply sampled action in the environment, receive reward and next state
    #     next_state, reward, done, _ = env.step(action)

    #     # if atari:
    #     #     next_state = extract_state(next_state)
            
    #     interaction = {'state': state_array, 'action': action, 'next_state': next_state.copy(),
    #                 'reward': reward, 'done':np.float32(done)}

    #     return interaction
    
    def interact_env(self, state, model, epsilon, n_actions, env):
        state_array = []

        # Preprocess state
        state_array = np.array(state) 
        state = tf.convert_to_tensor([state_array])

        # Sample action
        coin = np.random.random()
        if coin < epsilon:
            # random action
            action = np.random.choice(n_actions)
        else:
            # greedy action
            q_vals = model([state])
            action = int(tf.argmax(q_vals[0]).numpy())

        # Apply sampled action in the environment, receive reward and next state
        next_state, reward, done, _ = env.step(action)

        # save the interaction
        interaction = {'state': state_array, 'action': action, 'next_state': next_state.copy(),
                    'reward': reward, 'done':np.float32(done)}

        return interaction

    @tf.function
    def Q_learning_update(self, states, actions, rewards, next_states, done, model, gamma, n_actions, model_target):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        done = tf.convert_to_tensor(done)

        # Compute their target q_values and the masks on sampled actions
        future_rewards = model_target([next_states])
        target_q_values = rewards + (gamma * tf.reduce_max(future_rewards, axis=1)
                                                    * (1.0 - done))
        masks = tf.one_hot(actions, n_actions)

        # Train the model on the states and target Q-values
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            q_values = model([states])
            q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = tf.keras.losses.Huber()(target_q_values, q_values_masked)

        # Backpropagation
        grads = tape.gradient(loss, model.trainable_variables)
        for optimizer, w in zip([self.optimizer_in, self.optimizer_var, self.optimizer_out], [self.w_in, self.w_var, self.w_out]):
            optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])
