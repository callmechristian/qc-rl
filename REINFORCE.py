from PQC import *
import gym
from collections import defaultdict

class REINFORCE:
    ## PARAMS ##
    gamma = 1

    optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
    optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
    optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

    w_in, w_var, w_out = 1, 0, 2 # Assign the model parameters to each optimizer
    ## PARAMS ##

    class Alternating(tf.keras.layers.Layer):
        def __init__(self, output_dim):
            super(REINFORCE.Alternating, self).__init__()
            self.w = tf.Variable(
                initial_value=tf.constant([[(-1.)**i for i in range(output_dim)]]), dtype="float32",
                trainable=True, name="obs-weights")

        def call(self, inputs):
            return tf.matmul(inputs, self.w)

    def generate_model_policy(qubits, n_layers, n_actions, beta, observables):
        """Generates a Keras model for a data re-uploading PQC policy."""

        input_tensor = tf.keras.Input(shape=(len(qubits), ), dtype=tf.dtypes.float32, name='input')
        re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables)([input_tensor])
        process = tf.keras.Sequential([
            REINFORCE.Alternating(n_actions),
            tf.keras.layers.Lambda(lambda x: x * beta),
            tf.keras.layers.Softmax()
        ], name="observables-policy")
        policy = process(re_uploading_pqc)
        model = tf.keras.Model(inputs=[input_tensor], outputs=policy)

        return model

    def gather_episodes(state_bounds, n_actions, model, n_episodes, env_name):
        """Interact with environment in batched fashion."""

        trajectories = [defaultdict(list) for _ in range(n_episodes)]
        envs = [gym.make(env_name) for _ in range(n_episodes)]

        done = [False for _ in range(n_episodes)]
        states = [e.reset() for e in envs]

        while not all(done):
            unfinished_ids = [i for i in range(n_episodes) if not done[i]]
            normalized_states = [s/state_bounds for i, s in enumerate(states) if not done[i]]

            for i, state in zip(unfinished_ids, normalized_states):
                trajectories[i]['states'].append(state)

            # Compute policy for all unfinished envs in parallel
            states = tf.convert_to_tensor(normalized_states)
            action_probs = model([states])

            # Store action and transition all environments to the next state
            states = [None for i in range(n_episodes)]
            for i, policy in zip(unfinished_ids, action_probs.numpy()):
                action = np.random.choice(n_actions, p=policy)
                states[i], reward, done[i], _ = envs[i].step(action)
                trajectories[i]['actions'].append(action)
                trajectories[i]['rewards'].append(reward)

        return trajectories

    def compute_returns(rewards_history, gamma):
        """Compute discounted returns with discount factor `gamma`."""
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize them for faster and more stable learning
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        returns = returns.tolist()

        return returns

    @tf.function
    def reinforce_update(states, actions, returns, model, batch_size):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns)

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            logits = model(states)
            p_actions = tf.gather_nd(logits, actions)
            log_probs = tf.math.log(p_actions)
            loss = tf.math.reduce_sum(-log_probs * returns) / batch_size
        grads = tape.gradient(loss, model.trainable_variables)
        for optimizer, w in zip([REINFORCE.optimizer_in, REINFORCE.optimizer_var, REINFORCE.optimizer_out], [REINFORCE.w_in, REINFORCE.w_var, REINFORCE.w_out]):
            optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])
