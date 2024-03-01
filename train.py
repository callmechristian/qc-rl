from REINFORCE import *
from DeepQLearning import *
from PIL import Image
import os
import Environments
from enum import Enum
from scipy.special import softmax
import operator

class TrainMethod(Enum):
    REINFORCE = 0
    DeepQLearning = 1

episode_reward_history = []

def train(reward_target=500.0, realtime_render=False, batch_size=10, env_type=Environments.CartPole, method=TrainMethod.REINFORCE, n_episodes=1000):        

    if method==TrainMethod.REINFORCE:
        if env_type == Environments.AtariBreakout:
            raise NotImplementedError("Policy Gradient not implemented for atari.")
            return train_policy_gradient_atari(reward_target, realtime_render, batch_size, env_type, n_episodes=n_episodes)
        else:
            return train_policy_gradient(reward_target, realtime_render, batch_size, env_type, n_episodes=n_episodes)
    elif method == TrainMethod.DeepQLearning:
        if env_type == Environments.AtariBreakout:
            raise NotImplementedError("Deep Q-Learning not implemented for atari.")
            return train_deepq_atari(reward_target, env_type, batch_size=batch_size, n_episodes=n_episodes)
        else:
            return train_deepq(reward_target, env_type, batch_size=batch_size, n_episodes=n_episodes)
    else:
        raise ValueError("Unrecognized training method! Check the TrainMethod enum for valid methods.")

def train_policy_gradient(reward_target: float, realtime_render: bool, batch_size: int, env_type: Environments.Environment, n_episodes=1000):
    qubits = cirq.GridQubit.rect(1, env_type.n_qubits)

    ops = [cirq.Z(q) for q in qubits]
    observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3

    model = REINFORCE.generate_model_policy(qubits, env_type.n_layers, env_type.n_actions, 0.9, observables)
    
    env = None

    episode_reward_history = []
    # Start training the agent
    for batch in range(n_episodes // batch_size):
        # Gather episodes
        episodes = REINFORCE.gather_episodes(env_type.state_bounds, env_type.n_actions, model, batch_size, env_type.env_name)

        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]
        returns = np.concatenate([REINFORCE.compute_returns(ep_rwds, REINFORCE.gamma) for ep_rwds in rewards])
        returns = np.array(returns, dtype=np.float32)

        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

        # Update model parameters.
        REINFORCE.reinforce_update(states, id_action_pairs, returns, model, batch_size)

        # Store collected rewards
        for ep_rwds in rewards:
            episode_reward_history.append(np.sum(ep_rwds))

        avg_rewards = np.mean(episode_reward_history[-batch_size:])

        print('Finished episode', (batch + 1) * batch_size,
            'Average rewards: ', avg_rewards)
        
        if realtime_render:
            env = gym.make(env_type.env_name)
            state = env.reset()
            for t in range(500):
                env.render()
                policy = model([tf.convert_to_tensor([state/env_type.state_bounds])])
                action = np.random.choice(env_type.n_actions, p=policy.numpy()[0])
                state, _, done, _ = env.step(action)
                if done:
                    break
            env.close()

        if avg_rewards >= reward_target:
            break

    return episode_reward_history, model, env

def train_deepq(reward_target: float, env_type: Environments.Environment, batch_size=16, n_episodes=1000):
    qubits = cirq.GridQubit.rect(1, env_type.n_qubits)

    ops = [cirq.Z(q) for q in qubits]
    # observables = [ops[0]*ops[1], ops[2]*ops[3]] # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1
    observables = env_type.observables_func(ops)


    model = DeepQLearning.generate_model_Qlearning(qubits, env_type.n_layers, env_type.n_actions, observables, False)
    model_target = DeepQLearning.generate_model_Qlearning(qubits, env_type.n_layers, env_type.n_actions, observables, True)

    model_target.set_weights(model.get_weights())

    episode_reward_history = []
    step_count = 0
    env = gym.make(env_type.env_name)

    best_model = None

    for episode in range(n_episodes):
        episode_reward = 0
        state = env.reset()

        while True:
            # Interact with env
            interaction = DeepQLearning.interact_env(state, model, DeepQLearning.epsilon, env_type.n_actions, env)

            # Store interaction in the replay memory
            DeepQLearning.replay_memory.append(interaction)

            state = interaction['next_state']
            episode_reward += interaction['reward']
            step_count += 1

            # Update model
            if step_count % DeepQLearning.steps_per_update == 0:
                # Sample a batch of interactions and update Q_function
                training_batch = np.random.choice(DeepQLearning.replay_memory, size=batch_size)
                DeepQLearning.Q_learning_update(np.asarray([x['state'] for x in training_batch]),
                                np.asarray([x['action'] for x in training_batch]),
                                np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                np.asarray([x['next_state'] for x in training_batch]),
                                np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                                model, DeepQLearning.gamma, env_type.n_actions, model_target)

            # Update target model
            if step_count % DeepQLearning.steps_per_target_update == 0:
                model_target.set_weights(model.get_weights())

            # Check if the episode is finished
            if interaction['done']:
                break

        # Decay epsilon
        DeepQLearning.epsilon = max(DeepQLearning.epsilon * DeepQLearning.decay_epsilon, DeepQLearning.epsilon_min)
        
        
        # SAVE BEST MODEL -- if target reward is not reached
        if all(episode_reward > er for er in episode_reward_history):
            best_model = model
            print("ADDED MODEL")
            
        # ADD NEW EPISODE REWARD
        episode_reward_history.append(episode_reward)
        
        if (episode+1)%batch_size == 0:
            avg_rewards = np.mean(episode_reward_history[-batch_size:])
            print("Episode {}/{}, average last {} rewards {}".format(
                episode+1, n_episodes, batch_size, avg_rewards))
            if avg_rewards >= reward_target:
                break
    return episode_reward_history, model, env, best_model

def export(history: list, env_type, model, train_method: TrainMethod, dir="./images", episodes=0, note=""):
    if len(history) == 0:
        raise IndexError("Train a model first!")

    nr = len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])

    env = gym.make(env_type.env_name)
    state = env.reset()
    frames = []

    if train_method == TrainMethod.REINFORCE:
        for t in range(500):
            im = Image.fromarray(env.render(mode='rgb_array'))
            frames.append(im)
            policy = model([tf.convert_to_tensor([state/env_type.state_bounds])])
            action = np.random.choice(env_type.n_actions, p=policy.numpy()[0]) # only for two actions??
            state, _, done, _ = env.step(action)
            if done:
                break
    elif train_method == TrainMethod.DeepQLearning:
        for t in range(500):
            im = Image.fromarray(env.render(mode='rgb_array'))
            frames.append(im)
            # Use the model to predict the action probabilities
            action_probs = model([tf.convert_to_tensor([state/env_type.state_bounds])])
            action = int(tf.argmax(action_probs[0]).numpy())
            state, _, done, _ = env.step(action)
            if done:
                break
    env.close()
    if train_method == TrainMethod.REINFORCE:
        frames[0].save(f"{dir}/{nr}_gym_{env_type.env_name}_REINFORCE_batchSize=?_gamma={REINFORCE.gamma}_episodes={episodes}_{note}.gif",
                save_all=True, append_images=frames[1:], optimize=False, duration=40, loop=0)
    elif train_method == TrainMethod.DeepQLearning:
        frames[1].save(f"{dir}/{nr}_gym_{env_type.env_name}_DeepQLearning_batchSize={DeepQLearning.batch_size}_gamma={DeepQLearning.gamma}_episodes={episodes}_learningrate_{[DeepQLearning.learning_rate_in, DeepQLearning.learning_rate_var, DeepQLearning.learning_rate_out]}_{note}.gif",
                    save_all=True, append_images=frames[2:], optimize=False, duration=40, loop=0)