# Quantum Q-Policy Actor-Critic Reinforcement Learning #

from PolicyGradientRL import *
from DeepQRL import *
from Environments import Environment

class ActorCriticAgent:
    def __init__(self, env_type: Environment):
        self.env_type = env_type
        self.actor_model = self._initialize_actor_model()
        self.critic_model = self._initialize_critic_model()

    def _initialize_actor_model(self):
        qubits = cirq.GridQubit.rect(1, self.env_type.n_qubits)
        ops = [cirq.Z(q) for q in qubits]
        observables = [reduce((lambda x, y: x * y), ops)]  # Policy Gradient observables
        return generate_model_policy(qubits, self.env_type.n_layers, self.env_type.n_actions, 1.0, observables)

    def _initialize_critic_model(self):
        qubits = cirq.GridQubit.rect(1, self.env_type.n_qubits)
        ops = [cirq.Z(q) for q in qubits]
        observables = [ops[0] * ops[1], ops[2] * ops[3]]  # Q-Learning observables
        return DeepQRL.generate_model_Qlearning(qubits, self.env_type.n_layers, self.env_type.n_actions, observables, False)

    def train_actor_critic(self, n_episodes, batch_size, reward_target=500.0):
        episode_reward_history = []
        for episode in range(n_episodes):
            # Implement training steps for both Actor (Policy Gradient) and Critic (Q-Learning)
            # Training the Actor (Policy Gradient)
            actor_rewards = self._train_actor(batch_size, reward_target)

            # Training the Critic (Q-Learning)
            critic_rewards = self._train_critic()

            # Update both models based on their respective training methods
            if episode % batch_size == 0:
                # Update the Critic using the Actor's policy (Q-Learning)
                self._update_critic_with_actor()

                # Update the Actor using the Critic's feedback (Advantage, TD-error, etc.) (Policy Gradient)
                self._update_actor_with_critic_feedback()

            episode_reward_history.append(episode_reward)  # Append the episode reward

            # Check for target reward achieved
            if np.mean(episode_reward_history[-batch_size:]) >= reward_target:
                break

        return episode_reward_history, self.actor_model, self.critic_model

    def _train_actor(self):
        episode_rewards = []
        batch_size = 1 # NotImplemented
        # Gather episodes
        episodes = gather_episodes(self.env_type.state_bounds, self.env_type.n_actions, self.actor_model, batch_size, self.env_type.env_name)

        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]
        returns = np.concatenate([compute_returns(ep_rwds, gamma) for ep_rwds in rewards])
        returns = np.array(returns, dtype=np.float32)

        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

        # Update model parameters.
        reinforce_update(states, id_action_pairs, returns, self.actor_model, batch_size)
        # Update Q-model
        # self.critic_model.set_weights(self.actor_model.get_weights())

        # Store collected rewards
        for ep_rwds in rewards:
            episode_rewards.append(np.sum(ep_rwds))

        avg_rewards = np.mean(episode_rewards[-batch_size:])

        return avg_rewards

    def _train_critic(self, model_target):
        batch_size = 1 #NotImplemented

        step_count = 0
        env = gym.make(self.env_type.env_name)

        episode_reward = 0
        state = env.reset()

        while True:
            # Interact with env
            interaction = DeepQRL.interact_env(state, self.actor_model, DeepQRL.epsilon, self.env_type.n_actions, env)

            # Store interaction in the replay memory
            DeepQRL.replay_memory.append(interaction)

            state = interaction['next_state']
            episode_reward += interaction['reward']
            step_count += 1

            # Update model
            if step_count % DeepQRL.steps_per_update == 0:
                # Sample a batch of interactions and update Q_function
                training_batch = np.random.choice(DeepQRL.replay_memory, size=batch_size)
                DeepQRL.Q_learning_update(np.asarray([x['state'] for x in training_batch]),
                                np.asarray([x['action'] for x in training_batch]),
                                np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                np.asarray([x['next_state'] for x in training_batch]),
                                np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                                self.model, DeepQRL.gamma, self.env_type.n_actions, model_target)
                # Update Policy


            # Check if the episode is finished
            if interaction['done']:
                break

        # Decay epsilon
        DeepQRL.epsilon = max(DeepQRL.epsilon * DeepQRL.decay_epsilon, DeepQRL.epsilon_min)

        return episode_reward

    def _update_critic_with_actor(self):
        # Update the Critic using the Actor's policy (Q-Learning update based on Actor's policy)
        # Example:
        self.critic_model.update_with_actor(self.actor_model)  # Update the critic based on the actor's policy

    def _update_actor_with_critic_feedback(self):
        # Update the Actor using the Critic's feedback (Policy Gradient update based on Critic's feedback)
        # Example:
        self.actor_model.update_with_critic(self.critic_model)  # Update the actor based on the critic's feedback


# # Example of using the Actor-Critic Agent
# env = Environments.CartPole  # Define the environment type
# agent = ActorCriticAgent(env)
# episodes = 1000  # Number of training episodes
# batch_size = 10  # Batch size for actor-critic updates
# rewards, actor_model, critic_model = agent.train_actor_critic(episodes, batch_size)

