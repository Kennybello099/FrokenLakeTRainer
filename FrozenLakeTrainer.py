import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

class FrozenLake:
    def __init__(self, map_name="8x8", is_slippery=False):
        """Initialize the FrozenLake environment and Q-learning parameters."""
        self.env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery)
        self.q = None  # Q-table will be initialized in run method
        self.learning_rate_a = 0.9  # alpha or learning rate
        self.discount_factor_g = 0.9  # gamma or discount factor
        self.epsilon = 1  # 1 = 100% random actions initially
        self.epsilon_decay_rate = 0.0001  # 1/0.0001 = 10,000 episodes to decay fully
        self.rng = np.random.default_rng()  # random number generator

    def run(self, episodes, isTraining=True, render=False):
        """Run the Q-learning algorithm for the specified number of episodes."""
        # Update render mode based on render parameter
        self.env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, 
                            render_mode="human" if render else None)

        # Initialize or load Q-table
        if isTraining:
            self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))  # 64 x 4 array
        else:
            try:
                with open('frozen_lake8x8.pkl', 'rb') as f:
                    self.q = pickle.load(f)
            except FileNotFoundError:
                print("Error: 'frozen_lake8x8.pkl' not found. Please train the model first with isTraining=True.")
                return

        rewards_per_episode = np.zeros(episodes)

        for i in range(episodes):
            state = self.env.reset()[0]  # 0 - 63, 0 = top left, 63 = bottom right
            terminated = False  # True when hole or goal reached
            truncated = False  # True when actions > 200

            while not terminated and not truncated:
                if isTraining and self.rng.random() < self.epsilon:
                    action = self.env.action_space.sample()  # 0=left, 1=down, 2=right, 3=up
                else:
                    action = np.argmax(self.q[state, :])  # Select action from Q-table

                new_state, reward, terminated, truncated, _ = self.env.step(action)

                if isTraining:
                    self.q[state, action] = self.q[state, action] + self.learning_rate_a * (
                        reward + self.discount_factor_g * np.max(self.q[new_state, :]) - self.q[state, action]
                    )

                state = new_state

            if isTraining:
                self.epsilon = max(self.epsilon - self.epsilon_decay_rate, 0)
                if self.epsilon == 0:
                    self.learning_rate_a = 0.0001  # Adjust learning rate after exploration ends

            if reward == 1:
                rewards_per_episode[i] = 1

        self.env.close()

        # Plot cumulative rewards over a 100-episode window
        sum_rewards = np.zeros(episodes)
        for y in range(episodes):
            sum_rewards[y] = np.sum(rewards_per_episode[max(0, y-100):(y+1)])
        plt.plot(sum_rewards)
        plt.savefig('frozen_lake8x8.png')

        # Save Q-table only if training
        if isTraining:
            with open("frozen_lake8x8.pkl", "wb") as f:
                pickle.dump(self.q, f)

if __name__ == '__main__':
    # Create an instance of FrozenLake and run it
    lake = FrozenLake(map_name="8x8", is_slippery=True)
    #lake.run(1500)  # Uncomment to run with default isTraining=True, render=False
    lake.run(1000, isTraining=False, render=True)