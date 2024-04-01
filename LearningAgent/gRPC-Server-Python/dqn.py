import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
from gymnasium import Env
class DQN(nn.Module):
    def __init__(self, env: Env, hidden_size=64, lr=1e-3, gamma=0.99, epsilon=0.1, buffer_length=10000, update_freq=100, target_update_freq=500, batch_size=128):
        super(DQN, self).__init__()
        self.env : Env = env 
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer_length = buffer_length
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size

        # Neural network for Q-learning
        self.network = nn.Sequential(
            nn.Linear(self.state_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_space)
        )

        # Target network
        self.target_network = nn.Sequential(
            nn.Linear(self.state_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_space)
        )
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.replay_buffer = deque(maxlen=self.buffer_length)

        # TensorBoard
        self.writer = SummaryWriter(log_dir="logs/dqn_cartpole")

    def predict(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.network(state_tensor)
                action = torch.argmax(q_values).item()
        return action

    def learn(self, steps):
        state, _ = self.env.reset()
        total_rewards = 0
        episode_rewards = []
        mean_reward = np.NaN
        for step in range(steps):
            action = self.predict(state, self.epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = 1 if terminated or truncated else 0
            self.replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_rewards += reward

            if done:
                episode_rewards.append(total_rewards)
                state,_ = self.env.reset()
                total_rewards = 0

            if len(self.replay_buffer) >= self.batch_size and step % self.update_freq == 0:
                self.optimize_model()

            if step % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

            if (step + 1) % 100 == 0 and len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards)
                self.writer.add_scalar('Mean Episodic Reward', mean_reward, step)
                episode_rewards = []
            self.print_progress_bar(step,steps,f"{step}/{steps} steps done.")

        self.writer.flush()

    def optimize_model(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_network(next_states).max(1)[0]
        expected_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('Loss', loss.item(), self.optimizer.state_dict()['state'][list(self.optimizer.state_dict()['state'].keys())[0]]['step'])

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        self.network.load_state_dict(torch.load(filename))
    
    def print_progress_bar(self, index, total, label):
        n_bar = 50  # Progress bar width
        progress = index / total
        sys.stdout.write('\r')
        sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}\n")

# Example of using the DQN class with CartPole
if __name__ == "__main__":
    import gymnasium as gym
    env = gym.make("CartPole-v1",render_mode="rgb_array")
    agent = DQN(env)

    agent.learn(4_000_00)
    agent.save('dqn_cartpole.pth')
    # predict
    print("Learning done!")
    env = gym.make("CartPole-v1",render_mode="human")
    state,_ = env.reset()
    while True:
        action = agent.predict(state, epsilon=0.0)
        state, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            state,_ = env.reset()

