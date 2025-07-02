import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Hyperparameters
EPISODES = 500
GAMMA = 0.99
LR = 1e-3
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 10000

env = gym.make("CartPole-v1")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# Neural network for DQN
class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_states, 24),
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.ReLU(),
            nn.Linear(48, n_actions)
        )

    def forward(self, x):
        return self.layers(x)

# Agent class
class Agent:
    def __init__(self):
        self.model = DQN(n_states, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0

    def act(self, state):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        state = torch.tensor([state], dtype=torch.float32)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        s, a, r, s_, d = zip(*batch)
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float32)
        s_ = torch.tensor(s_, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.bool)

        q_values = self.model(s)
        next_q_values = self.model(s_).detach()

        target_q = q_values.clone()
        for i in range(BATCH_SIZE):
            target_q[i][a[i]] = r[i] + GAMMA * torch.max(next_q_values[i]) * (not d[i])

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

agent = Agent()
rewards = []

# Training loop
for ep in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward

    agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
    rewards.append(total_reward)
    print(f"Episode {ep+1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

# Plot rewards
plt.plot(rewards)
plt.title("DQN Training Rewards on CartPole")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.savefig("rewards.png")
env.close()
