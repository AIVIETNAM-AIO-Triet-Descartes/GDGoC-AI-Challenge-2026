"""
Training utilities — ReplayBuffer + TrainingAgent.

Copy từ agent/dqn_agent/agent.py. CHỈ dùng khi train (Kaggle/Colab), KHÔNG cần
khi submit/inference. Shape mặc định đổi theo plan.md: (13,13,13) + 6 aux.

Logic train giữ nguyên baseline (DQN + target net). encode_obs_enriched (Bước 5)
sẽ được dùng bên ngoài để tạo state trước khi push vào buffer / gọi act.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQNModel, MAP_SHAPE, AUX_DIM, OUTPUT_DIM


class ReplayBuffer:
    """Pre-allocated numpy circular buffer — sample() là pure array indexing, no Python objects."""
    def __init__(self, capacity: int, map_shape=MAP_SHAPE, aux_dim: int = AUX_DIM):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.map_shape = tuple(map_shape)
        self.aux_dim = int(aux_dim)
        self.map_states      = np.zeros((capacity, *self.map_shape), dtype=np.float32)
        self.aux_states      = np.zeros((capacity, self.aux_dim), dtype=np.float32)
        self.next_map_states = np.zeros((capacity, *self.map_shape), dtype=np.float32)
        self.next_aux_states = np.zeros((capacity, self.aux_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones   = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return self.size

    def push(self, map_state, aux_state, action, reward, next_map_state, next_aux_state, done):
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.map_states[self.pos]      = map_state
        self.aux_states[self.pos]      = aux_state
        self.next_map_states[self.pos] = next_map_state
        self.next_aux_states[self.pos] = next_aux_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos]   = done

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.map_states[idx],
            self.aux_states[idx],
            self.next_map_states[idx],
            self.next_aux_states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
        )


class TrainingAgent:
    """DQN trainer — Q-net + target-net, epsilon-greedy act, train_step (1 batch)."""
    team_id = "HybridAgent"

    def __init__(self, agent_id: int, input_spec=(MAP_SHAPE, AUX_DIM), num_actions: int = OUTPUT_DIM,
                 lr: float = 1e-3, device: str = "cpu", pretrained_model=None):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.device = device
        self.gamma = 0.99
        self.lr = lr
        self.global_step = 0
        self.epsilon = 1.0

        if pretrained_model:
            self.load_agent(pretrained_model)
        else:
            self.map_shape = tuple(input_spec[0])
            self.aux_dim = int(input_spec[1])
            self.q_net = DQNModel(self.map_shape, self.aux_dim, num_actions).to(device)
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr, eps=1e-08, weight_decay=1e-5)

        self.target_net = DQNModel(self.map_shape, self.aux_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())  # sync ban đầu

        self.loss_fn = nn.MSELoss()

    def act(self, map_state, aux_state, epsilon=0.0):
        """Epsilon-greedy. NOTE: chưa mask safe_actions — sẽ ghép ở Bước 8 khi train."""
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        map_tensor = torch.from_numpy(map_state).unsqueeze(0).to(self.device)
        aux_tensor = torch.from_numpy(aux_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.q_net(map_tensor, aux_tensor).argmax().item()
        return action

    def train_step(self, map_state, aux_state, next_map_state, next_aux_state, action, reward, done):
        """1 bước cập nhật DQN trên 1 batch (đã sample từ ReplayBuffer)."""
        map_state_t      = torch.from_numpy(map_state)
        aux_state_t      = torch.from_numpy(aux_state)
        next_map_state_t = torch.from_numpy(next_map_state)
        next_aux_state_t = torch.from_numpy(next_aux_state)
        action_t = torch.from_numpy(action).unsqueeze(1)
        reward_t = torch.from_numpy(reward).unsqueeze(1)
        done_t   = torch.from_numpy(done).unsqueeze(1)
        if self.device != "cpu":
            map_state_t      = map_state_t.to(self.device)
            aux_state_t      = aux_state_t.to(self.device)
            next_map_state_t = next_map_state_t.to(self.device)
            next_aux_state_t = next_aux_state_t.to(self.device)
            action_t = action_t.to(self.device)
            reward_t = reward_t.to(self.device)
            done_t   = done_t.to(self.device)

        q_values = self.q_net(map_state_t, aux_state_t).gather(1, action_t)

        with torch.no_grad():
            max_next_q = self.target_net(next_map_state_t, next_aux_state_t).max(1)[0].unsqueeze(1)
            target_q = reward_t + self.gamma * max_next_q * (1 - done_t)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return loss.item()

    def update_target_network(self):
        """Copy weights Q-net → target-net."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_agent(self, path):
        """Lưu checkpoint đầy đủ. Để submit: dùng model.pth chỉ state_dict (xem ghi chú)."""
        torch.save({
            "input_spec": (self.map_shape, self.aux_dim),
            "num_actions": self.num_actions,
            "model_state_dict": self.q_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr": self.lr,
            "global_step": self.global_step,
            "epsilon": self.epsilon,
        }, path)

    def load_agent(self, pretrained_model):
        checkpoint = torch.load(pretrained_model, map_location=self.device)
        input_spec = checkpoint.get("input_spec", checkpoint.get("input_shape", checkpoint.get("input_dim")))
        self.map_shape = tuple(input_spec[0])
        self.aux_dim = int(input_spec[1])
        self.num_actions = checkpoint["num_actions"]
        self.q_net = DQNModel(self.map_shape, self.aux_dim, self.num_actions).to(self.device)
        self.q_net.load_state_dict(checkpoint["model_state_dict"])
        self.lr = checkpoint["lr"]
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr, eps=1e-08, weight_decay=1e-5)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epsilon = checkpoint["epsilon"]
