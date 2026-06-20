import torch
import torch.nn as nn

class PhaseUtilityNet(nn.Module):
    def __init__(self, state_dim=13, action_dim=33, hidden_dims=[128, 64]):
        super(PhaseUtilityNet, self).__init__()
        input_dim = state_dim + action_dim
        
        # Actor network
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.Mish())  # Mish activation
            layers.append(nn.Dropout(0.1))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        
        # Critic network (predicts state value V(s) from state features only)
        critic_layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, h_dim))
            critic_layers.append(nn.LayerNorm(h_dim))
            critic_layers.append(nn.Mish())
            prev_dim = h_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
    def forward(self, state, action):
        # state: [batch_size, state_dim]
        # action: [batch_size, 6, action_dim]
        
        if len(state.shape) == 2:
            num_actions = action.shape[1]
            state = state.unsqueeze(1).repeat(1, num_actions, 1)  # [batch_size, 6, state_dim]
            
        # Concatenate along feature dimension
        x = torch.cat([state, action], dim=-1)  # [batch_size, 6, state_dim + action_dim]
        
        batch_size, num_actions, feat_dim = x.shape
        x_flat = x.view(-1, feat_dim)
        out_flat = self.net(x_flat)  # [batch_size * 6, 1]
        
        return out_flat.view(batch_size, num_actions)  # [batch_size, 6]

    def get_value(self, state):
        # state: [batch_size, state_dim]
        return self.critic(state).squeeze(-1)  # [batch_size]


