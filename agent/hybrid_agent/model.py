"""
DQNModel — kiến trúc 2 nhánh (Conv2D cho map + MLP cho aux).

Copy từ agent/dqn_agent/agent.py, KHÔNG import baseline (bản submit chỉ mang
agent.py + model.py + model.pth ở gốc zip). Kiến trúc giữ nguyên — chỉ đổi
shape mặc định theo plan.md: 13 channel + 6 aux (thay vì 9 + 3 của baseline).
"""

import torch
import torch.nn as nn

# Shape theo plan.md (Bước 5). model.pth của baseline (9ch/3aux) KHÔNG nạp được vào đây.
MAP_SHAPE = (13, 13, 13)   # (channels, H, W)
AUX_DIM = 6
OUTPUT_DIM = 6             # 6 action


class DQNModel(nn.Module):
    """
    Two-branch DQN:
      - Conv2D branch for spatial map/object channels
      - MLP branch for auxiliary scalar features
    Hoàn toàn parametrized theo map_shape/aux_dim — không hardcode số channel.
    """
    def __init__(self, map_shape=MAP_SHAPE, aux_dim=AUX_DIM, output_dim=OUTPUT_DIM):
        super().__init__()
        c, h, w = map_shape
        self.map_encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_dim = self.map_encoder(dummy).reshape(1, -1).size(1)

        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(conv_out_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, map_x, aux_x):
        map_feat = self.map_encoder(map_x).reshape(map_x.size(0), -1)
        aux_feat = self.aux_encoder(aux_x)
        feat = torch.cat([map_feat, aux_feat], dim=1)
        return self.head(feat)
