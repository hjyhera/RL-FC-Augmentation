import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class LightActor(nn.Module):
    """
    경량화된 Actor 모델
    - d_model: 256 → 128 (50% 감소)
    - num_layers: 4 → 2 (50% 감소)  
    - dim_feedforward: 512 → 256 (50% 감소)
    - nhead: 8 → 4 (50% 감소)
    """
    def __init__(
        self,
        input_dim: int = 6216,
        d_model: int = 256,        # 256 → 128
        nhead: int = 8,            # 8
        num_layers: int = 4,       # 4
        dim_feedforward: int = 512, # 512
        dropout: float = 0.25,
    ):
        super().__init__()
        self.d_model = d_model

        # 입력 projection을 더 효율적으로
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),  # 중간 차원 추가
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,  # Pre-norm for better stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)

        # Weight initialization
        self._init_weights()
        self.h_last: Optional[torch.Tensor] = None

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)                          # [B,T,d_model]
        h = h * math.sqrt(self.d_model)                 # 안정적 스케일
        h_enc = self.encoder(h)                         # [B,T,d_model]

        logits = self.head(h_enc)               # [B,T,2]
        self.h_last = h_enc.mean(dim=1)         # [B,d_model]
        return logits

    def forward_with_hidden(self, x: torch.Tensor):
        logits = self.forward(x)
        return logits, self.h_last


class LightCritic(nn.Module):
    """
    경량화된 Critic 모델
    - hidden: 128 → 64 (50% 감소)
    - 레이어 수 감소: 3 → 2
    """
    def __init__(self, h_dim: int, hidden: int = 64):  # 128 → 64
        super().__init__()
        self.in_norm = nn.LayerNorm(h_dim)      
        self.net = nn.Sequential(
            nn.Linear(h_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),  # 정규화 추가
            nn.Linear(hidden, 1)  # 중간 레이어 제거
        )
        
        # Weight initialization
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        h = self.in_norm(h)                     
        return self.net(h).squeeze(-1)


# 기존 Actor/Critic과 호환성을 위한 별칭
Actor = LightActor
Critic = LightCritic
