import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Actor(nn.Module):
    def __init__(
        self,
        input_dim: int = 6216,
        d_model: int = 256,        
        nhead: int = 8,           
        num_layers: int = 4,      
        dim_feedforward: int = 512, 
        dropout: float = 0.25,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model * 2), 
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
            norm_first=True, 
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
        h = h * math.sqrt(self.d_model)          
        h_enc = self.encoder(h)                         # [B,T,d_model]

        logits = self.head(h_enc)               # [B,T,2]
        self.h_last = h_enc.mean(dim=1)         # [B,d_model]
        return logits

    def forward_with_hidden(self, x: torch.Tensor):
        logits = self.forward(x)
        return logits, self.h_last


class Critic(nn.Module):
    def __init__(self, h_dim: int, hidden: int = 64):  
        super().__init__()
        self.in_norm = nn.LayerNorm(h_dim)      
        self.net = nn.Sequential(
            nn.Linear(h_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(hidden, 1)  
        )
        
        # Weight initialization
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        h = self.in_norm(h)                     
        return self.net(h).squeeze(-1)

