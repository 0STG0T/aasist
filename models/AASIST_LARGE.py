import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AASIST import CONV, Model as AASIST

class AASIST_LARGE(AASIST):
    def __init__(self, num_heads=8, sinc_filters=256, res2net_filters=512):
        super().__init__(d_args={"filts": [[sinc_filters], [1, res2net_filters], [res2net_filters, res2net_filters],
                                          [res2net_filters, res2net_filters], [res2net_filters, res2net_filters]],
                                "gat_dims": [res2net_filters, res2net_filters],
                                "pool_ratios": [0.5, 0.5, 0.5],
                                "temperatures": [0.7, 0.7, 0.7],
                                "first_conv": 1024})

        # Override dropout rates for larger model
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.3, inplace=True)

        # Add additional attention heads
        self.additional_attention = nn.ModuleList([
            nn.MultiheadAttention(res2net_filters, num_heads)
            for _ in range(2)
        ])

        # Override the number of attention heads
        self.GAT_layer_S.n_heads = num_heads
        self.GAT_layer_T.n_heads = num_heads
        self.HtrgGAT_layer_ST11.n_heads = num_heads
        self.HtrgGAT_layer_ST12.n_heads = num_heads
        self.HtrgGAT_layer_ST21.n_heads = num_heads
        self.HtrgGAT_layer_ST22.n_heads = num_heads

    def forward(self, x, Freq_aug=False):
        # Get base model features
        hidden, output = super().forward(x, Freq_aug)

        # Apply additional attention layers
        for attn in self.additional_attention:
            # Reshape for attention
            h = hidden.unsqueeze(0)  # Add sequence dimension
            h, _ = attn(h, h, h)
            h = h.squeeze(0)  # Remove sequence dimension
            # Residual connection
            hidden = hidden + h

        # Use the enhanced features for final prediction
        hidden = self.drop(hidden)
        output = self.out_layer(hidden)

        return hidden, output
