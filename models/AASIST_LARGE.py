"""
Enhanced AASIST model with increased capacity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1)

    def forward(self, x):
        h = self.fc(x)
        N = h.size(0)
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
        e = F.leaky_relu(self.attn_fc(a_input).squeeze(1))
        attention = F.softmax(e.view(N, N), dim=1)
        return torch.mm(attention, h)

class AASIST_LARGE(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Enhanced first conv layer
        self.conv_time = nn.Conv1d(1, 128, 3, padding=1)
        self.bn_time = nn.BatchNorm1d(128)

        # Larger conv blocks
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(128, 256, 3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.5)
            ),
            nn.Sequential(
                nn.Conv1d(256, 512, 3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.5)
            )
        ])

        # Enhanced GAT layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(512, 512),
            GraphAttentionLayer(512, 256)
        ])

        # Final layers
        self.fc = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Initial temporal convolution
        x = self.conv_time(x)
        x = self.bn_time(x)
        x = F.relu(x)

        # Convolutional blocks
        for block in self.conv_blocks:
            x = block(x)

        # Prepare for GAT
        x = x.transpose(1, 2)
        batch_size, seq_len, channels = x.size()

        # Apply GAT layers
        for gat in self.gat_layers:
            x = gat(x.reshape(-1, channels)).view(batch_size, seq_len, -1)

        # Global pooling and classification
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
