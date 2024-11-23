import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AASIST import SincConv, Res2NetBlock, AASIST

class AASIST_LARGE(AASIST):
    def __init__(self, num_heads=8, sinc_filters=256, res2net_filters=512):
        super(AASIST, self).__init__()
        self.conv_time = SincConv(sinc_filters, 1024, 16000)
        self.res2net_blocks = nn.ModuleList([
            Res2NetBlock(sinc_filters, res2net_filters, scale=8, padding=4)
            for _ in range(4)
        ])
        self.mha = nn.MultiheadAttention(res2net_filters, num_heads)
        self.fc = nn.Sequential(
            nn.Linear(res2net_filters, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
