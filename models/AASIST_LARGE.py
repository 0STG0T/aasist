"""
AASIST_LARGE: A larger version of the AASIST model with increased capacity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AASIST import Model as AASIST

class AASIST_LARGE(AASIST):
    def __init__(self, d_args):
        # Modify the architecture parameters for increased capacity
        d_args["filts"] = [70, [1, 64], [64, 64], [64, 128], [128, 128]]  # Doubled filter sizes
        d_args["gat_dims"] = [128, 64]  # Doubled GAT dimensions
        d_args["first_conv"] = 256  # Doubled first conv kernel size

        # Initialize the base model with modified parameters
        super().__init__(d_args)

        # Additional dropout for regularization
        self.drop = nn.Dropout(0.6, inplace=True)
        self.drop_way = nn.Dropout(0.3, inplace=True)

        # Larger output layer to match increased dimensions
        self.out_layer = nn.Linear(5 * d_args["gat_dims"][1], 2)
