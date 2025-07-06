import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_z = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)
        self.conv_r = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)
        self.conv_h = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_reset))
        h_next = (1 - z) * h_prev + z * h_tilde
        return h_next
