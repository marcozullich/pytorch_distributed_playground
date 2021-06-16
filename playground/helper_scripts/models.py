import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, dim_input=784, num_classes=10, latent_dim=64):
        super().__init__()
        self.chunk1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim_input, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.chunk2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_classes)
        )
    
    def forward(self, data, print_shape=False):
        if print_shape:
            print(f"\t\tData shape in forward: {data.shape}")
        out = self.chunk1(data)
        out = self.chunk2(out)
        return out