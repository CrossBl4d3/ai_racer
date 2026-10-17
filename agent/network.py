"""The same Q-network in PyTorch (PyTorch does the backpropagation for us)."""
import torch
import torch.nn as nn
import config


class QNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = config.HIDDEN_SIZES

        layers = []
        size = n_inputs
        for hidden in hidden_sizes:
            layers.append(nn.Linear(size, hidden))
            layers.append(nn.ReLU())
            size = hidden
        layers.append(nn.Linear(size, n_actions))      # no ReLU: Q-values can be negative
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def act(self, observation):
        with torch.no_grad():
            x = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            return int(self.forward(x).argmax(dim=1).item())

    def copy_from(self, other):
        self.load_state_dict(other.state_dict())

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))

    def linear_layers(self):
        return [module for module in self.layers if isinstance(module, nn.Linear)]

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())