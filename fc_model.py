from torch import nn

class SimpleNet(nn.Module):
    def __init__(self):
        """Initialize layers."""
        super().__init__()

        self.dense_0 = nn.Linear(784, 256)
        self.dense_1 = nn.Linear(256, 128)
        self.dense_2 = nn.Linear(128, 64)
        self.dense_3 = nn.Linear(64, 10)
        self.relu    = nn.ReLU()
        
    def forward(self, x):
        # Flatten imput (from [batch_size, 1, 28, 28] to [batch_size, 1, 784])
        x = x.flatten(start_dim = 2)
        # Flatten imput (from [batch_size, 1, 28, 28] to [batch_size, 1, 784])
        x = x.squeeze(dim = 1)

        """Forward pass of network."""
        x = self.dense_0(x)
        x = self.relu(x)
        x = self.dense_1(x)
        x = self.relu(x)
        x = self.dense_2(x)
        x = self.relu(x)
        x = self.dense_3(x)

        return x
