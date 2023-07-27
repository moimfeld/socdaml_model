from torch import nn

class ConvNet(nn.Module):
    def __init__(self):
        """Initialize layers."""
        super().__init__()

        self.conv_0  = nn.Conv2d(1, 1, 3)
        self.conv_1  = nn.Conv2d(1, 1, 3)
        self.conv_2  = nn.Conv2d(1, 1, 3)
        self.dense_0 = nn.Linear(22*22, 10)
        self.relu    = nn.ReLU()

    def forward(self, x):

        """Forward pass of network."""
        x = self.conv_0(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        # Flatten imput (from [batch_size, 1, 28, 28] to [batch_size, 1, 784])
        x = x.flatten(start_dim = 2)
        # Flatten imput (from [batch_size, 1, 28, 28] to [batch_size, 1, 784])
        x = x.squeeze(dim = 1)
        # print(x.shape)
        x = self.dense_0(x)

        return x
