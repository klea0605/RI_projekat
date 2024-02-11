import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride = (2, 2))
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.flatten = nn.Flatten()

        self.linearStack = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        # print(f'conv1{x.shape}')
        x = self.conv2(x)
        # print(f'conv2: {x.shape}')
        x = self.conv3(x)
        # print(f'conv3: {x.shape}')

        x = self.flatten(x)
        x = self.linearStack(x)

        predictions = self.softmax(x)

        return predictions