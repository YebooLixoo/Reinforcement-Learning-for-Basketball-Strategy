import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)  
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)

        self.fc1 = nn.Linear(in_features=65*32 + 32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, image_feature_input, flat_feature_input, batch_size, sequence_num):

        x = self.conv1(image_feature_input.reshape(-1, 3, 60, 120))
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = x.reshape(sequence_num, batch_size, -1)
        image_feature_conv_output = torch.sum(x, dim=0)
        feature_input = torch.cat([image_feature_conv_output, flat_feature_input], dim=1)
        x = self.fc1(feature_input)
        x = self.relu(x)
        x = self.fc2(x)
        return x