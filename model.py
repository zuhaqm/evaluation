import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model class with architecture and forward prop
import torch
import torch.nn as nn

class watermarkCNN(nn.Module):
    def __init__(self):
        super(watermarkCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm2d(64)
        
        # AdaptiveAvgPool2d to adjust dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))  
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(256, 2)  

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.batchnorm(x)
        x = self.avgpool(x)
        # Reshape for fully connected layers
        x = x.view(-1, 64 * 8 * 8)  
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = watermarkCNN().to(device)

#Defining loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
