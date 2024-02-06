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
        self.layer1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.activation2 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(2, stride=1)
        self.layer3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.activation3 = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(64)
        self.maxPool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(64*3*3 , 128)
        self.activation4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, 256)
        self.activation5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.25)
        self.output = nn.Linear(256, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.maxPool1(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.batchNorm(x)
        x = self.maxPool2(x)
        #x = torch.flatten(x, 1)
        x = x.view(-1, 64 * 3 * 3)
        x = self.fc1(x)
        x = self.activation4(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.activation5(x)
        x = self.dropout2(x)
        x = self.output(x)
        return x

model = watermarkCNN().to(device)

#Defining loss function and and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
