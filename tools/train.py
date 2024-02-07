import os
import torch
import sys
import argparse
import torch.nn as nn
import torch.optim as optim
import wandb

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset.dataloader import train_loader
from models.model import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.init(project="watermarked_image_classification")

def train_model(epochs, criterion, optimizer):
    for epoch in range(epochs):
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Train batch {batch_idx}: Loss --- {loss.item()}")
            wandb.log({"Loss": loss.item()})
            
        #os.makedirs('outputs/model_weights', exist_ok=True)
        torch.save(model.state_dict(), "outputs/model_weights.pth")

if __name__== "__main__":
    epochs = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_model(epochs, criterion, optimizer)
