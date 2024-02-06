import torch
from torch.utils.data import DataLoader
from dataloader import train_loader, dataset, train_size
from model import model, criterion, optimizer
from torch.utils.data import random_split

epochs = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

##############OVERFIT ON SUBSET########################################
# subset_size = int(len(train_loader.dataset) * 0.05)

# subset_train_dataset, _ = random_split(train_loader.dataset, [subset_size, len(train_loader.dataset) - subset_size])
# subset_train_loader = DataLoader(subset_train_dataset, batch_size=32, shuffle=False)

# for epoch in range(epochs):
#     for batch_idx, (data, labels) in enumerate(subset_train_loader):
#         data, labels = data.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         print(f"Train batch {batch_idx}: Loss --- {loss.item()}")

#########################TRAINING######################################
# for epoch in range(epochs):
#     for batch_idx, (data, labels) in enumerate(train_loader):
#         data, labels = data.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         print(f"Train batch {batch_idx}: Loss --- {loss.item()}")

#####################TESTING########################################
from dataloader import test_loader
import wandb
from sklearn.metrics import accuracy_score

wandb.init(project="watermarked_image_classification")
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        batch_accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
        print(f"Test batch {batch_idx} Accuracy: {batch_accuracy}")

accuracy = accuracy_score(true_labels, predictions)
wandb.log({"Test Accuracy": accuracy})


