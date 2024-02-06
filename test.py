import torch
from dataloader import test_loader, dataset, test_size
from train import model

model.eval()
predictions = []
labels = []

with torch.no_grad():
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, labels)
        _, predict = torch.max(outputs, 1)
        predictions.extend(predict.cpu().numpy())
        labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(labels, predictions)

        if batch_idx % 10 == 0:
            print(f"Test batch {batch_idx} Accuracy {accuracy}: Loss --- {loss.item()}")

