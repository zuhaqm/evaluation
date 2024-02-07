import torch
import wandb
from sklearn.metrics import accuracy_score
import sys, os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="watermarked_image_classification")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset.dataloader import test_loader
from models.model import model


def test_model(predictions, true_labels):
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            batch_accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
            print(f"Test batch {batch_idx} Accuracy: {batch_accuracy}")
            wandb.log({"Test Accuracy": batch_accuracy})

if __name__== "__main__":
    model.eval()##
    predictions = []
    true_labels = []
    test_model(predictions, true_labels)

