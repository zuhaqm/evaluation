import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = 0
def create_dataloader(data_path, batchSize, split_metric):
    #define transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    #Define datapath aswell as do train test split and create train test loaders
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    train_size = int(split_metric * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True)
    return train_loader, test_loader


data_path = "./watermark_dataset"
batch_size = 32
split_metric = 0.8
train_loader, test_loader = create_dataloader(data_path, batch_size, split_metric)

