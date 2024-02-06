import torch
from dataloader import train_loader, dataset, train_size
from model import model, criterion, optimizer

epochs = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#overfitting on 5 percent of total train data
# subset_size = int(train_size * 0.05)
# remaining_size = train_size - subset_size
# print(train_size)
# print(subset_size)
# print(remaining_size)

# subset, _ = torch.utils.data.random_split(dataset, [subset_size, remaining_size])
# subset_loader = DataLoader(subset, batch_size=32, shuffle=False)
# for epoch in range(epochs):
#     for batch_idx, (data, labels) in enumerate(subset_loader):
#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#     if batch_idx % 10 == 0:
#             print(f"Train batch {batch_idx}: Loss --- {loss.item()}")


for epoch in range(epochs):
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if batch_idx % 10 == 0:
            print(f"Train batch {batch_idx}: Loss --- {loss.item()}")



