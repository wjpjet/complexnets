import torch
import torchvision
from CLFM_Dataset import CLFM_Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch import optim, nn
from tqdm import tqdm

from models import AlexNet

# Datasets
train_dataset = CLFM_Dataset(".", train=True)
val_dataset = CLFM_Dataset(".", train=False)
test_dataset = CLFM_Dataset(".", train=False, test=True)

# Samplers
train_size, val_size = int(80000), int(10000)
train_indices, val_indices, test_indices = list(range(train_size)), list(range(val_size)), list(range(val_size))
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sample = SubsetRandomSampler(val_indices)

# Dataloaders
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                sampler=test_sample)

# GPU setup
device = torch.device('cuda')

net = AlexNet(in_channel=2, classes=10).to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)#, momentum=0.9)

# Train loop
num_epochs = 25
for epoch in range(num_epochs):


    print("Epoch: {} - Train".format(epoch))
    net.train()
    running_loss = 0.
    # Train:   
    for batch_index, (signals, labels) in enumerate(tqdm(train_loader)):

        signals, labels = signals.to(device=device), labels.to(device=device)

        optimizer.zero_grad()

        outputs = net(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #if batch_index > 50:
        # if batch_index % 100 == 0:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, batch_index + 1, running_loss / (batch_index+1) ))
        #     running_loss = 0.0

    print("Avg Loss Train: {}".format(running_loss/len(train_loader)))

    # Validation
    print("Epoch: {} - Val".format(epoch))
    net.eval()
    running_loss = 0.
    with torch.no_grad():
        for batch_index, (signals, labels) in enumerate(tqdm(val_loader)):

            signals, labels = signals.to(device=device), labels.to(device=device)

            outputs = net(signals)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

        print("Avg Loss Val: {}".format(running_loss/len(val_loader)))