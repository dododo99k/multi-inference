import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models

epoch_num = 100
# define data tansform function
weights = models.ResNet50_Weights.DEFAULT
transform = weights.transforms()

# load CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=16)

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=16)

# define ResNet-50_cifar10
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes

# cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# net = torch.compile(net)
# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# train
print('start training')
for epoch in range(epoch_num): 
    model.train()
    running_loss = 0.0
    record_time = time.perf_counter()
    for i, data in enumerate(trainloader, 0):
        # input
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # clear gradient
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # back propagation
        loss.backward()
        # update weights
        optimizer.step()

        # print some information
        running_loss += loss.item()
        # if i % 100 == 99: 
        #     time_length = time.perf_counter() - record_time
        #     print(f'Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}, time usage: {time_length}')
        #     running_loss = 0.0
        #     record_time = time.perf_counter()
    time_length = time.perf_counter() - record_time
    print(f'Epoch {epoch + 1}, loss: {running_loss / i:.3f}, time usage: {time_length}')
    # update learning rate
    scheduler.step()

print('Finished Training')

# validate
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

# save
torch.save(model, './weights/resnet50.pth')


