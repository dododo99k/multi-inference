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
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=16)

# define ResNet-50_cifar10
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes
# load the original weights
model = torch.load('./weights/resnet50.pth')

class EarlyExitResNet50(nn.Module):
    def __init__(self, original_model, num_classes=1000):
        super(EarlyExitResNet50, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-3])
        # Freeze the features part
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Early exit branch
        self.early_exit = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # Conv layer with 512 output channels
            nn.BatchNorm2d(512),                            # BatchNorm layer
            nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # Early exit logic
        early_exit_output = self.early_exit(x)
        
        # Continue with the remaining layers
        # main_exit_output = self.main_exit(x)
        
        return early_exit_output# , main_exit_output

# define ee_model
num_classes = 10  # 根据你的任务更改类别数量
ee_model = EarlyExitResNet50(model, num_classes)


# cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ee_model.to(device)


# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, ee_model.parameters()), lr=0.001, momentum=0.9)

# optimizer = optim.SGD(ee_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# train
print('start training')
for epoch in range(epoch_num):
    ee_model.train()
    running_loss = 0.0
    epoch_start_time = time.perf_counter()
    record_time = time.perf_counter()
    for i, data in enumerate(trainloader, 0):
        # input
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # clear gradient
        optimizer.zero_grad()

        # forward
        outputs = ee_model(inputs)
        loss = criterion(outputs, labels)
        # back propagation
        loss.backward()
        # update weights
        optimizer.step()

        # print some information
        running_loss += loss.item()
        if i % 100 == 99:
            time_length = time.perf_counter() - record_time
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}, time usage: {time_length}')
            running_loss = 0.0
            record_time = time.perf_counter()

    print(f'Epoch {epoch + 1}, time usage: {time.perf_counter() - epoch_start_time}')
    # update learning rate
    scheduler.step()

print('Finished Training')

# validate
ee_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = ee_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

# save
torch.save(ee_model, './weights/resnet50_EE.pth')
