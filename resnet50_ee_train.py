import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models

epoch_num = 300
# define data tansform function
weights = models.ResNet50_Weights.DEFAULT
transform = weights.transforms()

# load CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=16)

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=16)

# define ResNet-50_cifar10
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes
# load the original weights
model = torch.load('./weights/resnet50.pth')

# def hook_fn(module, input, output):
#     print(f'{module}, input size: {input.size()}, output size: {output.size()}')

def check_features(sub_layers):
    for i, module in enumerate(list(sub_layers.modules())):
        try:
            final_features = module.num_features
        except:
            pass
    # print('final_features: ',final_features)
    return final_features

def calculate_loss(exit_outputs, target):
    loss_fn = nn.CrossEntropyLoss()
    loss = 0
    for i, output in enumerate(exit_outputs):
        loss += loss_fn(output, target)
    return loss

class earlyexit_ramp(nn.Module):
    def __init__(self, num_feature=1024, num_classes =10):
        super(earlyexit_ramp, self).__init__()
        self.conv = nn.Conv2d(num_feature, int(num_feature/2), kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(int(num_feature/2))
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(num_feature/2), num_classes)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class EarlyExitResNet50(nn.Module):
    def __init__(self, original_model, num_class=1000):
        super(EarlyExitResNet50, self).__init__()

        # construct full model list
        self.train_mode = True
        detail_layers = []

        # construct early exit list in 
        self.earlyest_point = -15
        self.exit_list = range(self.earlyest_point,-3,1)

        for i, child in enumerate(list(original_model.children())):
            if len(list(child.children())) == 0:
                # print(i, child)
                detail_layers.append(child)
            else: # 
                for j, sub_child in enumerate(list(child.children())):
                    # print(i,j, sub_child)
                    # print('***********************************')
                    detail_layers.append(sub_child)

        self.features = nn.Sequential(*list(detail_layers[:-15])) # orginal top half model, from -2 to ....
        self.next_layers = nn.ModuleList() # orginal later model layers, 12 items
        self.exit_fcs = nn.ModuleList() # 13 items, not including full model exit

        self.original_fc = nn.ModuleList()
        self.original_fc.append(detail_layers[-3])
        self.original_fc.append(detail_layers[-2])
        self.original_fc.append(nn.Flatten(start_dim=1))
        self.original_fc.append(detail_layers[-1])
            
        print('features output:', (check_features(self.features)))
        self.exit_fcs.append(earlyexit_ramp(check_features(self.features), num_class)) # earlyest exit


        for i in range(-15,-3,1): #[-15,-14,...,-3] 13 ramps
            # tmp_layer = nn.Sequential(*list(detail_layers[i].children()))
            tmp_layer = detail_layers[i]
            self.next_layers.append(tmp_layer)
            print(i,' next layers output:', (check_features(tmp_layer)))
            for j, child in enumerate(list(tmp_layer.children())):
                print(j,child)
            print()
            self.exit_fcs.append(earlyexit_ramp(check_features(tmp_layer), num_class))
        
        print('full model exit')
        for j, child in enumerate(list(self.original_fc.modules())):
            print(j,child)
        # Freeze the features part
        for param in self.features.parameters():
            param.requires_grad = False
        for layer in self.next_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.original_fc.parameters():
            param.requires_grad = False
        
        print('middle layers number:', len(self.next_layers))
        print('early exit ramps number:', len(self.exit_fcs))
        
    def forward(self, x, ramp = None):
        early_exit_output = []
        x = self.features(x) # 
        if self.train_mode:
            early_exit_output.append(self.exit_fcs[0](x))

            for i,layer in enumerate(self.next_layers):
                x = layer(x)
                early_exit_output.append(self.exit_fcs[i+1](x))
            # early_exit_output.append(self.exit_fcs[i+1](x))
        else: # inference mode
            if type(ramp)==int: # early exit
                # for layer in self.next_layers:
                for i in range(ramp): # [-15,-14,-13,...,-2] -> [0,1,2,3,4,...] (ramps id)
                    x = self.next_layers[i](x)

                early_exit_output.append(self.exit_fcs[ramp](x))

            elif ramp == None: # full resnet model self.exit_list[-1]
                for i in range(len(self.next_layers)):
                    x = self.next_layers[i](x)
                for i in range(len(self.original_fc)):
                    x = self.original_fc[i](x)
                    # print(x.size())

                early_exit_output.append(x)
        
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
pass
optimizer = optim.SGD(filter(lambda p: p.requires_grad, ee_model.parameters()), lr=0.001, momentum=0.9)

# optimizer = optim.SGD(ee_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

pass
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
        outputs_list = ee_model(inputs)
        loss = calculate_loss(outputs_list, labels)
        # loss = criterion(outputs, labels)
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
ee_model.train_mode = False
correct = 0
total = 0
ramp_id = 0
with torch.no_grad():
    for ramp_id in range(13):
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = ee_model(images, ramp_id)
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'ramp id: {ramp_id}, Accuracy of the network on the 10000 test images: {100 * correct / total} %')

# save
torch.save(ee_model, './weights/resnet50_EE.pth')
