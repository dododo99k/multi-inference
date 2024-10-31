import torch
import torch.nn as nn
import torch.optim as optim

# 检查是否有可用的 CUDA 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义一个简单的 MLP 模型
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        # out = self.relu(out)
        out = self.fc2(out)
        return out

# 模型超参数
input_size = 10    # 输入特征的维度
hidden_size = 20   # 隐藏层的大小
output_size = 1    # 输出的维度

# 实例化模型并将其移动到 GPU 上
model = SimpleMLP(input_size, hidden_size, output_size).to(device)

# 随机生成输入数据并移动到 GPU 上
batch_size = 5
x = torch.randn(batch_size, input_size).to(device)

# 进行前向传播
output = model(x)

print("Model output:", output)
