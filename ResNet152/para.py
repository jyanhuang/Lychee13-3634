import torch
from model import resnet18  # 假设您的模型定义在'your_model_file.py'文件中

model = resnet18()  # 初始化EfficientNetV2-S模型
total = sum([param.nelement() for param in model.parameters()])

print("Number of parameter: %.2fM" % (total / 1e6))