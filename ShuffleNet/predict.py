import torch
from model import shufflenet_v2_x1_0  # 假设您的模型定义在'your_model_file.py'文件中

model =shufflenet_v2_x1_0()  # 初始化EfficientNetV2-S模型
total = sum([param.nelement() for param in model.parameters()])

print("Number of parameter: %.2fM" % (total / 1e6))