import torch
from vit_model import vit_base_patch16_224_in21k  # 假设您的模型定义在'your_model_file.py'文件中

model = vit_base_patch16_224_in21k()  # 初始化EfficientNetV2-S模型
total = sum([param.nelement() for param in model.parameters()])

print("Number of parameter: %.2fM" % (total / 1e6))
