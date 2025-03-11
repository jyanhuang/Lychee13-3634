import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from PIL import Image

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的均值和标准差
])

# 使用ImageFolder自动加载数据
data_dir = 'F:/zhenghanling/Resnet-101/data/val'  # 替换为你的数据集根目录
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

# 加载预训练的ResNet模型（或你自己的模型）
model = models.resnet101(pretrained=False)  # 假设你有一个已经训练好的模型
model.eval()
model.cuda() if torch.cuda.is_available() else model.cpu()  # 如果有GPU则使用GPU

# 初始化top1和top5的计数器
top1_correct = 0
top5_correct = 0
total = 0

with torch.no_grad():
    for images, labels in data_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        _, top5 = outputs.topk(5, 1, True, True)

        # 计算top1和top5的准确率
        top1_correct += (predicted == labels).sum().item()
        top5_correct += (top5 == labels.view(-1, 1).expand_as(top5)).any(1).sum().item()
        total += labels.size(0)

# 计算并打印top1和top5的准确率
top1_accuracy = top1_correct / total
top5_accuracy = top5_correct / total
print(f'Top-1 accuracy: {top1_accuracy * 100:.2f}%')
print(f'Top-5 accuracy: {top5_accuracy * 100:.2f}%')