import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from vit_model import vit_base_patch16_224_in21k as create_model

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的均值和标准差
])

# 使用ImageFolder自动加载数据
data_dir = 'F:/zhenghanling/Fruits-360-dataset/fruits-360/val'  # 替换为你的数据集根目录
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

# 加载预训练的ResNet-50模型，但不包括fc层
model = create_model(num_classes=131, has_logits=False)


# 加载你自己的模型权重
model_weights = torch.load('weights/best_model.pth')
model.load_state_dict(model_weights)  # 加载权重到模型中

# 将模型设置为评估模式
model.eval()

# 如果使用GPU，则移动模型到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 初始化top1和top5的计数器
top1_correct = 0
top5_correct = 0
total = 0

with torch.no_grad():
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)  # 将数据和标签移动到同一设备上
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
# 初始化预测正确的正例计数和真实正例的总计数
true_positives = 0
total_positives = 0

# 进行批量预测
with torch.no_grad():
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # 更新真实正例的总计数
        total_positives += labels.size(0)

        # 计算每个批次的TP（即预测正确的正例）
        batch_true_positives = (predicted == labels).sum().item()

        # 更新TP的总计数
        true_positives += batch_true_positives

# 计算召回率
recall = 100 * true_positives / total_positives if total_positives > 0 else 0
print(f'Recall of the model on the validation images: {recall:.2f}%')
