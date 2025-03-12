# import torch
# import torch.nn as nn
# from utils.readData import read_dataset
# from utils.ResNet import ResNet18
# import time
# import numpy as np #+
# from collections import defaultdict #+
# from sklearn.metrics import precision_score, f1_score #+
# # set device
# start_time=time.time()
# # 创建ResNet18模型实例
# model = ResNet18()
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# n_class = 13#10
# batch_size = 100
# train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size,pic_path='dataset')
# model = ResNet18() # 得到预训练模型
# model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
# model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层修改
# # 载入权重
# model.load_state_dict(torch.load('checkpoint/resnet18_cifar10.pt'))
# model = model.to(device)
#
# total_sample = 0
# right_sample = 0
#
# # 统计模型参数数量
# total_params = 0
# model.eval()  # 验证模型
# for data, target in test_loader:
#     data = data.to(device)
#     target = target.to(device)
#     # forward pass: compute predicted outputs by passing inputs to the model
#     output = model(data).to(device)
#     # convert output probabilities to predicted class(将输出概率转换为预测类)
#     _, pred = torch.max(output, 1)
#     # compare predictions to true label(将预测与真实标签进行比较)
#     correct_tensor = pred.eq(target.data.view_as(pred))
#     # correct = np.squeeze(correct_tensor.to(device).numpy())
#     total_sample += batch_size
#     for i in correct_tensor:
#         if i:
#             right_sample += 1
# print("Accuracy:",100*right_sample/total_sample,"%")
# for params in model.parameters():
#     total_params += params.numel()
#
#
# # 初始化统计变量
# top1_correct = 0
# top5_correct = 0
# total_samples = 0
# class_correct = defaultdict(int)
# class_total = defaultdict(int)
# class_predictions = []
# class_targets = []
#
#
# # 设置模型为评估模式
# model.eval()
#
# # 定义计算Top-k准确率的函数
# def accuracy(output, target, topk=(1,)):
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res
#
# # 计算Top-1和Top-5准确率
# with torch.no_grad():
#     for data, target in test_loader:
#         data = data.to(device)
#         target = target.to(device)
#
#         # forward pass: compute predicted outputs by passing inputs to the model
#         output = model(data)
#
#         # convert output probabilities to predicted class
#         _, pred = torch.max(output, 1)
#
#         # compare predictions to true label
#         correct_tensor = pred.eq(target.data.view_as(pred))
#
#         # convert correct_tensor to numpy array
#         correct = np.squeeze(correct_tensor.cpu().numpy())
#
#         # 记录每个类别的预测和目标值
#         class_predictions.extend(pred.cpu().numpy())
#         class_targets.extend(target.cpu().numpy())
#
#         # 计算每个类别的 true positives 和 actual positives
#         for i in range(len(target)):
#             label = target[i].item()
#             class_correct[label] += correct[i].item()
#             class_total[label] += 1
#
#         # 计算topk准确率
#         top1, top5 = accuracy(output, target, topk=(1, 5))
#
#         # 累积准确率和样本数
#         top1_correct += top1.item() * data.size(0)
#         top5_correct += top5.item() * data.size(0)
#         total_samples += data.size(0)
#
# # 计算平均准确率#+
# top1_accuracy = top1_correct / total_samples
# top5_accuracy = top5_correct / total_samples
#
# print(f'Top-1 Accuracy: {top1_accuracy:.2f}%')
# print(f'Top-5 Accuracy: {top5_accuracy:.2f}%')
#
# print(f"Total number of parameters in the model: {total_params}")
#
#
# # 计算精确度和F1-score
# precision = precision_score(class_targets, class_predictions, average='macro')
# f1 = f1_score(class_targets, class_predictions, average='macro')
#
# print(f"Precision: {precision:.2f}")
# print(f"F1-score: {f1:.2f}")
#
# # 计算并打印每个类别的召回率
# for i in range(n_class):
#     recall = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
#     print(f"Recall of class {i}: {recall:.2f}%")
#
# # 可以选择打印所有类别的平均值
# overall_recall = np.sum(list(class_correct.values())) / np.sum(list(class_total.values()))
# print(f"Overall Recall: {overall_recall:.2f}%")
# end_time=time.time()
# print('time:',end_time-start_time)

import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from utils.readData import read_dataset  # Assuming this is your data reading function
from utils.ResNet import ResNet18  # Assuming this is your ResNet model
import time

# Set device
start_time = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Constants
n_class = 131
batch_size = 100
pic_path = 'dataset'  # Assuming this is your dataset path

# Load dataset (assuming read_dataset returns train_loader, valid_loader, test_loader)
train_loader, valid_loader, test_loader = read_dataset(batch_size=batch_size, pic_path=pic_path)

# Load pre-trained ResNet18 model
model = ResNet18()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = nn.Linear(512, n_class)

# Load weights
model.load_state_dict(torch.load('checkpoint/resnet18_cifar10.pt'))
model = model.to(device)


# Function to calculate metrics
def calculate_metrics(model, data_loader):
    total_samples = 0
    right_samples = 0
    all_preds = []
    all_targets = []

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            # Forward pass
            output = model(data)
            _, preds = torch.max(output, 1)

            # Collect predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # Calculate number of correct predictions
            correct_tensor = preds.eq(target)
            correct = correct_tensor.sum().item()
            right_samples += correct
            total_samples += data.size(0)

    accuracy = 100.0 * right_samples / total_samples
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1score = f1_score(all_targets, all_preds, average='macro')

    return accuracy, precision, recall, f1score

# Calculate metrics on test set
accuracy, precision, f1score, recall = calculate_metrics(model, test_loader)
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"F1-score: {f1score:.4f}")
print(f"Recall: {recall:.4f}")


# Function to calculate Top-k accuracy
def topk_accuracy(model, data_loader, topk=(1, 5)):
    model.eval()
    correct = [0] * len(topk)
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            _, pred = output.topk(max(topk), 1, True, True)

            # Check top-k accuracy
            for i, k in enumerate(topk):
                correct[i] += torch.sum(torch.eq(pred[:, :k], target.view(-1, 1))).item()

            total += data.size(0)

    topk_accs = [100.0 * c / total for c in correct]
    return topk_accs


# Calculate Top-1 and Top-5 accuracies
top1_accuracy, top5_accuracy = topk_accuracy(model, test_loader)
print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

# Get model parameters count
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters (M): {num_params / 1e6} M")

end_time = time.time()
print('Time:', end_time - start_time,'s')
