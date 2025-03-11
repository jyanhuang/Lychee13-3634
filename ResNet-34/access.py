import torch
import torch.nn.functional as F
from torchvision import models, transforms,datasets
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from torch.utils.data import DataLoader
from multiprocessing import freeze_support
#
# # 加载预训练的ResNet-50模型，但不包括fc层
# model = models.resnet34(pretrained=True)
#
# # 因为我们只需要特征提取，所以冻结所有参数
# for param in model.parameters():
#     param.requires_grad = False
#
# # 替换fc层为新的层，输出大小为13（荔枝类别的数量）
# num_ftrs = model.fc.in_features  # 获取fc层的输入特征数
# model.fc = torch.nn.Linear(num_ftrs, 13)  # 替换fc层
#
# # 加载你自己的模型权重
# model_weights = torch.load('resnet34.pth')
# model.load_state_dict(model_weights)  # 加载权重到模型中
# # 1. 加载模型与权重
# # 如果使用GPU，则移动模型到GPU上
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# model.eval()
#
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的均值和标准差
# ])
# # 假设您有一个DataLoader来加载评估数据集
# # data_loader = ...
# data_dir = "F:/zhenghanling/EfficientNetV2/data/val"
# data_dataset = datasets.ImageFolder(data_dir, transform=transform)
# data_loader = DataLoader(data_dataset, batch_size=32, shuffle=False, num_workers=4)
#
# # 初始化性能指标的列表
# aps = []
# accuracies = []
# precisions = []
# recalls = []
# f1s = []
#
# # 2. 准备数据集（这里仅作为示例，您需要自己的DataLoader）
# # ...
#
# # 3. 预测
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# with torch.no_grad():
#     for images, labels in data_loader:
#         images = images.to(device)
#         outputs = model(images)
#         probs = F.softmax(outputs, dim=1)
#         _, predicted = torch.max(probs, 1)
#
#         # 将预测结果和真实标签移动到CPU（如果它们在GPU上）
#         predicted = predicted.cpu().numpy()
#         labels = labels.cpu().numpy()
#
#         # 4. 计算性能指标
#         # 准确率
#         accuracy = accuracy_score(labels, predicted)
#         accuracies.append(accuracy)
#
#         # 对于每个类别计算Precision, Recall, F1-Score和AP
#         binary_labels = label_binarize(labels, classes=[i for i in range(13)])  # 假设有13个类别
#         n_labels = binary_labels.shape[1]
#         precision = precision_score(binary_labels, probs, average=None)
#         recall = recall_score(binary_labels, probs, average=None)
#         f1 = f1_score(binary_labels, probs, average=None)
#
#         # 计算每个类别的AP
#         for i in range(n_labels):
#             precision_curve, recall_curve, _ = precision_recall_curve(binary_labels[:, i], probs[:, i])
#             ap = auc(recall_curve, precision_curve)
#             aps.append(ap)
#
#         # 汇总每个类别的性能指标（可选）
#         precisions.extend(precision)
#         recalls.extend(recall)
#         f1s.extend(f1)
#
# # 计算mAP
# mAP = sum(aps) / len(aps)
#
#
# average_accuracy = sum(accuracies) / len(accuracies)
# average_precisions = sum(precisions) / len(precisions) if precisions else 0.0  # 假设所有类别的precision都被计算了
# average_recalls = sum(recalls) / len(recalls) if recalls else 0.0  # 假设所有类别的recall都被计算了
# average_f1s = sum(f1s) / len(f1s) if f1s else 0.0  # 假设所有类别的F1-Score都被计算了
#
# # 打印结果
# print(f"mAP: {mAP}")
# print(f"Average Accuracy: {average_accuracy}")
# print(f"Average Precision: {average_precisions}")  # 注意这里假设所有类别都有precision值
# print(f"Average Recall: {average_recalls}")  # 注意这里假设所有类别都有recall值
# print(f"Average F1-Score: {average_f1s}")  # 注意这里假设所有类别都有F1-Score值
#
# # 注意：上述代码可能需要调整以适应您的具体设置和数据集

import numpy as np


model = models.resnet34(pretrained=False)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 替换fc层为新的层，输出大小为13（类别数量）
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 13)

# 加载您自己的模型权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('resNet34.pth', map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的均值和标准差
])
# 假设您有一个DataLoader来加载评估数据集
# data_loader = ...
data_dir = ("F:/zhenghanling/EfficientNetV2/data/val")
data_dataset = datasets.ImageFolder(data_dir, transform=transform)
data_loader = DataLoader(data_dataset, batch_size=32, shuffle=False, num_workers=4)

# 初始化性能指标列表（针对每个类别）
aps_per_class = [0] * 13
# aps_per_class = []
precisions_per_class = []
recalls_per_class = []
accuracies = []  # 初始化准确率的列表

if __name__ == '__main__':
    freeze_support()  # 调用 freeze_support()

    # # 3. 预测（保持之前的循环）
    # with torch.no_grad():
    #     for images, labels in data_loader:
    #         images = images.to(device)
    #         outputs = model(images)
    #         probs = F.softmax(outputs, dim=1)
    #
    #         # 4. 计算性能指标（在循环内部，但需要为每个类别分别计算）
    #         for class_idx in range(13):  # 假设有13个类别
    #             # 获取每个类别的预测概率和真实标签
    #             class_probs = probs[:, class_idx]
    #             class_labels = (labels == class_idx).to(torch.long) # 将真实标签转换为二分类问题
    #
    #             # 计算Precision-Recall曲线所需的precision和recall
    #             precision, recall, _ = precision_recall_curve(class_labels, class_probs)
    #
    #             # 计算Average Precision (AP)
    #             ap = auc(recall, precision)
    #             aps_per_class.append(ap)
    #
    #             # 计算当前类别的precision, recall和f1-score（需要设定一个阈值，这里以0.5为例）
    #             threshold = 0.5
    #             class_predicted = (class_probs > threshold).to(torch.long)
    #             precision_class = precision_score(class_labels, class_predicted)
    #             recall_class = recall_score(class_labels, class_predicted)
    #             f1_class = f1_score(class_labels, class_predicted)
    #
    #             precisions_per_class.append(precision_class)
    #             recalls_per_class.append(recall_class)
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # 获取预测类别

            # 4. 计算性能指标（在循环内部，但需要为每个类别分别计算）
            for class_idx in range(13):  # 假设有13个类别
                # 获取每个类别的预测和真实标签
                class_preds = (preds == class_idx).to(torch.float32)  # 将预测类别转换为二分类问题的预测
                class_labels = (labels == class_idx).to(torch.float32)  # 将真实标签转换为二分类问题的标签

                # 计算TP（真正例）、FP（假正例）、FN（假负例）
                tp = (class_preds * class_labels).sum().item()
                fp = ((1 - class_labels) * class_preds).sum().item()
                fn = (class_labels * (1 - class_preds)).sum().item()

                # 避免除以零
                precision = tp / (tp + fp + 1e-10) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn + 1e-10) if (tp + fn) > 0 else 0

                # 因为我们没有足够的点来绘制PR曲线，我们直接计算AP
                # 在实际的多标签或目标检测任务中，我们会使用precision_recall_curve

                # 假设我们在这里简单地使用precision和recall来计算AP（这不是标准方法）
                # 在实际中，您可能需要使用更复杂的插值方法（如VOC 2012的11点插值）
                aps_per_class[class_idx] += precision * recall if recall > 0 else 0

# 计算mAP（对于图像多分类，这实际上就是AP的平均值
    mAP50 = sum(aps_per_class) / len(aps_per_class) if len(aps_per_class) > 0 else 0
    print(f"mAP50: {mAP50}")
#
# # 计算整体的性能指标（平均值）
#     mAP = np.mean(aps_per_class)
#     accuracy_list = []  # 这里我们需要收集每个batch的accuracy来得到最终accuracy
#
#     # 由于accuracy需要在整个数据集上计算，我们需要在循环外部进行
#     total_correct = 0
#     total_samples = 0
#
#     with torch.no_grad():
#         for images, labels in data_loader:
#             images = images.to(device)
#             outputs = model(images)
#             probs = F.softmax(outputs, dim=1)
#             _, predicted = torch.max(probs, 1)
#             total_correct += (predicted.cpu() == labels.cpu()).sum().item()
#             total_samples += labels.size(0)
#
#     # 计算准确率
#     accuracy = total_correct / total_samples
#
#     # 其他的性能指标已经是针对每个类别的，如果需要平均，可以直接计算
#     average_precision = np.mean(precisions_per_class)
#     average_recall = np.mean(recalls_per_class)
#     average_f1 = 2 * average_precision * average_recall / (
#                 average_precision + average_recall) if average_precision + average_recall > 0 else 0
#
#     # 打印结果
#     print(f"mAP: {mAP}")
#     print(f"Accuracy: {accuracy}")
#     print(f"Average Precision: {average_precision}")
#     print(f"Average Recall: {average_recall}")
#     print(f"Average F1-Score: {average_f1}")
