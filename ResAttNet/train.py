# from __future__ import print_function, division
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# from torchvision import transforms, datasets, models
# import time
# from torch.utils.data import DataLoader
# from sklearn.metrics import precision_score, recall_score, f1_score
#
# # Model definition (Residual Attention Network)
# from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
#
# # Check if GPU is available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # File paths and settings
# model_file = 'model_92_sgd.pkl'
# start_time = time.time()
#
# # Function to evaluate model performance
# def test(model, test_loader, model_file):
#     model.load_state_dict(torch.load(model_file))
#     model.eval()
#
#     correct_top1 = 0
#     total = 0
#     class_correct = [0.0] * 131  # Number of classes in your custom dataset
#     class_total = [0.0] * 131   # Number of classes in your custom dataset
#     predictions = []
#     true_labels = []
#
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#
#             # Top-1 accuracy
#             _, predicted_top1 = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct_top1 += (predicted_top1 == labels).sum().item()
#
#             # Collect predictions and true labels for further evaluation
#             predictions.extend(predicted_top1.cpu().numpy())
#             true_labels.extend(labels.cpu().numpy())
#
#             # Class-wise accuracy
#             for i in range(labels.size(0)):
#                 label = labels[i]
#                 class_correct[label] += 1 if label == predicted_top1[i] else 0
#                 class_total[label] += 1
#
#     # Calculate metrics
#     accuracy_top1 = correct_top1 / total if total > 0 else 0.0
#     overall_accuracy = sum(class_correct) / sum(class_total) if sum(class_total) > 0 else 0.0
#
#     print('Overall Accuracy: {:.2f}%'.format(100 * overall_accuracy))
#     print('Top-1 Accuracy: {:.2f}%'.format(100 * accuracy_top1))
#
#     precision = precision_score(true_labels, predictions, average='macro')
#     recall = recall_score(true_labels, predictions, average='macro')
#     f1 = f1_score(true_labels, predictions, average='macro')
#
#     print('Precision: {:.2f}'.format(precision))
#     print('Recall: {:.2f}'.format(recall))
#     print('F1-score: {:.2f}'.format(f1))
#
#     # Print class-wise accuracy
#     for i in range(131):  # Assuming 131 classes based on your dataset
#         if class_total[i] > 0:
#             print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
#         else:
#             print('Accuracy of %5s : N/A (no samples)' % (classes[i]))
#
#     return accuracy_top1, precision, recall, f1
#
# # Image transformations and datasets
# transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# test_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# # Custom dataset based on CIFAR-10 structure
# train_dataset = datasets.CIFAR10(root='./data/', train=True, transform=transform, download=False)
# test_dataset = datasets.CIFAR10(root='./data/', train=False, transform=test_transform)
#
# train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=0)#4,0
# test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=0)#4,0
#
# # Classes for your custom dataset
# classes = ('Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3',
#            'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry',
#            'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow',
#            'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink',
#            'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats',
#            'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat', 'Nut Forest',
#            'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle',
#            'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk',
#            'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince',
#            'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red',
#            'Tomato Heart', 'Tomato Maroon', 'Tomato not Ripened', 'Tomato Yellow', 'Walnut', 'Watermelon')
#
# # Model initialization and optimization
# model = ResidualAttentionModel().to(device)
# lr = 0.1
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
# is_train = True
# is_pretrain = False
# acc_best = 0
# total_epoch = 184
#
# # Training loop
# if is_train:
#     if is_pretrain:
#         model.load_state_dict(torch.load(model_file))
#
#     for epoch in range(total_epoch):
#         model.train()
#         epoch_start_time = time.time()
#         for i, (images, labels) in enumerate(train_loader):
#             images = images.to(device)
#             labels = labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             if (i+1) % 100 == 0:
#                 print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, total_epoch, i+1, len(train_loader), loss.item()))
#
#         epoch_time = time.time() - epoch_start_time
#         print('Epoch {} takes {:.2f} seconds'.format(epoch + 1, epoch_time))
#         print('Evaluating test set:')
#         overall_acc, prec, rec, f1 = test(model, test_loader, model_file)
#
#         if overall_acc > acc_best:
#             acc_best = overall_acc
#             print('Current best accuracy: {:.2f}'.format(acc_best))
#             torch.save(model.state_dict(), model_file)
#
#         if (epoch + 1) / total_epoch in [0.3, 0.6, 0.9]:
#             lr /= 10
#             print('Reset learning rate to: {:.5f}'.format(lr))
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr
#                 print('Current learning rate: {:.5f}'.format(param_group['lr']))
#
#     torch.save(model.state_dict(), 'last_model_92_sgd.pkl')
#
# else:
#     test(model, test_loader, model_file)
#
# # Calculate model parameters
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('Total model parameters (M):', total_params / 1e6)  # Convert to millions
#
# end_time = time.time()
# print('Total time:', end_time - start_time, 's')



# EPOCH=6 IS BEST
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
import os
import cv2
import time
from sklearn.metrics import precision_score, recall_score, f1_score
# from model.residual_attention_network_pre import ResidualAttentionModel
# based https://github.com/liudaizong/Residual-Attention-Network
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel

import torch
#print("PyTorch version:", torch.__version__)



model_file = 'last_model_92_sgd.pkl'

start_time = time.time()
# for test
def test(model, test_loader, btrain=False, model_file='last_model_92_sgd.pkl'):
    # Test
    if not btrain:
        model.load_state_dict(torch.load(model_file))
    model.eval()


    correct_top1 = 0
    correct_top5 = 0
    total = 0
    #
    class_correct = list(0. for i in range(131))#10
    class_total = list(0. for i in range(131))#10
    predictions = []
    true_labels = []

    for images, labels in test_loader:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs = model(images)

        # Top-1 accuracy
        _, predicted_top1 = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_top1 += (predicted_top1 == labels.data).sum()

        # Top-5 accuracy
        _, predicted_top5 = torch.topk(outputs.data, 5, dim=1)
        for i in range(labels.size(0)):
            label = labels.data[i]
            class_correct[label] += 1 if label in predicted_top5[i] else 0
            class_total[label] += 1

        # Collect predictions and true labels for further evaluation
        predictions.extend(predicted_top1.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    # Calculate overall accuracy
    accuracy_top1 = float(correct_top1) / total if total > 0 else 0.0
    accuracy_top5 = sum(class_correct) / sum(class_total) if sum(class_total) > 0 else 0.0
    overall_accuracy = float(sum(class_correct)) / sum(class_total) if sum(class_total) > 0 else 0.0

    print('Overall Accuracy: {:.2f}%'.format(100 * overall_accuracy))
    print('Top-1 Accuracy: {:.2f}%'.format(100 * accuracy_top1))
    print('Top-5 Accuracy: {:.2f}%'.format(100 * accuracy_top5))

    # Calculate precision, recall, and F1-score
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')

    print('Precision: {:.2f}'.format(precision))
    print('Recall: {:.2f}'.format(recall))
    print('F1-score: {:.2f}'.format(f1))

    # for i in range(131):#10
    #     if class_total[i] > 0:
    #         print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    #     else:
    #         print('Accuracy of %5s : N/A (no samples)' % (classes[i]))

    return overall_accuracy, accuracy_top1, accuracy_top5, precision, recall, f1



# Image Preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((32, 32), padding=4),   #left, top, right, bottom
    # transforms.Scale(224),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.ToTensor()
])
# when image is rgb, totensor do the division 255
# CIFAR-10 Dataset
train_dataset = datasets.CIFAR10(root='./data/',
                               train=True,
                               transform=transform,
                               download=False)# true

test_dataset = datasets.CIFAR10(root='./data/',
                              train=False,
                              transform=test_transform)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=4, #64,4
                                           shuffle=True, num_workers=0)#8
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=4,#20,4
                                          shuffle=False)

classes = ('Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3',
           'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry',
           'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow',
           'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink',
           'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats',
           'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat', 'Nut Forest',
           'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle',
           'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk',
           'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince',
           'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red',
           'Tomato Heart', 'Tomato Maroon', 'Tomato not Ripened', 'Tomato Yellow', 'Walnut', 'Watermelon')  #classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# num_classes =131  # 更新类别数
# model = ResidualAttentionModel_92_32input_update(num_classes=num_classes)
model = ResidualAttentionModel().cuda()
#print(model)

lr = 0.1  # 0.1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
is_train =False
is_pretrain = False
total_epoch = 184#184
acc_best = 0
if is_train is True:
    if is_pretrain == True:
        model.load_state_dict((torch.load(model_file)))
    # Training
    for epoch in range(total_epoch):
        model.train()
        tims = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.cuda())
            # print(images.data)
            labels = Variable(labels.cuda())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print("hello")
            if (i+1) % 100 == 0:
                print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (
                epoch + 1, total_epoch, i + 1, len(train_loader), loss.item()))
                #print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %(epoch+1, total_epoch, i+1, len(train_loader), loss.data[0]))


        print('Epoch {} takes {:.2f} seconds'.format(epoch + 1, time.time() - tims))
        print('Evaluating test set:')
        overall_acc, acc_top1, acc_top5, prec, rec, f1 = test(model, test_loader, btrain=True)

        if acc_top1 > acc_best:
            acc_best = acc_top1
            print('Current best accuracy: {:.2f}'.format(acc_best))
            torch.save(model.state_dict(), model_file)

        if (epoch + 1) / float(total_epoch) in [0.3, 0.6, 0.9]:
            lr /= 10
            print('Reset learning rate to: {:.5f}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print('Current learning rate: {:.5f}'.format(param_group['lr']))

    torch.save(model.state_dict(), 'last_model_92_sgd.pkl')

else:
    test(model, test_loader, btrain=False)

# Calculate model parameters (M)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total model parameters (M):', total_params / 1e6)  # Convert to millions

end_time = time.time()
print('Time:', end_time - start_time,'s')








