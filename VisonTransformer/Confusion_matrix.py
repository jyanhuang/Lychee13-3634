import os
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model  # 假设你使用的是TensorFlow的Keras API
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# 加载你的模型（假设模型已经被保存为'model.h5'）
import torch

# 假设您的模型类定义为 MyModel，且定义了加载权重的方法
model = vit_base_patch16_224_in21k()
model.load_state_dict(torch.load('weights/best_model.pth'))
model.eval()  # 设置模型为评估模式

# 设定数据集的路径
data_dir = 'F:/zhenghanling/VIT/data/val'

# 设定ImageDataGenerator，用于预处理图像（如果需要的话）
test_datagen = ImageDataGenerator(rescale=1./255)

# 使用ImageDataGenerator的flow_from_directory方法来读取测试数据
test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # 假设你的模型输入大小为224x224
    batch_size=32,
    class_mode='categorical',  # 假设是多分类问题
    shuffle=False)  # 测试时不需要打乱数据

# 预测所有测试数据
y_pred = model.predict_generator(test_generator, steps=len(test_generator))
y_pred_classes = np.argmax(y_pred, axis=1)  # 将预测的概率转换为类别

# 获取真实的标签
y_true = test_generator.classes

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred_classes)

# 打印混淆矩阵
print(cm)

# 如果你想要更清晰的输出，可以使用matplotlib或seaborn来可视化混淆矩阵
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()