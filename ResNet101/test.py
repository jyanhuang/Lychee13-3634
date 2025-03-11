from model import resnet50
from PIL import Image
import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import time

start_time = time.time()
# 设置图像尺寸
im_height = 224
im_width = 224

img_dir = 'F:/zhenghanling/Resnet-101/data/val'
# 假设每个类别的子文件夹名称即为类别标签
class_folders = os.listdir(img_dir)

# 加载class_indices.json中的标签字典
try:
    with open('./class_indices.json', 'r') as json_file:
        class_indices = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# 反转class_indices字典，以从标签索引映射到类别名称
index_to_class = {v: k for k, v in class_indices.items()}

# 加载模型
feature = resnet50(num_classes=len(class_indices), include_top=False)
feature.trainable = False
model = tf.keras.models.Sequential([
    feature,
    tf.keras.layers.GlobalAvgPool2D(),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(len(class_indices), activation='softmax')
])

# 加载训练好的模型权重
model.load_weights('./save_weights/resNet_101.ckpt')

# 初始化预测结果列表
predictions = []
true_labels = []

# 遍历每个类别的文件夹
for class_folder in class_folders:
    class_path = os.path.join(img_dir, class_folder)
    if not os.path.isdir(class_path):
        continue

    # 遍历该类别文件夹下的每个图像文件
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # 加载并预处理图像
        img = load_img(img_path, target_size=(im_height, im_width))
        img_tensor = img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor = tf.keras.applications.resnet50.preprocess_input(img_tensor)

        # 进行预测
        preds = model.predict(img_tensor)
        prediction = np.argmax(preds)

        # 记录预测结果和真实标签
        predictions.append(prediction)
        true_labels.append(class_indices[class_folder])

# 计算准确率
num_correct = sum(p == t for p, t in zip(predictions, true_labels))
accuracy = num_correct / len(predictions)
print(f'批量预测准确率: {accuracy:.4f}')

end_time=time.time()

print('time:',end_time-start_time)