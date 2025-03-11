# import torch
# from model import resnet152
# from PIL import Image
# from torchvision import transforms
# import matplotlib.pyplot as plt
# import json
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# data_transform = transforms.Compose(
#     [transforms.Resize(256),
#      transforms.CenterCrop(224),
#      transforms.ToTensor(),
#      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 预处理
#
# # load image
# img = Image.open("./huanglongbing.JPG")  # 导入需要检测的图片
# plt.imshow(img)
# # [N, C, H, W]
# img = data_transform(img)
# # expand batch dimension
# img = torch.unsqueeze(img, dim=0)
#
# # read class_indict
# try:
#     json_file = open('./class_indices.json', 'r')
#     class_indict = json.load(json_file)
# except Exception as e:
#     print(e)
#     exit(-1)
#
# # create model
# model = resnet152(num_classes=13)  # 修改为你训练时一共的种类数
# # load model weights
# model_weight_path = "./resNet152.pth"  # 导入训练好的模型
# model.load_state_dict(torch.load(model_weight_path, map_location=device))
# model.eval()
# with torch.no_grad():  # 不对损失梯度进行跟踪
#     # predict class
#     output = torch.squeeze(model(img))  # 压缩batch维度
#     predict = torch.softmax(output, dim=0)  # 得到概率分布
#     predict_cla = torch.argmax(predict).numpy()  # argmax寻找最大值对应的索引
# print(class_indict[str(predict_cla)], predict[predict_cla].numpy())
# plt.show()
import torch
from model import resnet152
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os
import time
from glob import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 读取类别索引
try:
    with open('./class_indices.json', 'r') as json_file:
        class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

start_time = time.time()
# 创建模型并加载权重
model = resnet152(num_classes=131)
model_weight_path = "./resNet152.pth"  # 导入训练好的模型
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()
model.to(device)

# 遍历test文件夹下的所有子文件夹
test_dir = 'F:/zhenghanling/Fruits-360-dataset/fruits-360/val'
for class_folder in os.listdir(test_dir):
    class_folder_path = os.path.join(test_dir, class_folder)
    if os.path.isdir(class_folder_path):
        print(f"Processing {class_folder} images...")
        for img_path in glob(os.path.join(class_folder_path, '*.jpg')):  # 假设图片是jpg格式
            img = Image.open(img_path)
            img_tensor = data_transform(img).to(device)
            img_batch = torch.unsqueeze(img_tensor, dim=0)

            with torch.no_grad():
                output = model(img_batch)
                predict = torch.softmax(output, dim=1)
                predict_cla = torch.argmax(predict).item()

            print(f"Predicted: {class_indict[str(predict_cla)]}, Probability: {predict[0, predict_cla].item()}")

end_time = time.time()

# 计算并打印运行时间
elapsed_time = end_time - start_time
print(f"Total Elapsed Time: {elapsed_time:.4f} seconds")