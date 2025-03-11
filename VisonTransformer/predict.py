# import os
# import json
#
# import torch
# from PIL import Image
# from torchvision import transforms
# import matplotlib.pyplot as plt
#
# from vit_model import vit_base_patch16_224_in21k as create_model
#
#
# def main():
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     data_transform = transforms.Compose(
#         [transforms.Resize(256),
#          transforms.CenterCrop(224),
#          transforms.ToTensor(),
#          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
#
#     # load image
#     img_path = r"F:/zhenghanling/VIT/data/val/Blackleaf_lychee/2024-06-05 201030.jpg"
#     assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
#     img = Image.open(img_path)
#     plt.imshow(img)
#     # [N, C, H, W]
#     img = data_transform(img)
#     # expand batch dimension
#     img = torch.unsqueeze(img, dim=0)
#
#     # read class_indict
#     json_path = 'class_indices.json'
#     with open(json_path, "r") as f:
#         class_indict = json.load(f)
#
#     # create model
#     model = create_model(num_classes=12, has_logits=False).to(device)
#     # load model weights
#     model_weight_path = "./weights/best_model.pth"
#     model.load_state_dict(torch.load(model_weight_path, map_location=device))
#     model.eval()
#     with torch.no_grad():
#         # predict class
#         output = torch.squeeze(model(img.to(device))).cpu()
#         predict = torch.softmax(output, dim=0)
#         predict_cla = torch.argmax(predict).numpy()
#
#     print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
#                                                  predict[predict_cla].numpy())
#     plt.title(print_res)
#     for i in range(len(predict)):
#         print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
#                                                   predict[i].numpy()))
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()
import os
import json

import torch
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model
import time

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # 假设你有一个包含所有图像路径和对应标签的CSV文件或类似的格式
    # 这里我们使用ImageFolder来加载数据集
    dataset = datasets.ImageFolder(root="F:/zhenghanling/Fruits-360-dataset/fruits-360/val", transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # 批量大小为32

    # read class_indices
    json_path = 'class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 反转class_indict以从索引获取类名
    class_indict_inverse = {v: k for k, v in class_indict.items()}

    # create model
    model = create_model(num_classes=len(class_indict), has_logits=False).to(device)
    # load model weights
    model_weight_path = "./weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {len(dataset)} test images: {100 * correct / total} %')


if __name__ == '__main__':
    start_time = time.time()

    main()
    end_time = time.time()

    print('time:', end_time - start_time)