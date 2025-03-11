import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import time
import argparse
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont

# Define the classes (replace the placeholders with your actual class names)
classes = ('1 AM_General_Hummer_SUV_2000','10 Aston_Martin_Virage_Convertible_2012',
'11 Aston_Martin_Virage_Coupe_2012',
'12 Audi_RS_4_Convertible_2008',
'13 Audi_A5_Coupe_2012',
'14 Audi_TTS_Coupe_2012',
'15 Audi_R8_Coupe_2012',
'16 Audi_V8_Sedan_1994',
'17 Audi_100_Sedan_1994',
'18 Audi_100_Wagon_1994',
'19 Audi_TT_Hatchback_2011',
'2 Acura_RL_Sedan_2012',
'20 Audi_S6_Sedan_2011',
'21 Audi_S5_Convertible_2012',
'22 Audi_S5_Coupe_2012',
'23 Audi_S4_Sedan_2012',
'24 Audi_S4_Sedan_2007',
'25 Audi_TT_RS_Coupe_2012',
'26 BMW_ActiveHybrid_5_Sedan_2012',
'27 BMW_1_Series_Convertible_2012',
'28 BMW_1_Series_Coupe_2012',
'29 BMW_3_Series_Sedan_2012',
'3 Acura_TL_Sedan_2012',
'30 BMW_3_Series_Wagon_2012',
'31 BMW_6_Series_Convertible_2007',
'32 BMW_X5_SUV_2007',
'33 BMW_X6_SUV_2012',
'34 BMW_M3_Coupe_2012',
'35 BMW_M5_Sedan_2010',
'36 BMW_M6_Convertible_2010',
'37 BMW_X3_SUV_2012',
'38 BMW_Z4_Convertible_2012',
'39 Bentley_Continental_Supersports_Conv._Convertible_2012',
'4 Acura_TL_Type-S_2008',
'40 Bentley_Arnage_Sedan_2009',
'41 Bentley_Mulsanne_Sedan_2011',
'42 Bentley_Continental_GT_Coupe_2012',
'43 Bentley_Continental_GT_Coupe_2007',
'44 Bentley_Continental_Flying_Spur_Sedan_2007',
'45 Bugatti_Veyron_16.4_Convertible_2009',
'46 Bugatti_Veyron_16.4_Coupe_2009',
'47 Buick_Regal_GS_2012',
'48 Buick_Rainier_SUV_2007',
'49 Buick_Verano_Sedan_2012',
'5 Acura_TSX_Sedan_2012',
'50 Buick_Enclave_SUV_2012',
'6 Acura_Integra_Type_R_2001',
'7 Acura_ZDX_Hatchback_2012',
'8 Aston_Martin_V8_Vantage_Convertible_2012',
'9 Aston_Martin_V8_Vantage_Coupe_2012',
)

parser = argparse.ArgumentParser(description='Evaluate a trained model.')
parser.add_argument('--model_path', type=str, default="model/model_109_0.639.pth", help='Path to the model file.')       #模型
parser.add_argument('--test_dir', type=str, default='data/test/', help='Path to the test directory.')      #测试集
parser.add_argument('--labels_csv', type=str, default='data/test/tru.csv',help='Path to the CSV file containing true labels.')   #真实数据标签
parser.add_argument('--output_path', type=str, default='output.txt', help='Path to the output file.')      #输出图像文本
args = parser.parse_args()

model_path = args.model_path
test_dir = args.test_dir
labels_csv = args.labels_csv
output_path = args.output_path

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path)
model.to(DEVICE)
model.eval()

testList = os.listdir(test_dir)
total_images = len(testList)
labels_df = pd.read_csv(labels_csv)
filename_to_label = {row['filename']: row['label'] for index, row in labels_df.iterrows()}   #filename：图像名称  label：真实数据标签

correct = 0
total_with_label = 0
total_time = 0.0
results = []

for file in testList:
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):    #图片类别
        continue

    img_path = os.path.join(test_dir, file)
    true_label = filename_to_label.get(file)

    if true_label is None:
        print(f"Warning: No label found for file {file}")
        continue

    start_time = time.time()
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform_test(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)

    probabilities = torch.softmax(logits, dim=1)
    _, pred = probabilities.topk(k=1, dim=1)
    pred_label = classes[pred.item()] if pred.item() < len(classes) else "Unknown"

    end_time = time.time()
    single_image_time = end_time - start_time
    total_time += single_image_time

    if pred_label == true_label:
        correct += 1

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 10)  # 确保字体文件存在或替换为其他字体
    except IOError:
        print("Font file not found, using default font.")
        font = ImageFont.load_default()  # 使用默认字体作为备选方案
    text = f"{pred_label}: {probabilities.max().item() * 100:.2f}%"  # 格式化文本显示标签和概率
    textwidth, textheight = draw.textsize(text, font=font)  # 获取文本大小，确保指定font参数
    # 计算文本位置（左上角，留出一些边距）
    margin = 10
    x = margin
    y = margin

    # 绘制文本（描边和填充）
    draw.text((x, y), text, fill=(255, 255, 255, 0), stroke=(0, 0, 0, 255), strokewidth=2, font=font)
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    # 显示图像
    #plt.imshow(img)
    #plt.show()
    #plt.close()  # 关闭图像以避免内存泄漏

    # 保存带标签的图像到文件
    output_img_path = os.path.join("output_images", f"predicted_{file}")  # 设置输出路径和文件名
    os.makedirs("output_images", exist_ok=True)  # 确保输出目录存在
    img.save(output_img_path)  # 保存图像

    print(f'Image Name: {file}, Predict: {pred_label}, True: {true_label}, Time: {single_image_time:.4f}s')
    total_with_label += 1
    results.append({
        'filename': file,
        'predicted_label': pred_label,
        'true_label': true_label,
        'time': single_image_time

    })#输出各项参数

accuracy = correct / total_with_label if total_with_label > 0 else 0               #输出准确率
accuracy_percent = accuracy * 100      #将准确率转化为百分数形式
average_time = total_time / total_with_label if total_with_label > 0 else 0        #输出平均每张图片的输出时间


def output_results(results, accuracy, average_time, total_with_label, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f'Accuracy: {accuracy_percent:.4f}%\n')
        f.write(f'Average recognition time per image: {average_time:.4f}s\n')
        f.write(f'Total processed images: {total_with_label}\n')
        f.write('\n')
        f.write('Filename,Predicted label,True label,Recognition time (s)\n')
        for result in results:
            f.write(f'{result["filename"]},{result["predicted_label"]},{result["true_label"]},{result["time"]:.4f}\n')


output_results(results, accuracy, average_time, total_with_label, output_path)

print(f'Accuracy: {accuracy_percent:.4f}%')
print(f'Average recognition time per image: {average_time:.4f}s')
print(f'Total processed images: {total_with_label}')