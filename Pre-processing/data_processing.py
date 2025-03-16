
from PIL import Image

global mapcontx, mapconty, imgmap

# print(imgmap)

def bgTrans2white(img):
    L_x = []
    L_y = []
    k = 170  # 像素亮度变化率
    # imgmap=[0]*2000         #2-前景，1分界，0背景,

    imgmap = [[2] * 3024 for i in range(0, 3024)]

    imgmap[0][0] = 0

    mapcontx = 1000
    mapconty = 1000


    sp = img.size
    width = sp[0]
    height = sp[1]

    # print(sp)

    for i in range(3023):
        for j in range(3023):
            dot1 = (i, j)
            color_d1 = img.getpixel(dot1)  # 与cv2不同的是，这里需要用getpixel方法来获取维度数据
            dot2 = (i, j + 1)
            color_d2 = img.getpixel(dot2)  # 与cv2不同的是，这里需要用getpixel方法来获取维度数据
            dot3 = (i + 1, j)
            color_d3 = img.getpixel(dot3)  # 与cv2不同的是，这里需要用getpixel方法来获取维度数据
            dot4 = (i + 1, j + 1)
            color_d4 = img.getpixel(dot4)  # 与cv2不同的是，这里需要用getpixel方法来获取维度数据
            # print(abs(color_d2[2]-color_d1[2]))

            if (abs(color_d2[2] - color_d1[2]) < k and imgmap[i][j + 1] == 2 and imgmap[i][j] == 0):
                imgmap[i][j + 1] = 0
                img.putpixel(dot2, (255, 255, 255))  # 赋值的方法是通过putpixel
            if (abs(color_d3[2] - color_d1[2]) < k and imgmap[i + 1][j] == 2 and imgmap[i][j] == 0):
                imgmap[i + 1][j] = 0
                img.putpixel(dot3, (255, 255, 255))  # 赋值的方法是通过putpixel
            if (abs(color_d4[2] - color_d1[2]) < k and imgmap[i + 1][j + 1] == 2 and imgmap[i][j] == 0):
                imgmap[i + 1][j + 1] = 0
                img.putpixel(dot4, (255, 255, 255))  # 赋值的方法是通过putpixel

            # print(imgmap[i][j],imgmap[i][j+1],imgmap[i+1][j],abs(color_d2[2]-color_d1[2]))

    for m in range(3023):
        i = 3023 - m
        for n in range(3023):
            j = 3023 - n
            dot1 = (i, j)
            color_d1 = img.getpixel(dot1)  # 与cv2不同的是，这里需要用getpixel方法来获取维度数据
            dot2 = (i, j - 1)
            color_d2 = img.getpixel(dot2)  # 与cv2不同的是，这里需要用getpixel方法来获取维度数据
            dot3 = (i - 1, j)
            color_d3 = img.getpixel(dot3)  # 与cv2不同的是，这里需要用getpixel方法来获取维度数据
            dot4 = (i - 1, j - 1)
            color_d4 = img.getpixel(dot4)  # 与cv2不同的是，这里需要用getpixel方法来获取维度数据
            # print(abs(color_d2[2]-color_d1[2]))

            if (abs(color_d2[2] - color_d1[2]) < k and imgmap[i][j - 1] == 2 and imgmap[i][j] == 0):
                imgmap[i][j - 1] = 0
                img.putpixel(dot2, (255, 255, 255))  # 赋值的方法是通过putpixel
            if (abs(color_d3[2] - color_d1[2]) < k and imgmap[i - 1][j] == 2 and imgmap[i][j] == 0):
                imgmap[i - 1][j] = 0
                img.putpixel(dot3, (255, 255, 255))  # 赋值的方法是通过putpixel
            if (abs(color_d4[2] - color_d1[2]) < k and imgmap[i - 1][j - 1] == 2 and imgmap[i][j] == 0):
                imgmap[i - 1][j - 1] = 0
                img.putpixel(dot4, (255, 255, 255))  # 赋值的方法是通过putpixel

            # print(imgmap[i][j],imgmap[i][j+1],imgmap[i+1][j],abs(color_d2[2]-color_d1[2]))

    return img

import os

img_dir = 'F:/zhenghanling/Resnet/data/train/baila/'
out_dir = './Output/train/baila/'
# img = Image.open('F:/zhenghanling/Resnet/data/train/baila/IMG_E5202.jpg')
for filename in os.listdir(img_dir):
    print(filename)
    img = Image.open(img_dir+filename)
    img1 = bgTrans2white(img)
    # img1.show()  # 显示图片
    img1.save(out_dir+filename)  # 保存图片
# print(imgmap)
print("处理完成！")


# # 设置输入和输出文件夹路径
# input_folder = 'F:/zhenghanling/Resnet/data/train/baila/'  # 替换为你的输入文件夹路径
# output_folder = 'F:/zhenghanling/data/train/baitangyin/'  # 替换为你的输出文件夹路径
#
# for filename in os.listdir(input_folder):
#     # input_path = os.path.join(input_folder, filename)
#     # output_path = os.path.join(output_folder, filename)
#     img = Image.open(input_folder+filename)
#     img = bgTrans2white(img)
#     # img.show()  # 显示图片
#     img.save('F:/zhenghanling/data/train/baila/23.jpg')  # 保存图片
#     print(filename+"finished!")
# print("处理完成！")



