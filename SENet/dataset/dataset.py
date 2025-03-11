# coding:utf8
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from sklearn.model_selection import train_test_split

Labels = {'1 AM_General_Hummer_SUV_2000':   0,
          '10 Aston_Martin_Virage_Convertible_2012':  1,
          '11 Aston_Martin_Virage_Coupe_2012': 2,
          '12 Audi_RS_4_Convertible_2008': 3,
          '13 Audi_A5_Coupe_2012': 4,
          '14 Audi_TTS_Coupe_2012': 5,
          '15 Audi_R8_Coupe_2012': 6,
          '16 Audi_V8_Sedan_1994': 7,
          '17 Audi_100_Sedan_1994': 8,
          '18 Audi_100_Wagon_1994': 9,
          '19 Audi_TT_Hatchback_2011': 10,
          '2 Acura_RL_Sedan_2012': 11,
          '20 Audi_S6_Sedan_2011': 12,
          '21 Audi_S5_Convertible_2012': 13,
          '22 Audi_S5_Coupe_2012': 14,
          '23 Audi_S4_Sedan_2012': 15,
          '24 Audi_S4_Sedan_2007': 16,
          '25 Audi_TT_RS_Coupe_2012': 17,
          '26 BMW_ActiveHybrid_5_Sedan_2012': 18,
          '27 BMW_1_Series_Convertible_2012': 19,
          '28 BMW_1_Series_Coupe_2012': 20,
          '29 BMW_3_Series_Sedan_2012': 21,
          '3 Acura_TL_Sedan_2012': 22,
          '30 BMW_3_Series_Wagon_2012': 23,
          '31 BMW_6_Series_Convertible_2007': 24,
          '32 BMW_X5_SUV_2007': 25,
          '33 BMW_X6_SUV_2012': 26,
          '34 BMW_M3_Coupe_2012': 27,
          '35 BMW_M5_Sedan_2010': 28,
          '36 BMW_M6_Convertible_2010':  29,
          '37 BMW_X3_SUV_2012': 30,
          '38 BMW_Z4_Convertible_2012': 31,
          '39 Bentley_Continental_Supersports_Conv._Convertible_2012': 32,
          '4 Acura_TL_Type-S_2008': 33,
          '40 Bentley_Arnage_Sedan_2009': 34,
          '41 Bentley_Mulsanne_Sedan_2011': 35,
          '42 Bentley_Continental_GT_Coupe_2012': 36,
          '43 Bentley_Continental_GT_Coupe_2007': 37,
          '44 Bentley_Continental_Flying_Spur_Sedan_2007': 38,
          '45 Bugatti_Veyron_16.4_Convertible_2009': 39,
          '46 Bugatti_Veyron_16.4_Coupe_2009': 40,
          '47 Buick_Regal_GS_2012': 41,
          '48 Buick_Rainier_SUV_2007': 42,
          '49 Buick_Verano_Sedan_2012': 43,
          '5 Acura_TSX_Sedan_2012': 44,
          '50 Buick_Enclave_SUV_2012': 45,
          '6 Acura_Integra_Type_R_2001': 46,
          '7 Acura_ZDX_Hatchback_2012': 47,
          '8 Aston_Martin_V8_Vantage_Convertible_2012': 48,
          '9 Aston_Martin_V8_Vantage_Coupe_2012': 49,
}


class SeedlingData(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        self.transforms = transforms

        if self.test:
            imgs = [os.path.join(root, img) for img in os.listdir(root)]
            self.imgs = imgs
        else:
            imgs_labels = [os.path.join(root, img) for img in os.listdir(root)]
            imgs = []
            for imglable in imgs_labels:
                for imgname in os.listdir(imglable):
                    imgpath = os.path.join(imglable, imgname)
                    imgs.append(imgpath)
            trainval_files, val_files = train_test_split(imgs, test_size=0.3, random_state=42)
            if train:
                self.imgs = trainval_files
            else:
                self.imgs = val_files

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        img_path = img_path.replace("\\", '/')
        if self.test:
            label = -1
        else:
            labelname = img_path.split('/')[-2]
            label = Labels[labelname]
        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)