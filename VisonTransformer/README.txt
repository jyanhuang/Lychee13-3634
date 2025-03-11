Vision Transformer
## 目录结构
　　　VIT/
　　　│
　　　├── dataset/
　　　│   ├── train
　　　│   └── val
　　　│
　　　├── runs/
　　　│   
　　　└── weights
　　　│    └──  best_model.pth（最好的模型）
　　　│   
　　　└── class_indices.json
　　　│   
　　　└── jx_vit_base_patch16_224_in21k-e5005f0a.pth(预训练模型)
　　　│   
　　　└── my_dataset.py(数据集格式) 
　　　│   
　　　└── parameters.py(计算参数量)
　　　│   
　　　└── predict.py(单张图片的种类预测)  
　　　│   
　　　└── test.py
　　　│   
　　　└── train.py
　　　│   
　　　└── utils.py 
　　　│   
　　　└── vit_model.py(模型)

## 数据集文件格式(.jpg)
　　　数据集包括了训练集train，测试集val，其中训练集、测试集包含了五十个车型种类，命名格式为品牌_车型_年份。数据集共4203张照片，其中训练集1972张，测试集2028张，训练集：测试集=1 ：1。

##读取数据
　　　train.py第166行将数据集所在根目录更改为自己的数据集
　　　
　　　
##如何训练
　　　1.更改train.py第166行数据集路径
	   2.更改train.py第160行{parser.add_argument('--num_classes', type=int, default=3)}的default
　　　3.train.py第171行，预训练模型路径，如果不想载入就设置为空字符
　　　4.运行train.py

　　　
##实现对单张图片的预测
	  1.更改predict.py第22行路径，传入单张图片
	  2..json文件

##引用
	[1] Dosovitskiy A , Beyer L , Kolesnikov A ,et al.An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale[J].  	2020.DOI:10.48550/arXiv.2010.11929.

　　　                        