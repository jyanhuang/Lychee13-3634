## 代码使用简介
EfficinetNetV2
|--__pycache
|--car_ims
   |--test
   |--train__
|--runs
|--torch_efficientv2
|  |--pre_efficientnetv2-l.pth
|  |--pre_efficientnetv2-m.pth
|  |--pre_efficientnetv2-s.pth
|--weights
|--class_indices.json
|--classify.py
|--model
|--my_dataset.py
|--predict.py
|-- README.md
|--requirement
|--train.py
|--trans_effv2_weights.py
|--utils.py

权重下载地址： https://pan.baidu.com/s/1uZX36rvrfEss-JGj4yfzbQ 密码： 5gu1
1.数据集分为train和test，两个子文件夹类别数皆为50类，其中'train'有2011张图像，'test'2028张图象，运行'train.py'使用train文件训练，运行'classify.py'使用test文件测试
2.在`train.py`中将`--data-path`设置数据集路径
3.在`train.py`中将`--weights`参数设成预训练权重路径（下载好的权重放在torch_efficientv2文件中）
4.设置好数据集的路径`--data-path`以及预训练权重的路径`--weights`就能使用`train.py`脚本开始训练了(训练过程中会自动生成`class_indices.json`文件)（已有）
5.训练不同模型（s,m,l）时需使以下参数（efficientnetv2_s，"s"，pre_efficientnetv2-s.pth，）一致（'train.py'中）
#from model import efficientnetv2_s as create_model ('classify.py'同)
#num_model = "s" ('classify.py'同)
#parser.add_argument('--weights', type=str, default='./torch_efficientnetv2/pre_efficientnetv2-s.pth',
                        help='initial weights path')
  可通过"./weights/s.model-{}.pth"更改训练好的模型权重名字，注意模型对应
6.训练完成后会生成多个模型权重，可选择准确率最高的权重进行测试
7. 在`classify.py`脚本中导入和训练脚本中同样的模型，并将`weights_path `设置成训练好的模型权重路径(默认保存在weights文件夹下)
8.运行classify.py'中将'imgs_root'设置数据集路径 
9.如果要使用自己的数据集，请按照原分类数据集的文件结构进行摆放(即一个类别对应一个文件夹)，并且将训练以及预测脚本中的`num_classes`设置成你自己数据的类别数
10.requirement仅供参考
