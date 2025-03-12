import torch
import torch.nn as nn
import torchvision.models as models


class ModifiedSqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedSqueezeNet, self).__init__()
        # Load SqueezeNet pre-trained model
        self.squeezenet = models.squeezenet1_1(pretrained=False)

        # Modify the classifier to adapt to num_classes
        self.squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        self.squeezenet.classifier[3] = nn.LogSoftmax(dim=1)  # Use LogSoftmax for classification

    def forward(self, x):
        return self.squeezenet(x)
