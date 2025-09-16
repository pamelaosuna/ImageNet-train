import torch.nn as nn
from torchvision.models import (
    alexnet,
    resnet50
)
# vit
# from torchvision.models import vit_b_16


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# AlexNet
def get_model(model_name, device):
    if model_name == "alexnet":
        model = alexnet(num_classes=1000)
    elif model_name == "resnet50":
        model = resnet50(num_classes=1000)
    # elif model_name == "vit_b_16":
    #     model = vit_b_16(num_classes=1000)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    model.apply(init_weights)  # Custom init
    model = model.to(device)

    return model