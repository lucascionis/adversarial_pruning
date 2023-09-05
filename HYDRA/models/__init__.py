"""
# small-caps refers to cifar-style models i.e., resnet18 -> for cifar vs ResNet18 -> standard arch.
from HYDRA.models import (
    vgg16,
    vgg16_bn,
)
from basic import (
    lin_1,
    lin_2,
    lin_3,
    lin_4,
    mnist_model,
    mnist_model_large,
    cifar_model,
    cifar_model_large,
    cifar_model_resnet,
    vgg4_without_maxpool,
)

from HYDRA.models import ResNet18, ResNet34, ResNet50

__all__ = [
    "vgg2",
    "vgg2_bn",
    "vgg4",
    "vgg4_bn",
    "vgg6",
    "vgg6_bn",
    "vgg8",
    "vgg8_bn",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "wrn_28_10",
    "wrn_28_1",
    "wrn_28_4",
    "wrn_34_10",
    "wrn_40_2",
    "lin_1",
    "lin_2",
    "lin_3",
    "lin_4",
    "mnist_model",
    "mnist_model_large",
    "cifar_model",
    "cifar_model_large",
    "cifar_model_resnet",
    "vgg4_without_maxpool",
    "ResNet18",
    "ResNet34",
    "ResNet50",
]
"""