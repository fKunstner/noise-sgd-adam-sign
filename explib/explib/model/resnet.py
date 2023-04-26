import torchvision.models as models


def getResNet(size, pretrained=False):
    if size == 50:
        return models.resnet50(pretrained=pretrained)
    elif size == 34:
        return models.resnet34(pretrained=pretrained)
    elif size == 101:
        return models.resnet101(pretrained=pretrained)
    elif size == 18:
        return models.resnet18(pretrained=pretrained)
