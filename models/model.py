from models.sfdet_vgg import build_SFDetVGG
from models.sfdet_resnet import build_SFDetResNet


def get_model(config, anchors):
    """
    returns the model
    """

    model = None

    if config['model'] == 'SFDet-VGG':
        model = build_SFDetVGG(mode=config['mode'],
                               new_size=config['new_size'],
                               anchors=anchors,
                               class_count=config['class_count'])

    elif config['model'] == 'SFDet-ResNet':
        model = build_SFDetResNet(mode=config['mode'],
                                  new_size=config['new_size'],
                                  resnet_model=config['resnet_model'],
                                  anchors=anchors,
                                  class_count=config['class_count'])

    return model
