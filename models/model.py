from models.sfdet_vgg import build_SFDetVGG
from models.sfdet_resnet import build_SFDetResNet
from models.sfdet_densenet import build_SFDetDenseNet


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

    elif config['model'] == 'SFDet-DenseNet':
        model = build_SFDetDenseNet(mode=config['mode'],
                                    new_size=config['new_size'],
                                    densenet_model=config['densenet_model'],
                                    anchors=anchors,
                                    class_count=config['class_count'])

    return model
