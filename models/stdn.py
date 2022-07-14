import torch
import torch.nn as nn
from utils.init import xavier_init
from layers.detection import Detect
from torchvision.models import densenet169
from layers.scale_transfer_module import ScaleTransferModule


class STDN(nn.Module):

    """STDN Architecture"""

    def __init__(self,
                 mode,
                 stem_block,
                 base,
                 scale_transfer_module,
                 head,
                 anchors,
                 new_size,
                 class_count):

        super(STDN, self).__init__()

        self.mode = mode
        self.stem_block = stem_block

        self.base = base
        self.base.features.conv0 = None
        self.base.features.norm0 = None
        self.base.features.relu0 = None
        self.base.features.pool0 = None
        self.base.features.norm5 = None
        self.base.classifier = None

        self.scale_transfer_module = scale_transfer_module

        self.class_head = head[0]
        self.loc_head = head[1]

        self.stdn_in = stdn_in[str(new_size)]
        self.stdn_out = stdn_out[str(new_size)]
        self.anchors = anchors
        self.class_count = class_count

        if mode == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect.apply

    def init_weights(self,
                     model_save_path,
                     base_network):
        # self.stem_block.apply(fn=xavier_init)
        # self.class_head.apply(fn=xavier_init)
        # self.loc_head.apply(fn=xavier_init)
        self.init_weights_(self.stem_block)
        self.init_weights_(self.class_head)
        self.init_weights_(self.loc_head)

    def init_weights_(self, block):
        for module in block.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.stem_block(x)
        x = self.base.features.denseblock1(x)
        x = self.base.features.transition1(x)
        x = self.base.features.denseblock2(x)
        x = self.base.features.transition2(x)
        x = self.base.features.denseblock3(x)
        x = self.base.features.transition3(x)
        y = self.base.features.denseblock4(x)

        output = []
        for stop in self.stdn_in:
            output.append(y[:, :stop, :, :])

        y = self.scale_transfer_module(output)

        class_preds = []
        loc_preds = []

        for i in range(len(y)):
            class_pred = self.class_head[i](y[i].contiguous())
            b = class_pred.shape[0]
            class_pred = class_pred.permute(0, 2, 3, 1).contiguous()
            class_pred = class_pred.view(b, -1, self.class_count)
            class_preds.append(class_pred)

            loc_pred = self.loc_head[i](y[i].contiguous())
            b = loc_pred.shape[0]
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_pred = loc_pred.view(b, -1, 4)
            loc_preds.append(loc_pred)

        class_preds = torch.cat(class_preds, 1)
        loc_preds = torch.cat(loc_preds, 1)

        if self.mode == 'test':
            output = self.detect(self.class_count,
                                 self.softmax(class_preds),
                                 loc_preds,
                                 self.anchors)

        else:
            output = (class_preds,
                      loc_preds)

        return output


stdn_in = {
    '300': [800, 960, 1120, 1280, 1440, 1664],
    '513': [800, 960, 1120, 1280, 1440, 1600, 1664]
}

stdn_out = {
    '300': [(1, 800), (3, 960), (5, 1120), (9, 1280), (18, 360), (36, 104)],
    '513': [(1, 800), (2, 960), (4, 1120), (8, 1280), (16, 1440), (32, 400),
            (64, 104)]
}


def get_stem_block():

    layers = []

    layers += [nn.Conv2d(in_channels=3,
                         out_channels=64,
                         kernel_size=3,
                         stride=2)]
    layers += [nn.BatchNorm2d(num_features=64)]
    layers += [nn.ReLU(inplace=True)]

    layers += [nn.Conv2d(in_channels=64,
                         out_channels=64,
                         kernel_size=3,
                         stride=1)]
    layers += [nn.BatchNorm2d(num_features=64)]
    layers += [nn.ReLU(inplace=True)]

    layers += [nn.Conv2d(in_channels=64,
                         out_channels=64,
                         kernel_size=3,
                         stride=1)]
    layers += [nn.BatchNorm2d(num_features=64)]
    layers += [nn.ReLU(inplace=True)]

    layers += [nn.AvgPool2d(kernel_size=2,
                            stride=2)]

    return nn.Sequential(*layers)


def multibox(num_channels,
             num_anchors,
             class_count):

    class_head = nn.ModuleList()
    loc_head = nn.ModuleList()

    for _, channel in num_channels:
        class_head.append(get_class_subnet(channel=channel,
                                           num_anchors=num_anchors,
                                           class_count=class_count))
        loc_head.append(get_loc_subnet(channel=channel,
                                       num_anchors=num_anchors,
                                       class_count=class_count))

    return (class_head, loc_head)


def get_class_subnet(channel,
                     num_anchors,
                     class_count):
    layers = []
    mid_channels = 256

    layers += [nn.BatchNorm2d(num_features=channel)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(in_channels=channel,
                         out_channels=mid_channels,
                         kernel_size=1,
                         stride=1)]

    layers += [nn.BatchNorm2d(num_features=mid_channels)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(in_channels=mid_channels,
                         out_channels=mid_channels,
                         kernel_size=3,
                         stride=1,
                         padding=1)]

    layers += [nn.BatchNorm2d(num_features=mid_channels)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(in_channels=mid_channels,
                         out_channels=class_count * num_anchors,
                         kernel_size=3,
                         stride=1,
                         padding=1)]

    return nn.Sequential(*layers)


def get_loc_subnet(channel,
                   num_anchors,
                   class_count):
    layers = []
    mid_channels = 256

    layers += [nn.BatchNorm2d(num_features=channel)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(in_channels=channel,
                         out_channels=mid_channels,
                         kernel_size=1,
                         stride=1)]

    layers += [nn.BatchNorm2d(num_features=mid_channels)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(in_channels=mid_channels,
                         out_channels=mid_channels,
                         kernel_size=3,
                         stride=1,
                         padding=1)]

    layers += [nn.BatchNorm2d(num_features=mid_channels)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(in_channels=mid_channels,
                         out_channels=4 * num_anchors,
                         kernel_size=3,
                         stride=1,
                         padding=1)]

    return nn.Sequential(*layers)


def build_STDN(mode,
               new_size,
               anchors,
               class_count):

    stem_block = get_stem_block()
    base = densenet169(pretrained=True)
    scale_transfer_module = ScaleTransferModule(new_size=new_size)
    head = multibox(num_channels=stdn_out[str(new_size)],
                    num_anchors=8,
                    class_count=class_count)

    return STDN(stem_block=stem_block,
                mode=mode,
                base=base,
                scale_transfer_module=scale_transfer_module,
                head=head,
                anchors=anchors,
                new_size=new_size,
                class_count=class_count)
