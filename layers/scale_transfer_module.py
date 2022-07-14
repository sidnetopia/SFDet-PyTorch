import torch.nn as nn


class ScaleTransferModule(nn.Module):

    def __init__(self,
                 new_size):
        super(ScaleTransferModule, self).__init__()

        self.new_size = new_size
        self.module_list = self.get_modules()

    def get_modules(self):

        layers = []

        if self.new_size == 300:
            layers += [nn.AvgPool2d(kernel_size=9,
                                    stride=9)]
            layers += [nn.AvgPool2d(kernel_size=3,
                                    stride=3)]
            layers += [nn.AvgPool2d(kernel_size=2,
                                    stride=2,
                                    padding=1)]
            layers += [nn.PixelShuffle(upscale_factor=2)]
            layers += [nn.PixelShuffle(upscale_factor=4)]

        elif self.new_size == 513:
            layers += [nn.AvgPool2d(kernel_size=16,
                                    stride=16)]
            layers += [nn.AvgPool2d(kernel_size=8,
                                    stride=8)]
            layers += [nn.AvgPool2d(kernel_size=4,
                                    stride=4)]
            layers += [nn.AvgPool2d(kernel_size=2,
                                    stride=2)]
            layers += [nn.PixelShuffle(upscale_factor=2)]
            layers += [nn.PixelShuffle(upscale_factor=4)]

        return nn.ModuleList(layers)

    def forward(self, x):

        y = []
        if self.new_size == 300:
            val = 3
        elif self.new_size == 513:
            val = 4

        for i in range(len(x)):

            # Average pooling layers
            if i < val:
                y.append(self.module_list[i](x[i]))

            # Pixel Shuffle layers
            elif i > val:
                y.append(self.module_list[i - 1](x[i]))

            # Identity layer
            elif i == val:
                y.append(x[i])

        return y
