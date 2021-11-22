import torch
from torch.utils.data import DataLoader
from data.pascal_voc import PascalVOC
from data.augmentations import Augmentations, BaseTransform


VOC_CONFIG = {
    '0712': ([('2007', 'trainval'), ('2012', 'trainval')],
             [('2007', 'test')]),
    '0712+': ([('2007', 'trainval'), ('2012', 'trainval'), ('2007', 'test')],
              [('2012', 'test')])
}


def detection_collate(batch):
    images = []
    targets = []
    for sample in batch:
        images.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(images, 0), targets


def get_loader(config):

    dataset = None
    data_loader = None

    batch_size = config.batch_size
    new_size = config.new_size
    means = config.means

    if config.dataset == 'voc':
        voc_config = VOC_CONFIG[config.voc_config]
        if config.mode == 'train' or config.mode == 'trainval':
            image_transform = Augmentations(new_size, means)
            dataset = PascalVOC(data_path=config.voc_data_path,
                                image_sets=voc_config[0],
                                new_size=new_size,
                                mode='trainval',
                                image_transform=image_transform)

        elif config.mode == 'test':
            image_transform = BaseTransform(new_size, means)
            dataset = PascalVOC(data_path=config.voc_data_path,
                                image_sets=voc_config[1],
                                new_size=new_size,
                                mode=config.mode,
                                image_transform=image_transform)

    if dataset is not None:
        if config.mode == 'train':
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     collate_fn=detection_collate,
                                     num_workers=4,
                                     pin_memory=True)

        elif config.mode == 'test':
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     collate_fn=detection_collate,
                                     num_workers=4,
                                     pin_memory=True)

    return data_loader
