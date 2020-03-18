import os
from utils.genutils import write_print
from datetime import datetime
import zipfile
import argparse
import torch
import numpy as np


SAVE_NAME_FORMAT = 'files_{}.{}'


def zip_directory(path, zip_file):
    """Stores all py and cfg project files inside a zip file

    [description]

    Arguments:
        path {string} -- current path
        zip_file {zipfile.ZipFile} -- zip file to contain the project files
    """
    files = os.listdir(path)
    for file in files:
        if file.endswith('.py') or file.endswith('cfg'):
            zip_file.write(os.path.join(path, file))
            if file.endswith('cfg'):
                os.remove(file)


def save_config(config):
    """saves the configuration of the experiment

    [description]

    Arguments:
        config {dict} -- contains argument and its value

    Returns:
        string -- version based on the current time
    """
    version = str(datetime.now()).replace(':', '_')
    cfg_name = SAVE_NAME_FORMAT.format(version, 'cfg')
    with open(cfg_name, 'w') as f:
        for k, v in config.items():
            f.write('{}: {}\n'.format(str(k), str(v)))

    zip_name = SAVE_NAME_FORMAT.format(version, 'zip')
    zip_file = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    zip_directory('.', zip_file)
    zip_file.close()

    return version


def string_to_boolean(v):
    """Converts string to boolean

    [description]

    Arguments:
        v {string} -- string representation of a boolean values;
        must be true or false

    Returns:
        boolean -- boolean true or false
    """
    return v.lower() in ('true')


if __name__ == '__main__':
    torch.set_printoptions(threshold=np.nan)
    parser = argparse.ArgumentParser()

    # dataset info
    parser.add_argument('--input_channels', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--class_count', type=int, default=21,
                        help='Number of classes in dataset')
    parser.add_argument('--dataset', type=str, default='voc',
                        choices=['voc', 'coco'],
                        help='Dataset to use')
    parser.add_argument('--new_size', type=int, default=300,
                        help='New height and width of input images')
    parser.add_argument('--means', type=tuple, default=(104, 117, 123),
                        help='Mean values of the dataset')
    parser.add_argument('--anchor_config', type=str, default='SFDet-300',
                        choices=['SFDet-300', 'SFDet-512'],
                        help='Anchor box configuration to use')

    # training settings
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('--batch_multiplier', type=int, default=1,
                        help='Batch multiplier')
    parser.add_argument('--basenet', type=str, default='vgg16_reducedfc.pth',
                        help='Base network for VGG')
    parser.add_argument('--pretrained_model', type=str,
                        default=None,
                        help='Pre-trained model')

    # architecture settings
    parser.add_argument('--model', type=str, default='VGG-SFDet',
                        choices=['VGG-SFDet', 'ResNet-SFDet'],
                        help='Model to use')
    parser.add_argument('--resnet_model', type=str, default='50',
                        choices=['18', '34', '50', '101'],
                        help='ResNet base network configuration')

    # step size
    parser.add_argument('--counter', type=str, default='iter',
                        choices=['iter', 'epoch'],
                        help='Type of counter to use in training')
    parser.add_argument('--num_iterations', type=int, default=120000,
                        help='Number of iterations')
    parser.add_argument('--num_epochs', type=int, default=250,
                        help='Number of epochs')
    parser.add_argument('--loss_log_step', type=int, default=100,
                        help='Number of steps for logging loss')
    parser.add_argument('--model_save_step', type=int, default=4000,
                        help='Number of step for saving model')

    # scheduler settings
    parser.add_argument('--warmup', type=string_to_boolean, default=False,
                        help='Toggles the use of warm-up training')
    parser.add_argument('--warmup_step', type=int, default=6,
                        help='Warm-up steps')
    parser.add_argument('--sched_milestones', type=list,
                        default=[80000, 100000, 120000],
                        help='Number of steps before adjusting learning rate')
    parser.add_argument('--sched_gamma', type=float, default=0.1,
                        help='Controls adjustment made to the learning rate')

    # loss settings
    parser.add_argument('--loss_config', type=str, default='multibox',
                        choices=['multibox'],
                        help='Type of loss')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='IoU threshold')
    parser.add_argument('--pos_neg_ratio', type=int, default=3,
                        help='Ratio for hard negative mining')

    # misc
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='Mode of execution')
    parser.add_argument('--use_gpu', type=string_to_boolean, default=True,
                        help='Toggles the use of GPU')

    # pascal voc dataset
    parser.add_argument('--voc_config', type=str, default='0712',
                        choices=['0712', '0712+'],
                        help='Pascal VOC dataset configuration')
    parser.add_argument('--voc_data_path', type=str,
                        default='../../Datasets/PascalVOC/',
                        help='Pascal VOC dataset path')

    # coco dataset
    parser.add_argument('--coco_config', type=str, default='2014',
                        choices=['2014', '2017'],
                        help='COCO dataset configuration')
    parser.add_argument('--coco_data_path', type=str,
                        default='../../data/Coco/',
                        help='COCO dataset path')

    # path
    parser.add_argument('--model_save_path', type=str, default='./weights',
                        help='Path for saving weights')
    parser.add_argument('--result_save_path', type=str, default='./results',
                        help='Path for saving results')

    config = parser.parse_args()

    args = vars(config)
    print(args)
    write_print('hello.txt', '------------ Options -------------')
    for k, v in args.items():
        write_print('hello.txt', '{}: {}'.format(str(k), str(v)))
    write_print('hello.txt', '-------------- End ----------------')

