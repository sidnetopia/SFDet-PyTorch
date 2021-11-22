import os

weights = '2021-11-21 11_52_31.214516_train'
model = 'SFDet-ResNet'
mode = 'test'
batch = 32
use_gpu = 'True'

start = 40
save_step = 4
num_epochs = 220

for i in range(start + save_step, num_epochs + save_step, save_step):
    pretrained_model = '"{}/{}"'.format(weights, i)
    args = ('--mode {} --pretrained_model {} --model {} --use_gpu {} '
            '--batch_size {}')
    args = args.format(mode, pretrained_model, model, use_gpu, batch)
    command = 'python main.py {}'.format(args)
    os.system(command)
