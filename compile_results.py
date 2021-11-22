import os.path as osp

weights = '2021-11-21 11_52_31.214516_train'
mode = 'test'
start = 40
save_step = 4
num_epochs = 220

for i in range(start + save_step, num_epochs + save_step, save_step):
    text_file = '{}_{}_{}.txt'.format(weights, mode, i)
    file_path = osp.join('tests', weights, text_file)
    with open(file_path, 'r') as f:
        last_line = f.readlines()[-4]
        print(i, last_line)
