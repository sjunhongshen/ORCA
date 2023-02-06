import numpy as np
from torch.utils.data import Dataset
import os

field_type = {'10595_2': 'GC',
              '10595_7': 'GC',
              '9442_1': 'GC',
              '9442_3': 'GC',
              '9442_5': 'GC',
              '10760_2': 'GAL',
              '10760_4': 'GAL',
              '10631_3': 'EX',
              '10631_1': 'EX',
              '10631_4': 'EX',
              '12103_a3': 'EX',
              '13498_b1': 'EX',
              '13737_2': 'GAL',
              '13737_3': 'GAL',
              '9490_a3': 'GAL',
              '10349_30': 'GC',
              '10005_10': 'GC',
              '10120_3': 'GC',
              '12513_2': 'GAL',
              '12513_3': 'GAL',
              '14164_9': 'EX',
              '13718_6': 'EX',
              '10524_7': 'GC',
              '10182_pb': 'GAL',
              '10182_pd': 'GAL',
              '9425_2': 'EX',
              '9425_4': 'EX',
              '9583_99': 'EX',
              '10584_13': 'GAL',
              '9978_5e': 'EX',
              '15100_2': 'EX',
              '15647_13': 'EX',
              '11340_11': 'GC',
              '13389_10': 'EX',
              '9694_6': 'EX',
              '10342_3': 'GAL',

              '14343_1': 'GAL',
              '10536_13': 'EX',
              '13057_1': 'GAL',
              '10260_7': 'GAL',
              '10260_5': 'GAL',
              '10407_3': 'GAL',
              '13375_4': 'EX',
              '13375_7': 'EX',
              '13364_95': 'GAL',
              '10190_28': 'GAL',
              '10190_13': 'GAL',
              '10146_4': 'GC',
              '10146_3': 'GC',
              '10775_ab': 'GC',
              '11586_5': 'GC',
              '12438_1': 'EX',
              '13671_35': 'EX',
              '14164_1': 'GC',

              '9490_a2': 'GAL',
              '9405_6d': 'EX',
              '9405_4b': 'EX',
              '9450_14': 'EX',
              '10092_1': 'EX',
              '13691_11': 'GAL',
              '12058_12': 'GAL',
              '12058_16': 'GAL',
              '12058_1': 'GAL',
              '9450_16': 'EX',
              '10775_52': 'GC',
              '12602_1': 'GC',
              '12602_2': 'GC',
              '10775_29': 'GC',
              '10775_ad': 'GC',
              '12058_6': 'GAL',  # NEW
              '14704_1': 'GAL',  # NEW
              '13804_6': 'GAL'  # NEW
              }


base_dir = './cosmic'
data_base = './cosmic'
if True:
    train_dirs = []
    test_dirs = []

    test_base = os.path.join(data_base,'npy_test')
    train_base = os.path.join(data_base,'npy_train')

    print('------------------------------------------------------------')
    print('Fetching directories for the test set')
    print('------------------------------------------------------------')
    for _filter in os.listdir(test_base):
        filter_dir = os.path.join(test_base,_filter)
        if os.path.isdir(filter_dir) and _filter == 'f435w':
            for prop_id in os.listdir(filter_dir):
                prop_id_dir = os.path.join(filter_dir,prop_id)
                if os.path.isdir(prop_id_dir):
                    for vis_num in os.listdir(prop_id_dir):
                        vis_num_dir = os.path.join(prop_id_dir,vis_num)
                        if os.path.isdir(vis_num_dir):
                            for f in os.listdir(vis_num_dir):
                                if '.npy' in f and f != 'sky.npy':
                                    key = f'{prop_id}_{vis_num}'
                                    if field_type[key] == 'GAL':
                                        test_dirs.append(os.path.join(vis_num_dir,f))

    print('------------------------------------------------------------')
    print('Fetching directories for the training set')
    print('------------------------------------------------------------')
    for _filter in os.listdir(train_base):
        filter_dir = os.path.join(train_base,_filter)
        if os.path.isdir(filter_dir) and _filter == 'f435w':
            for prop_id in os.listdir(filter_dir):
                prop_id_dir = os.path.join(filter_dir,prop_id)
                if os.path.isdir(prop_id_dir):
                    for vis_num in os.listdir(prop_id_dir):
                        vis_num_dir = os.path.join(prop_id_dir,vis_num)
                        if os.path.isdir(vis_num_dir):
                            for f in os.listdir(vis_num_dir):
                                if '.npy' in f and f != 'sky.npy':
                                    key = f'{prop_id}_{vis_num}'
                                    if field_type[key] == 'GAL':
                                        train_dirs.append(os.path.join(vis_num_dir,f))


#     print(train_dirs)
    np.save(os.path.join(base_dir,'test_dirs.npy'), test_dirs)
    np.save(os.path.join(base_dir,'train_dirs.npy'), train_dirs)


