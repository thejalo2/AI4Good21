import os


class Params:

    # set to model path to continue training
    resume = 'reweight-vit-deferred_base_patch16_224_best.pth.tar'

    # paths
    if os.name == 'nt':
        data_root = 'C:/Users/janik/inat_data/'
    else:
        data_root = '/cluster/scratch/ljanik/inat_data/'
    train_file = data_root + 'train2018.json'
    val_file = data_root + 'val2018.json'
    test_file = data_root + 'test2018.json'
    cat_file = data_root + 'categories.json'
    save_path = 'vit_base_patch16_224.pth.tar'
    op_file_name = 'inat2018_test_preds.csv'  # submission file

    # hyper-parameters
    num_classes = 8142
    batch_size = 16
    lr = 1e-5
    epochs = 100
    start_epoch = 0
    start_alpha = 1.0
    reweighting = 'deferred'  # 'normal' / 'deferred'

    # system variables
    print_freq = 100
    acc_freq = 1
    if os.name == 'nt':
        workers = 0
    else:
        workers = 4
