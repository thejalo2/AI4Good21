import os


class Params:

    # hyper-parameters
    num_classes = 8142
    if os.name == 'nt':
        workers = 0
    else:
        workers = 4
    epochs = 100
    start_epoch = 0
    batch_size = 16
    lr = 1e-5
    lr_decay = 0.94
    epoch_decay = 4
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 100
    acc_freq = 1

    # paths
    resume = 'adam_wide_resnet_50_best.pth.tar'
    save_path = 'adam_wide_resnet_50_best.pth.tar'
    if os.name == 'nt':
        data_root = 'C:/Users/janik/inat_data/'
    else:
        data_root = '/cluster/scratch/ljanik/inat_data/'
    train_file = data_root + 'train2018.json'
    val_file = data_root + 'val2018.json'
    test_file = data_root + 'test2018.json'
    cat_file = data_root + 'categories.json'

    # set evaluate to True to run the test set
    evaluate = False
    save_preds = True
    op_file_name = 'inat2018_test_preds.csv'  # submission file
