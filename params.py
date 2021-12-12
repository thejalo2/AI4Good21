import os


class Params:
    # set to model path to continue training
    resume = 'exit_dynamic_10_5'

    # paths
    if os.name == 'nt':
        data_root = 'C:/Users/janik/inat_data/'
    else:
        data_root = '/cluster/scratch/ljanik/inat_data/'
    train_file = data_root + 'train2018.json'
    val_file = data_root + 'val2018.json'
    test_file = data_root + 'test2018.json'
    cat_file = data_root + 'categories.json'
    save_path = 'exit_dynamic_10_5.pth.tar'
    op_file_name = 'inat2018_test_preds.csv'  # submission file

    # hyper-parameters
    num_classes = 8142
    if os.name == 'nt':
        batch_size = 16
    else:
        batch_size = 16
    lr = 1e-5
    epochs = 50
    start_epoch = 0
    start_alpha = 1.0
    inference_alpha = 0.0
    share_embedder = True
    use_ldam = False
    reweighting = True
    resampling = False
    combine_logits = False
    merged_training = False
    beta = None

    # rank performance based down-weighting
    exit_strategy = 'downweight_dynamic'  # downweight_fixed / downweight_dynamic / downweight_min / dropout
    dw_factor = 0.1
    chunk_size = 815
    # chunk_size = 221
    weight_both_branches = False
    if exit_strategy == 'downweight_fixed':
        exit_thresh = 80
    elif exit_strategy == 'downweight_dynamic':
        exit_thresh = 10
        thresh_increase = 5
        max_thresh = 90
    elif exit_strategy == 'downweight_min':
        exit_thresh = 0
        max_thresh = 90
        nearest = 10

    # system variables
    print_freq = 100
    acc_freq = 1
    if os.name == 'nt':
        workers = 0
    else:
        workers = 4