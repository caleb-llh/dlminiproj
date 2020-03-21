from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Deep Learning Mini Project')
    
    # Mode
    parser.add_argument('--run', type=str, default='train', help="train/results")
    
    # Input directories
    parser.add_argument('--data_dir', type=str, default='/content/VOC2012/', 
                    help="root path to data directory")
    
    # Output directories
    parser.add_argument('--saved_img_dir', type=str, default='saved_img')
    parser.add_argument('--saved_model_dir', type=str, default='saved_model')
    parser.add_argument('--saved_pkl_dir', type=str, default='saved_pkl_dir')

    # Training hyperparameters configuration
    parser.add_argument('--num_epoch', default=20, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=32, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=32, type=int, 
                    help="test batch size")
    # parser.add_argument('--learn_rate', default=0.005, type=float,
    #                 help="initial learning rate")
    parser.add_argument('--weight_decay', default=0.0005, type=float,
                    help="initial learning rate")

    # Others
    parser.add_argument('--random_seed', type=int, default=999)
    parser.add_argument('--gpu', default=0, type=int, 
                    help='In homework, please always set to 0')
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    parser.add_argument('--best_learn_rate', default=0.01, type=float,
                    help="best learning rate selected")            
    
    
    args = parser.parse_args()

    return args
