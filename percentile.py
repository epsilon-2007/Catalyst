import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from configs import ModelConfig
from torchvision import datasets
import torchvision.transforms as transforms

from utils.datasets import dataset_loader
from utils.utils import set_model, get_device, redirect_stdout_to_file


def args_parser():
    parser = argparse.ArgumentParser(description='Reproducing ReAct work')

    parser.add_argument('--seed', default=7, type=int, help='random seed')
    parser.add_argument('--p', default=None, type=int, help='DICE pruning level')
    parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
    parser.add_argument('--id_loc', default="datasets/in/", type=str, help='location of in-distribution dataset')

    parser.add_argument('--epochs', default=100, type=int, help='checkpoint loading epoch')
    parser.add_argument('--checkpoint', default = 'model', type=str, help='checkpoint name')
    parser.add_argument('--dim_in', default = 512, type=int, help='penultimate feature dim')
    parser.add_argument('--pool', default = 'avg', type=str, help='custom operation in [avg, std, max, median, entropy]')
    parser.add_argument('--model', default='resnet18', type=str, help='model architecture: [resnet18, resnet34, densenet101]')

    parser.set_defaults(argument=True)
    device = get_device()
    parser.add_argument('--device', type=torch.device, default=device, help = 'device type for accelerated computation')

    args = parser.parse_args()

    return args

def setup_directory(args):
    base_directory_name = f"{args.model}/{args.in_dataset}/"
    model_statistics_directory = os.path.join(ModelConfig.model_statistics_directory, base_directory_name)
    model_percentile_directory = os.path.join(ModelConfig.model_percentile_directory, base_directory_name)

    in_features_directory = os.path.join(model_statistics_directory, f"{args.in_dataset}/")
    in_percentile_directory = os.path.join(model_percentile_directory, f"{args.in_dataset}/")
    # in_statistics_file_name = os.path.join(in_features_directory, f"in_")

    if not os.path.exists(in_percentile_directory):
            os.makedirs(in_percentile_directory)

    return in_features_directory, in_percentile_directory

def main(args):
    # saving activation central measure tendency
    in_features_directory, in_percentile_directory = setup_directory(args)
    in_statistics_file_name = os.path.join(in_features_directory, f"in_{args.pool}.npy")
    print(f"loading features: {in_statistics_file_name}")
    activation_log = np.load(in_statistics_file_name)

    in_percentile_file_name = f"in_{args.pool}_percentile_stats.txt"
    print(f"percentile file name: {in_percentile_file_name }")
    feature_stats_log = redirect_stdout_to_file(in_percentile_directory, in_percentile_file_name)


    print("----------------------------------------------------------------------------------------------------------------------")
    print(f"std activation value:     {np.std(activation_log.flatten())}")
    print(f"mean activation value:    {np.mean(activation_log.flatten())}")
    print(f"minimum activation value: {np.min(activation_log.flatten())}")
    print(f"maximum activation value: {np.max(activation_log.flatten())}")
    print("----------------------------------------------------------------------------------------------------------------------")

    for percentile in range(10, 100):
        threshold = np.percentile(activation_log.flatten(), percentile)
        print(f"THRESHOLD at percentile {percentile} is:{threshold}")
    print("----------------------------------------------------------------------------------------------------------------------")

    from scipy import stats
    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 10.0]
    for threshold in thresholds:
        percentile = stats.percentileofscore(activation_log.flatten(), threshold)
        print(f"PERCENTILE at threshold {threshold} is: {percentile}")
    feature_stats_log.close()

if __name__ == '__main__':
    args = args_parser()
    main(args)