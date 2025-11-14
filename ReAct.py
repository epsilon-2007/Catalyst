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
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes in in-dataset')
    parser.add_argument('--pool', default = 'avg', type=str, help='custom operation in [avg, std, max, median, entropy]')
    parser.add_argument('--model', default='resnet18', type=str, help='model architecture: [resnet18, resnet34, densenet101]')

    parser.set_defaults(argument=True)
    device = get_device()
    parser.add_argument('--device', type=torch.device, default=device, help = 'device type for accelerated computation')

    args = parser.parse_args()

    # load checkpoints paths
    base_directory_name = f"{args.model}/{args.in_dataset}/"
    model_checkpoint_directory = os.path.join(ModelConfig.model_checkpoint_directory, base_directory_name)

    # load checkpoints
    model_checkpoint_name = f"{args.checkpoint}.pt"
    args.ckpt = os.path.join(model_checkpoint_directory, model_checkpoint_name)

    # feature stats directory
    args.feat_stats_dir = model_checkpoint_directory

    if args.in_dataset in ["CIFAR-10"]:
        args.num_classes = 10
    elif args.in_dataset in ["CIFAR-100"]:
        args.num_classes = 100
    elif args.in_dataset in ["ImageNet-1K"]:
        args.num_classes = 1000

    return args


def get_activation_log(model, train_set):
    batch_size = 1000
    device = args.device
    id_train_size = 50000
    feature_dim = model.dim_in
    activation_log = np.zeros((id_train_size, feature_dim))
    train_loader = dataset_loader(args, train_set, batch_size=batch_size)
    model.eval() # already set to eval during setting up model

    with torch.inference_mode():
        for batch_idx, (x, y) in tqdm(enumerate(train_loader)):

            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, id_train_size)

            x = x.to(device)
            y = y.to(device)

            features = model.my_features(x)
            activation_log[start_ind:end_ind, :] = features.data.cpu().numpy()

    print(f"activation log shape: {activation_log.shape}")
    return activation_log

def main(args):
    # saving activation central measure tendency
    info = f"{args.pool}_percentile_stats.txt"
    print(f"file location/name: {info}")
    feature_stats_log = redirect_stdout_to_file(args.feat_stats_dir, info)

    # setting up model
    print(f"setting up model: {args.model}")
    model = set_model(args)
    if os.path.exists(args.ckpt):
        print(f'loading existing model:{args.ckpt}')
        model.load_state_dict(torch.load(args.ckpt))
        model.eval()

    # load in-datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.244, 0.262]),
    ])
    transform_imagenet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    if args.in_dataset == 'CIFAR-10':
        train_set = datasets.CIFAR10(root=args.id_loc, train=True, download=True, transform=transform)
    elif args.in_dataset == 'CIFAR-100':
        train_set = datasets.CIFAR100(root=args.id_loc, train=True, download=True, transform=transform)
    elif args.in_dataset == 'ImageNet-1K':
        train_set = datasets.ImageFolder(root=args.id_loc, transform = transform_imagenet)

    print(f"start getting activation logs")
    activation_log = get_activation_log(model, train_set)

    print("-----------------------------------------------------------")

    print(f"std activation value:     {np.std(activation_log.flatten())}")
    print(f"mean activation value:    {np.mean(activation_log.flatten())}")
    print(f"minimum activation value: {np.min(activation_log.flatten())}")
    print(f"maximum activation value: {np.max(activation_log.flatten())}")
    
    print("-----------------------------------------------------------")

    percentiles = [5, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 96, 97, 98, 99.0]
    for percentile in percentiles:
        threshold = np.percentile(activation_log.flatten(), percentile)
        print(f"THRESHOLD at percentile {percentile} is:{threshold}")
    
    print("-----------------------------------------------------------")

    from scipy import stats
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 10.0]
    for threshold in thresholds:
        percentile = stats.percentileofscore(activation_log.flatten(), threshold)
        print(f"PERCENTILE at threshold {threshold} is: {percentile}")
    feature_stats_log.close()

if __name__ == '__main__':
    args = args_parser()
    main(args)