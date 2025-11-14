import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from configs import ModelConfig
from utils.utils import set_model, get_device
from utils.datasets import load_in_dataset, load_out_dataset, dataset_loader

def args_parser():
    parser = argparse.ArgumentParser(description='Analyze Distribution',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--p', default=None, type=int, help='DICE pruning level')
    parser.add_argument('--dim_in', default = 512, type=int, help='penultimate feature dim')
    parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
    parser.add_argument('--ood_loc', default="datasets/ood/", type=str, help='location of ood datasets')
    parser.add_argument('--out-dataset', default="iSUN-dummy", type=str, help='out-distribution dataset')
    parser.add_argument('--id_loc', default="datasets/in/", type=str, help='location of in-distribution dataset')
    
    parser.add_argument('--batch-size', default= 256, type=int, help='mini-batch size')
    parser.add_argument('--checkpoint', default = 'model', type=str, help='checkpoint name')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes in in-dataset')
    parser.add_argument('--model', default='densenet101', type=str, help='model architecture: [ resnet18, resnet34, densenet101]')

    device = get_device()
    parser.add_argument('--device', type=torch.device, default=device, help = 'device type for accelerated training')

    args = parser.parse_args()

    if args.in_dataset in ["CIFAR-10"]:
        args.num_classes = 10
    elif args.in_dataset in ["CIFAR-100"]:
        args.num_classes = 100
    elif args.in_dataset in ["ImageNet-1K"]:
        args.num_classes = 1000
    return args

def setup_directory(args):

    # base directory
    base_directory_name = f"{args.model}/{args.in_dataset}/"
    model_checkpoint_directory = os.path.join(ModelConfig.model_checkpoint_directory, base_directory_name)
    model_statistics_directory = os.path.join(ModelConfig.model_statistics_directory, base_directory_name)

    # load checkpoints
    model_checkpoint_name = f"{args.checkpoint}.pt"
    model_checkpoint_directory = os.path.join(ModelConfig.model_checkpoint_directory, base_directory_name)
    args.ckpt = os.path.join(model_checkpoint_directory, model_checkpoint_name)
    
    if not os.path.exists(model_statistics_directory):
        os.makedirs(model_statistics_directory)
    return model_statistics_directory


def save_weights(args, model, weight_directory):
    W_fname = weight_directory + 'W.npy'
    b_fname = weight_directory + 'b.npy'

    if args.model in ['resnet18', 'resnet34', 'densenet101', 'mobilenetv2', 'resnet18_silu']: 
        W = model.output_layer.weight.detach().cpu().numpy()
        b = model.output_layer.bias.detach().cpu().numpy()
    elif args.model in ['resnet34_imagenet', 'resnet50_imagenet']:
        W = model.fc.weight.detach().cpu().numpy()
        b = model.fc.bias.detach().cpu().numpy()
    elif args.model in ['mobilenetv2_imagenet']:
        W = model.classifier[1].weight.detach().cpu().numpy()
        b = model.classifier[1].bias.detach().cpu().numpy()
    elif args.model in ['densenet121_imagenet']:
        W = model.classifier.weight.detach().cpu().numpy()
        b = model.classifier.bias.detach().cpu().numpy()
    elif args.model in ['efficientnetb0_imagenet']:
        W = model.classifier[1].weight.detach().cpu().numpy()
        b = model.classifier[1].bias.detach().cpu().numpy()
    elif args.model in ['swinB_imagenet', 'swinT_imagenet']:
        W = model.head.weight.detach().cpu().numpy()
        b = model.head.bias.detach().cpu().numpy()

    np.save(W_fname, W)
    np.save(b_fname, b)

def save_in_features(feature_directory, labels, avg, std, maxi, median, entropy):
    # save featuures
    features = [labels, avg, std, maxi, median, entropy]
    feature_extension = ['labels.npy','avg.npy', 'std.npy', 'max.npy', 'median.npy', 'entropy.npy']
    for feature, extension in zip(features,feature_extension):
        f_name = feature_directory + extension
        np.save(f_name, feature)

def save_out_features(feature_directory, avg, std, maxi, median, entropy):
    # save featuures
    features = [avg, std, maxi, median, entropy]
    feature_extension = ['avg.npy', 'std.npy', 'max.npy', 'median.npy', 'entropy.npy']
    for feature, extension in zip(features,feature_extension):
        f_name = feature_directory + extension
        np.save(f_name, feature)

def obtain_in_statistics(args, model, data_set, batch_size = None):
    if batch_size is None:
        batch_size = args.batch_size
    device = args.device
    data_size = len(data_set)
    feature_dim = model.dim_in

    avgpool = nn.AdaptiveAvgPool2d((1,1))
    maxpool = nn.AdaptiveMaxPool2d((1,1))

    labels = np.zeros(data_size)
    avg = np.zeros((data_size, feature_dim))
    std = np.zeros((data_size, feature_dim))
    maxi = np.zeros((data_size, feature_dim))
    median = np.zeros((data_size, feature_dim))
    entropy = np.zeros((data_size, feature_dim))

    data_loader = dataset_loader(args, data_set, batch_size=batch_size)
    model.eval() # already set to eval during setting up model

    with torch.inference_mode():
        for batch_idx, (x, y) in tqdm(enumerate(data_loader)):

            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, data_size)

            x = x.to(device)
            y = y.to(device)
            labels[start_ind:end_ind]  = y.data.cpu().numpy()    

            # +++++++++++++++++++++++ start: only for ViT ++++++++++++++++++++++++++
            # batch_avg = model.features(x)
            # batch_avg = batch_avg.view(-1, feature_dim)
            # labels[start_ind:end_ind]  = y.data.cpu().numpy()
            # avg[start_ind:end_ind, :]  = batch_avg.data.cpu().numpy()
            # +++++++++++++++++++++++ end: only for ViT ++++++++++++++++++++++++++++


            # get statistics: avg, std, max
            activation_map = model.my_encoder(x)              #[batch_size, feature_dim, height, width]
            batch_avg = avgpool(activation_map)               #[batch_size, feature_dim, 1, 1]
            batch_avg = batch_avg.view(-1, feature_dim)       #[batch_size, feature_dim]
            batch_std = activation_map.std(dim = (2, 3))      #[batch_size, feature_dim]
            batch_max = maxpool(activation_map)               #[batch_size, feature_dim, 1, 1]
            batch_max = batch_max.view(-1, feature_dim)       #[batch_size, feature_dim]
            
            avg[start_ind:end_ind, :]  = batch_avg.data.cpu().numpy()
            std[start_ind:end_ind, :]  = batch_std.data.cpu().numpy()
            maxi[start_ind:end_ind, :] = batch_max.data.cpu().numpy()

            # get statistics: median, entropy
            b, d, w, h = activation_map.shape
            activation_map_flat = activation_map.view(b, d, -1)          #[batch_size, feature_dim, w*h]
            batch_median = activation_map_flat.median(dim=2).values
            probs = F.softmax(activation_map_flat, dim=-1)
            batch_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

            median[start_ind:end_ind, :] = batch_median.data.cpu().numpy()
            entropy[start_ind:end_ind, :] = batch_entropy.data.cpu().numpy()

    return labels, avg, std, maxi, median, entropy
    # return labels, avg, avg, avg, avg, avg # only for ViT

def obtain_out_statistics(args, model, data_set, batch_size = None):
    if batch_size is None:
        batch_size = args.batch_size
    device = args.device
    data_size = len(data_set)
    feature_dim = model.dim_in

    avgpool = nn.AdaptiveAvgPool2d((1,1))
    maxpool = nn.AdaptiveMaxPool2d((1,1))

    avg = np.zeros((data_size, feature_dim))
    std = np.zeros((data_size, feature_dim))
    maxi = np.zeros((data_size, feature_dim))
    median = np.zeros((data_size, feature_dim))
    entropy = np.zeros((data_size, feature_dim))

    data_loader = dataset_loader(args, data_set, batch_size=batch_size)
    model.eval() # already set to eval during setting up model

    with torch.inference_mode():
        for batch_idx, (x,y) in tqdm(enumerate(data_loader)):
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, data_size)

            x = x.to(device)

            # +++++++++++++++++++++++ start: only for ViT ++++++++++++++++++++++++++
            # batch_avg = model.features(x)
            # batch_avg = batch_avg.view(-1, feature_dim)
            # avg[start_ind:end_ind, :]  = batch_avg.data.cpu().numpy()
            # +++++++++++++++++++++++ end: only for ViT ++++++++++++++++++++++++++++

            # get statistics: avg, std, max
            activation_map = model.my_encoder(x)           #[batch_size, feature_dim, height, width]
            batch_avg = avgpool(activation_map)            #[batch_size, feature_dim, 1, 1]
            batch_avg = batch_avg.view(-1, feature_dim)    #[batch_size, feature_dim]
            batch_std = activation_map.std(dim = (2, 3))   #[batch_size, feature_dim]
            batch_max = maxpool(activation_map)            #[batch_size, feature_dim, 1, 1]
            batch_max = batch_max.view(-1, feature_dim)    #[batch_size, feature_dim]

            avg[start_ind:end_ind, :]  = batch_avg.data.cpu().numpy()
            std[start_ind:end_ind, :]  = batch_std.data.cpu().numpy()
            maxi[start_ind:end_ind, :] = batch_max.data.cpu().numpy()
            

            #median and entropy
            b, d, w, h = activation_map.shape
            activation_map_flat = activation_map.view(b, d, -1)
            batch_median = activation_map_flat.median(dim=2).values
            probs = F.softmax(activation_map_flat, dim=-1)
            batch_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

            median[start_ind:end_ind, :] = batch_median.data.cpu().numpy()
            entropy[start_ind:end_ind, :] = batch_entropy.data.cpu().numpy()
       
    return avg, std, maxi, median, entropy
    # return avg, avg, avg, avg, avg # only for ViT

     
def main(args):

    # setting up statistics directory
    model_statistics_directory = setup_directory(args)
    # print(f"statistics location: {model_statistics_directory}")
    
    # model parameters and model setup
    print(f"evaluation parameter: {args}")
    print(f"setting up model: {args.model}")
    model = set_model(args)
    if os.path.exists(args.ckpt):
        print(f'loading existing model:{args.ckpt}')
        model.load_state_dict(torch.load(args.ckpt))
        model.eval()
    else:
        print(f"{args.ckpt} does not exit, check checkpoint information")
        return

#----------------------------------------------- ANALYZE ID DATA------------------------------------------#

    print('---------- Processing ID Starts ------------')
    weight_directory = os.path.join(model_statistics_directory, f"Weights/")
    in_features_directory = os.path.join(model_statistics_directory, f"{args.in_dataset}/")
    in_train_features_directory = os.path.join(model_statistics_directory, f"{args.in_dataset}/train")
    in_statistics_file_name = os.path.join(in_features_directory, f"in_")
    
    for directory in [weight_directory, in_features_directory, in_train_features_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    print(f"in stats file location: {weight_directory, in_features_directory}")

    # save W: weight matrix f(x) = W^Th(x) + b
    # save_weights(args, model, weight_directory)

    #save features
    # train_set, test_set = load_in_dataset(args)
    # labels, avg, std, maxi, median, entropy = obtain_in_statistics(args, model, test_set)
    # save_in_features(in_statistics_file_name, labels, avg, std, maxi, median, entropy)

    # save in features for knn experiments. 
    # labels, avg, std, maxi, median, entropy = obtain_in_statistics(args, model, train_set)
    # in_statistics_file_name = os.path.join(in_train_features_directory, f"in_")
    # save_in_features(in_statistics_file_name, labels, avg, std, maxi, median, entropy)
    # return

    print('---------- Processing ID Finished -------------')

#----------------------------------------------- ANALYZE OOD DATA-----------------------------------------#

    print('---------- Processing OOD Starts ------------')
    if args.in_dataset == 'ImageNet-1K':
        # out_datasets = ['imagenet_noise']
        # out_datasets = ['openimage-o']
        out_datasets = ['SUN', 'Places', 'imagenet_dtd', 'iNaturalist', 'openimage-o']
    elif args.in_dataset in ["CIFAR-10", "CIFAR-100"]: 
        # if args.in_dataset in ["CIFAR-10"]:
        #     out_datasets = ['cifar10_noise']
        # else:
        #     out_datasets = ['cifar100_noise']
        out_datasets = ['CIFAR-100'] # simulate Near-OOD for CIFAR-10
        # out_datasets = [ 'SVHN', 'places365', 'iSUN', 'dtd', 'LSUN', 'LSUN_resize']

    for out_dataset in out_datasets:
        args.out_dataset = out_dataset
        print(f"processing out_dataset: {out_dataset}")

        out_features_directory = os.path.join(model_statistics_directory, f"{args.out_dataset}")
        out_statistics_file_name = os.path.join(out_features_directory, f"out_")
        for directory in [out_features_directory]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        train_set, test_set = load_out_dataset(args) # train_set is dummy string.
        avg, std, maxi, median, entropy = obtain_out_statistics(args, model, test_set)
        save_out_features(out_statistics_file_name, avg, std, maxi, median, entropy)

    print('---------- Processing OOD Finished ------------')

    
if __name__ == '__main__':
    args = args_parser()
    main(args)