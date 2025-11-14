import os
import time
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset, TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR

from configs import ModelConfig
from utils.datasets import load_in_dataset, dataset_loader
from utils.utils import set_model, get_device, get_current_human_time, set_global_seed, print_model_summary, validate, accuracy, set_optimizer, redirect_stdout_to_file

def args_parser():
    parser = argparse.ArgumentParser(description='training module')

    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--p', default=None, type=int, help='DICE prunning level')

    parser.add_argument('--num_classes', default=10, type=int, help='number of classes in in-dataset')
    parser.add_argument('--dim_in', default = 512, type=int, help='penultimate feature dim')
    parser.add_argument('--in-dataset', default="CIFAR-10 ", type=str, help='in-distribution dataset')
    parser.add_argument('--id_loc', default="datasets/in/", type=str, help='location of in-distribution dataset')
    parser.add_argument('--pool', default = 'avg', type=str, help='custom operation in [avg, max, avg+std, median]')
    parser.add_argument('--model', default='resnet', type=str, help='model architecture: [resnet18, resnet34, resnet101, densenet101]')
    
    parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--save-epoch', default=25, type=int, help='save the model every save_epoch')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default= [50, 75, 90], help='list of milestone, where to decay lr')

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--optimizer_type', default='sgd', type=str, help='optimizer type')
    parser.add_argument('--normalize', action='store_true', help='normalize feat embeddings')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--temp', type=float, default=0.1, help='temperature for loss function')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 0.0001)')
   
    device = get_device()
    parser.add_argument('--device', type=torch.device, default=device, help = 'device type for accelerated computation')
    args = parser.parse_args()

    args.ckpt = f"model.pt"

    if args.in_dataset in ["CIFAR-10"]:
        args.num_classes = 10
    elif args.in_dataset in ["CIFAR-100"]:
        args.num_classes = 100
    elif args.in_dataset in ["ImageNet-1K"]:
        args.num_classes = 1000

    return args

def build_related_dirs(args):
    base_directory_name = f"{args.model}/{args.in_dataset}/"
    model_checkpoint_directory = os.path.join(ModelConfig.model_checkpoint_directory, base_directory_name)
    runtime_log_directory = model_checkpoint_directory.replace(ModelConfig.model_checkpoint_directory, ModelConfig.runtime_log_directory)
    for curr_dir in [model_checkpoint_directory, runtime_log_directory]:
        os.makedirs(curr_dir, exist_ok=True)
    return model_checkpoint_directory, runtime_log_directory

def save_checkpoint(model, model_checkpoint_name, model_checkpoint_directory, epoch = None):
    if epoch is None:
        torch.save(model.state_dict(), os.path.join(model_checkpoint_directory, f"{model_checkpoint_name}.pt"))
    else:
        torch.save(model.state_dict(), os.path.join(model_checkpoint_directory, f"{model_checkpoint_name}_at_epoch_{epoch}.pt"))

def evaluate_model(args, model, test_loader):
    test_accuracy = accuracy(args, model, test_loader)
    return test_accuracy

def train(args, model, train_set, test_set, model_checkpoint_name, model_checkpoint_directory):
    start_time = time.time()
    print(f"\r[{get_current_human_time()}][START TRAINING] model checkpiont name: {model_checkpoint_name} - checkpoint directory: {model_checkpoint_directory}")
    
    # load pre-trained model is exist
    checkpoint_loc = os.path.join(model_checkpoint_directory, args.ckpt)
    if os.path.exists(checkpoint_loc):
        print(f'loading existing model:{checkpoint_loc}')
        model.load_state_dict(torch.load(checkpoint_loc))

    test_loader = dataset_loader(args, test_set, batch_size=args.batch_size)
    loss_fn = nn.CrossEntropyLoss()
    print(f'loading dataset: {args.in_dataset} with batch size : {args.batch_size}')
    train_loader = dataset_loader(args, train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = set_optimizer(args=args, model=model)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate)

    best_model = deepcopy(model)
    
    # evaluate model before training
    loss = 0.0
    loss = validate(args, model, test_loader, loss_fn)
    best_epoch_loss = loss
    test_accuracy = evaluate_model(args, model, test_loader)
    print("prior to training, validation loss = {:.4f}, test accuracy = {:.2f}".format(loss, test_accuracy*100))

    for epoch in tqdm(range(args.start_epoch, args.epochs)): 
        #-------------------- training one epoch starts -------------------------#
        # epoch_start_time = time.time()
        model.train()
        device = args.device
        train_size, mean_loss, loss = 0, 0.0, torch.tensor(0.0, device=device)
        
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            #optimizer step
            optimizer.zero_grad()
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            # track training loss
            train_size += len(x)
            mean_loss += len(x) * (loss.item() - mean_loss)/train_size

        #-------------------- Training one epoch ends -------------------------#
        scheduler.step()
        # print(f"learning rate: {optimizer.param_groups[0]['lr']}")
    
        test_accuracy = evaluate_model(args, model, test_loader)
        mean_loss = validate(args, model, test_loader, loss_fn)
        print("at epochs = {}, validation loss = {:.4f}, test accuracy = {:.2f}".format(epoch + 1, mean_loss, test_accuracy*100))

        if mean_loss < best_epoch_loss:
            best_epoch_loss = mean_loss
            best_model = deepcopy(model)
            save_checkpoint(model, model_checkpoint_name, model_checkpoint_directory)

        #save the  model every 10 epochs:
        if (epoch + 1) % args.save_epoch == 0:
            save_checkpoint(model, model_checkpoint_name, model_checkpoint_directory, epoch=epoch+1)

    model = deepcopy(best_model)
    save_checkpoint(model, model_checkpoint_name, model_checkpoint_directory)
    total_time = time.time() - start_time
    print(f"total training for {args.model} using {args.in_dataset}: {total_time/60} minutes")

def main(args):
    set_global_seed(args.seed)
    model_checkpoint_directory, runtime_log_directory = build_related_dirs(args)
    
    if not model_checkpoint_directory:
        print(f'model checkpoint directory does not exist')
        return 0
    print("[training process started]\n model_checkpoint_directory: {} \n".format(model_checkpoint_directory))

    model_checkpoint_name = f"model"
    runtime_log_writer = redirect_stdout_to_file(runtime_log_directory, f"{model_checkpoint_name}.log")
    print(runtime_log_directory)
    model = set_model(args)
    train_set, test_set = load_in_dataset(args)

    print_model_summary(model)
    print(f"args dictionary: {args.__dict__}")
    print(f"[Model:{args.model} - Dataset:{args.in_dataset} - Train Size:{len(train_set)} - Test Size:{len(test_set)}")
    
    # training the  model
    train(args, model, train_set, test_set, model_checkpoint_name=model_checkpoint_name, model_checkpoint_directory=model_checkpoint_directory)
    runtime_log_writer.close()

if __name__ == '__main__':
   args = args_parser() 
   main(args)