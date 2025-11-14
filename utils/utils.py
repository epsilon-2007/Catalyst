import os
import sys
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import torch.nn.functional as F

from experiments.model.resnet import MyResNet
from experiments.model.densenet import MyDenseNet
from experiments.model.imagenet import  resnet34, resnet50, mobilenet_v2, densenet121


from datetime import datetime
from prettytable import PrettyTable

import time
from datetime import datetime

def get_device():
    # get type of device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def validate(args, model, val_loader, loss_fn = nn.CrossEntropyLoss()):
    '''measure performace on validation set.'''
    model.eval()
    device = args.device
    with torch.inference_mode():
        N = 0
        mean_loss = 0
        
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = loss_fn(outputs, y)

            N += len(x)
            mean_loss += len(x) * (loss.item() - mean_loss)/N         
    return mean_loss


def accuracy(args, model, data_loader):
    model.eval()
    device = args.device
    with torch.inference_mode():
      total = 0
      correct = 0
      for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        prob = torch.softmax(output, dim = 1)
        pred = prob.max(dim = 1, keepdim = True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        total += len(y)
    acc = correct/total
    return acc

def redirect_stdout_to_file(file_dir, file_name):
    out_log_f = open(os.path.join(file_dir, file_name), "w")
    sys.stderr = out_log_f
    sys.stdout = out_log_f
    return out_log_f

def get_current_human_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_current_timestamp():
    return int(time.time())

def print_model_summary(model):
    # summary(model, input_size=(64, 3, 32, 32))
    columns = ["Modules", "Parameters", "Param Shape"]
    table = PrettyTable(columns)
    for i, col in enumerate(columns):
        if i == 0:
            table.align[col] = "l"
        else:
            table.align[col] = "r"
    total_param_nums = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param_nums = parameter.numel()
        param_shape = list(parameter.shape)
        table.add_row([name, "{:,}".format(param_nums), "{}".format(param_shape)])
        total_param_nums += param_nums

    separator = ["-" * len(x) for x in table.field_names]
    table.add_row(separator)
    table.add_row(["Total", "{:,}".format(total_param_nums), "{}".format("_")])

    print(table, "\n")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_model(args):
    if args.in_dataset in ['CIFAR-10', 'CIFAR-100']:
        if args.model in ['resnet18']:
            model = MyResNet(args)
        elif args.model in ['densenet101']:
            model = MyDenseNet(args)
    else:
        if args.model in ['resnet34_imagenet']:
            model = resnet34(args)
        elif args.model in ['resnet50_imagenet']:
            model = resnet50(args)
        elif args.model in ['mobilenetv2_imagenet']:
            model = mobilenet_v2(args)
        elif args.model in ['densenet121_imagenet']:
            model = densenet121(args)
            
    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    model = model.to(args.device)
    return model


def set_optimizer(args, model):
    if args.optimizer_type == 'sgd':
        print(f'setting up sgd optimizer with lr: {args.learning_rate}')
        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    elif args.optimizer_type == 'adam':
        print(f'setting up Adam optimizer with lr: {args.learning_rate}')
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    return optimizer

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * ( 1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def set_global_seed(seed=1):
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)