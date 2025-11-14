import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# def ash_s(x, p=90):
#     s1 = x.sum(dim = 1)
#     t = torch.quantile(x, p/100.0, dim=1, keepdim=True)
#     x = torch.where(x > t, x, torch.zeros_like(x))
#     s2 = x.sum(dim = 1)
#     scale = s1/s2
#     x = x * torch.exp(scale[:, None])
#     return x


def ash_s(x, p):
    # x := [bt, d]
    batch_size, c = x.shape

    s1 = x.sum(dim=1) 
    k = c - int(np.round(c * p / 100.0))
    v, i = torch.topk(x, k, dim=1)
    x_pruned = torch.zeros_like(x)
    x_pruned.scatter_(dim=1, index=i, src=v)

    s2 = x_pruned.sum(dim=1)
    scale = s1 / (s2 ) #+ 1e-8)
    x_sharpened = x_pruned * torch.exp(scale[:, None])

    return x_sharpened


def custom_operation(x, pool, k):
    # x := [bt, d, w, h]
    avg_pool = nn.AdaptiveAvgPool2d((1,1))
    max_pool = nn.AdaptiveMaxPool2d((1,1))
    if pool == 'avg':
        x = avg_pool(x)
    elif pool == 'max':
        x = max_pool(x)
    elif pool == 'avg+std':
        mean = x.mean(dim = (2, 3))
        std = x.std(dim = (2, 3))
        x = mean + k * std
    elif pool == 'all':
        b,c,h,w = x.shape
        mean = x.mean(dim = (2, 3))
        std = x.std(dim = (2, 3))
        x = max_pool(x)
        x = x.view(-1, c)
        x = x + mean + k*std
    elif pool == 'median':
        b,c,h,w = x.shape
        x_flat = x.view(b,c,-1)
        x = x_flat.median(dim=2).values
    elif pool == 'entropy':
        b,c,h,w = x.shape
        x_flat = x.view(b,c,-1)
        probs = F.softmax(x_flat, dim=-1)
        x = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return x


class RouteDICE(nn.Linear):

    def __init__(self, in_features, out_features, device, bias=True, p=90, conv1x1=False, info=None):
        super(RouteDICE, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.p = p
        self.info = info
        self.masked_w = None
        self.device = device
        # print(f" inside RouteDICE: {self.info}")
        

    def calculate_mask_weight(self):
        self.contrib = self.info[None, :] * self.weight.data.cpu().numpy()
        self.thresh = np.percentile(self.contrib, self.p)
        mask = torch.Tensor((self.contrib > self.thresh))
        self.masked_w = (self.weight.squeeze().cpu() * mask).to(self.device)

    # static pruning 
    def forward(self, input):
        if self.masked_w is None:
            self.calculate_mask_weight()
        vote = input[:, None, :] * self.masked_w.to(self.device)
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out
    
    # # dynamic pruning 
    # def forward(self, input):
    #     batch_size = input.shape[0]
    #     contribution = input[:, None, :] * self.weight
    #     thresholds = torch.quantile(contribution.view(batch_size, -1), self.p/100.0, dim=1, keepdim=True)
    #     thresholds = thresholds.view(batch_size, 1, 1)
    #     top_contribution = torch.where(contribution > thresholds, contribution, torch.zeros_like(contribution))

    #     out = top_contribution.sum(2)
    #     if self.bias is not None:
    #         out = out + self.bias
    #     return out