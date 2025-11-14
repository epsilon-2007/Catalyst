import os
import torch
import argparse
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, Subset, TensorDataset, DataLoader

#---------------------------------------------------- FEATURE DATASET ----------------------------------------------------#

class FeatureDataset(Dataset):
    def __init__(self, features_path):
        self.features = np.load(features_path).astype(np.float32)  # shape: [N, D]
        self.scales = np.ones(len(self.features), dtype=np.int64)  # dummy scales
        # assert len(self.scales) == len(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        scale = self.scales[idx]
        return torch.from_numpy(feature), torch.tensor(scale)
    
def load_feature_dataset(feature_path):
    dataset = FeatureDataset(feature_path)
    return dataset

class FeatureScaleDataset(Dataset):
    def __init__(self, features_path, scales_path):
        self.scales = np.load(scales_path).astype(np.float32)
        self.features = np.load(features_path).astype(np.float32)
        assert len(self.scales) == len(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        scale = self.scales[idx]
        return torch.tensor(feature), torch.tensor(scale)
    
def load_feature_scale_dataset(scale_path, score_path):
    dataset = FeatureScaleDataset(scale_path, score_path)
    return dataset

# class ScaleScoreDataset(Dataset):
#     def __init__(self, scale_path, score_path):
#         self.scales = np.load(scale_path).astype(np.float32)
#         self.scores = np.load(score_path).astype(np.float32)

#         assert len(self.scales) == len(self.scores)

#     def __len__(self):
#         return len(self.scores)

#     def __getitem__(self, idx):
#         scores = self.scores[idx]
#         scales = self.scales[idx]
#         return torch.tensor(scores), torch.tensor(scales)
    
# def load_scale_score_dataset(scale_path, score_path):
#     dataset = ScaleScoreDataset(scale_path, score_path)
#     return dataset


#---------------------------------------------------- SET UP ID DATASET ----------------------------------------------------#
def load_cifar10(args):
    root = args.id_loc
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                     std=[0.247, 0.244, 0.262])
                                     
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    test_set =  datasets.CIFAR10(root, train=False, download=True, transform=transform_test)
    train_set = datasets.CIFAR10(root, train=True,  download=True, transform=transform_train)

    return train_set, test_set


def load_cifar100(args):
    root = args.id_loc
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                     std=[0.247, 0.244, 0.262])
 
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    test_set =  datasets.CIFAR100(root, train=False, download=True, transform=transform_test)
    train_set = datasets.CIFAR100(root, train=True,  download=True, transform=transform_train)

    return train_set, test_set

def load_imagenet1k(args):
    root = args.id_loc
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
   
    test_set =  datasets.ImageNet(root, split='val',   transform=transform_test)
    train_set = datasets.ImageNet(root, split='train', transform=transform_train)

    fraction = 0.15
    subset_size = int(len(train_set) * fraction)
    remaining_size = len(train_set) - subset_size
    train_subset, _ = torch.utils.data.random_split(
        train_set,
        [subset_size, remaining_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )

    return train_subset, test_set

    train_set = "dummy"
    test_set = datasets.ImageFolder(root = root, transform = transform_test)
    #passing test_set as train_set just to setup the directory structure within the project. It should be dummy, because we are using pre-trained model
    return test_set, test_set

    test_set =  datasets.ImageNet(root, split='val',   transform=transform_test)
    train_set = datasets.ImageNet(root, split='train', transform=transform_train)


def load_in_dataset(args):
    if args.in_dataset == 'CIFAR-10':
        return load_cifar10(args)
    elif args.in_dataset == 'CIFAR-100':
        return load_cifar100(args)
    elif args.in_dataset == 'ImageNet-1K':
        return load_imagenet1k(args)



#---------------------------------------------------- SET UP OOD DATASET ----------------------------------------------------#
   
def load_svhn(args, num_samples = 10000, img_size = 32):
    root = args.ood_loc
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.244, 0.262])
    transform = transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(img_size),
                                transforms.ToTensor(), 
                                normalize
                ])
    svhn_test_set = datasets.SVHN(root = root, split = 'test', download = True, transform = transform)
    svhn_train_set = "dummy" # datasets.SVHN(root = root, split = 'train', download = True, transform = transform)

    if len(svhn_test_set) > num_samples:
        svhn_test_set = torch.utils.data.Subset(svhn_test_set, np.random.choice(len(svhn_test_set), num_samples, replace=False))
    
    return svhn_train_set, svhn_test_set

def load_imagenet_iNaturalist(args, num_samples = 10000, img_size = 256):
    root = os.path.join(args.ood_loc, args.out_dataset)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                normalize
                ])
    
    imagenet_iNaturalist_test_set = datasets.ImageFolder(root = root, transform = transform)
    imagenet_iNaturalist_train_set = "dummy"

    # if len(imagenet_iNaturalist_test_set) > num_samples:
    #     imagenet_iNaturalist_test_set = torch.utils.data.Subset(imagenet_iNaturalist_test_set, np.random.choice(len(imagenet_iNaturalist_test_set), num_samples, replace=False))

    return imagenet_iNaturalist_train_set, imagenet_iNaturalist_test_set

def load_imagenet_openimage(args, num_samples = 10000, img_size = 256):
    root = os.path.join(args.ood_loc, args.out_dataset)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                normalize
                ])
    
    imagenet_iopenimage_test_set = datasets.ImageFolder(root = root, transform = transform)
    imagenet_openimage_train_set = "dummy"

    return imagenet_openimage_train_set, imagenet_iopenimage_test_set

def load_imagenet_Places(args, num_samples = 10000, img_size = 256):
    root = os.path.join(args.ood_loc, args.out_dataset)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                normalize
                ])
    
    imagenet_places_test_set = datasets.ImageFolder(root = root, transform = transform)
    imagenet_places_train_set = "dummy"

    # if len(imagenet_places_test_set) > num_samples:
    #     imagenet_places_test_set = torch.utils.data.Subset(imagenet_places_test_set, np.random.choice(len(imagenet_places_test_set), num_samples, replace=False))

    return imagenet_places_train_set, imagenet_places_test_set

def load_places365(args, num_samples = 10000, img_size = 32):
    root = os.path.join(args.ood_loc, args.out_dataset)
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.244, 0.262])
    transform = transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(img_size),
                                transforms.ToTensor(), 
                                normalize
                ])
    
    places365_test_set = datasets.ImageFolder(root = root, transform = transform)
    places365_train_set = "dummy"

    if len(places365_test_set) > num_samples:
        places365_test_set = torch.utils.data.Subset(places365_test_set, np.random.choice(len(places365_test_set), num_samples, replace=False))

    return places365_train_set, places365_test_set


def load_iSUN(args, num_samples = 10000, img_size = 32):
    root = os.path.join(args.ood_loc, args.out_dataset)
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.244, 0.262])
    transform = transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(img_size),
                                transforms.ToTensor(), 
                                normalize
                ])
    iSUN_test_set =  datasets.ImageFolder(root = root, transform = transform)
    iSUN_train_set = "dummy"

    if len(iSUN_test_set) > num_samples:
        iSUN_test_set = torch.utils.data.Subset(iSUN_test_set, np.random.choice(len(iSUN_test_set), num_samples, replace=False))

    return iSUN_train_set, iSUN_test_set

def load_imagenet_dtd(args, num_samples = 10000, img_size = 256):
    root = os.path.join(args.ood_loc, args.out_dataset, "images")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                normalize
                ])
    dtd_test_set =  datasets.ImageFolder(root = root, transform = transform)
    dtd_train_set = "dummy"

    # if len(dtd_test_set) > num_samples:
    #     dtd_test_set = torch.utils.data.Subset(dtd_test_set, np.random.choice(len(dtd_test_set), num_samples, replace=False))

    return dtd_train_set, dtd_test_set

def load_imagenet_noise(args, num_samples = 10000, img_size = 256):
    root = os.path.join(args.ood_loc, args.out_dataset)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                normalize
                ])
    
    imagenet_noise_test_set = datasets.ImageFolder(root = root, transform = transform)
    imagenet_noise_train_set = "dummy"

    # if len(imagenet_noise_test_set) > num_samples:
    #     imagenet_noise_test_set = torch.utils.data.Subset(imagenet_noise_test_set, np.random.choice(len(imagenet_noise_test_set), num_samples, replace=False))

    return imagenet_noise_train_set, imagenet_noise_test_set

def load_dtd(args, num_samples = 10000, img_size = 32):
    root = os.path.join(args.ood_loc, args.out_dataset, "images")
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.244, 0.262])
    transform = transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(img_size),
                                transforms.ToTensor(), 
                                normalize
                ])
    dtd_test_set =  datasets.ImageFolder(root = root, transform = transform)
    dtd_train_set = "dummy"

    if len(dtd_test_set) > num_samples:
        dtd_test_set = torch.utils.data.Subset(dtd_test_set, np.random.choice(len(dtd_test_set), num_samples, replace=False))

    return dtd_train_set, dtd_test_set

def load_imagenet_SUN(args, num_samples = 10000, img_size = 256):
    root = os.path.join(args.ood_loc, args.out_dataset)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                normalize
                ])
    lsun_test_set =  datasets.ImageFolder(root = root, transform = transform)
    lsun_train_set = "dummy"

    # if len(lsun_test_set) > num_samples:
    #     lsun_test_set = torch.utils.data.Subset(lsun_test_set, np.random.choice(len(lsun_test_set), num_samples, replace=False))

    return lsun_train_set, lsun_test_set

def load_LSUN(args, num_samples = 10000, img_size = 32):
    root = os.path.join(args.ood_loc, args.out_dataset)
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.244, 0.262])
    transform = transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(img_size),
                                transforms.ToTensor(), 
                                normalize
                ])
    lsun_test_set =  datasets.ImageFolder(root = root, transform = transform)
    lsun_train_set = "dummy"

    if len(lsun_test_set) > num_samples:
        lsun_test_set = torch.utils.data.Subset(lsun_test_set, np.random.choice(len(lsun_test_set), num_samples, replace=False))

    return lsun_train_set, lsun_test_set

def load_LSUN_resize(args, num_samples = 10000, img_size = 32):
    root = os.path.join(args.ood_loc, args.out_dataset)
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.244, 0.262])
    transform = transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(img_size),
                                transforms.ToTensor(), 
                                normalize
                ])
    lsun_resize_test_set =  datasets.ImageFolder(root = root, transform = transform)
    lsun_resize_train_set = "dummy"

    if len(lsun_resize_test_set) > num_samples:
        lsun_resize_test_set = torch.utils.data.Subset(lsun_resize_test_set, np.random.choice(len(lsun_resize_test_set), num_samples, replace=False))

    return lsun_resize_train_set, lsun_resize_test_set

def load_cifar_noise(args, num_samples = 10000, img_size = 32):
    root = os.path.join(args.ood_loc, args.out_dataset)
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.244, 0.262])
    transform = transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(img_size),
                                transforms.ToTensor(), 
                                normalize
                ])
    cifar_noise_test_set =  datasets.ImageFolder(root = root, transform = transform)
    cifar_noise_train_set = "dummy"

    if len(cifar_noise_test_set) > num_samples:
        cifar_noise_test_set = torch.utils.data.Subset(cifar_noise_test_set, np.random.choice(len(cifar_noise_test_set), num_samples, replace=False))

    return cifar_noise_train_set, cifar_noise_test_set

    
def load_out_dataset(args):
    if args.out_dataset == 'SVHN':
        return load_svhn(args)
    elif args.out_dataset == 'places365':
        return load_places365(args)
    elif args.out_dataset == 'iSUN':
        return load_iSUN(args)
    elif args.out_dataset == 'dtd':
        return load_dtd(args)
    elif args.out_dataset == 'LSUN':
        return load_LSUN(args)
    elif args.out_dataset == 'LSUN_resize':
        return load_LSUN_resize(args)
    elif args.out_dataset == 'imagenet_dtd':
        return load_imagenet_dtd(args)
    elif args.out_dataset == 'Places':
        return load_imagenet_Places(args)
    elif args.out_dataset == 'SUN':
        return load_imagenet_SUN(args)
    elif args.out_dataset == 'iNaturalist':
        return load_imagenet_iNaturalist(args)
    elif args.out_dataset == 'openimage-o':
        return load_imagenet_openimage(args)
    elif args.out_dataset == 'CIFAR-100':
        return load_cifar100(args)
    elif args.out_dataset == 'cifar_noise':
        return load_cifar_noise(args)
    elif args.out_dataset == 'imagenet_noise':
        return load_imagenet_noise(args)
    
def dataset_loader( args, datasets, batch_size = None, shuffle = False, drop_last = False):
    if batch_size is None:
        batch_size = args.batch_size
    try:
        data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    except Exception as e:
        print(f"an error {e} occured making data loader of {args.in_dataset}")
    return data_loader

def args_parser():
    parser = argparse.ArgumentParser(description='sanity checks for datasets')

    parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
    parser.add_argument('--out-dataset', default="omniglot", type=str, help='out-distribution dataset')
    parser.add_argument('--id_loc', default="./../datasets/in/", type=str, help='location of in-distribution dataset')
    parser.add_argument('--ood_loc', default="./../datasets/ood/", type=str, help='location of in-distribution dataset')

    args = parser.parse_args()
    return args

def main(args):
    train_set, test_set = load_in_dataset(args)
    print(f"test set size: {len(test_set)}")
    print(f"training set size: {len(train_set)}")
    # x,y = train_set[0][0], train_set[0][1]
    # print(f"shape of images: {x.shape} and lable is: {y}")
    load_out_dataset(args)

if __name__ == '__main__':
    args = args_parser()
    main(args)