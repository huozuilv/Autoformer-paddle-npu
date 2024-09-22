import os
import paddle
import cv2
import json
import scipy
import scipy.io as sio
from skimage import io
#from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


import paddle.vision.transforms as transforms

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)





def build_dataset(is_train, args, folder_name=None):
    transform = build_transform(is_train, args)
    if args.data_set == 'CIFAR10':
        dataset = paddle.vision.datasets.Cifar10(data_file=None,mode='train', transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'CIFAR100':
        dataset = paddle.vision.datasets.Cifar100(args.data_path, mode='train', transform=transform, download=True)
        nb_classes = 100

    elif args.data_set == 'SUB_CIFAR10':
        subCIFAR10_path = os.path.join(args.data_path, 'subCIFAR10')
        dataset = paddle.vision.datasets.ImageFolder(subCIFAR10_path,
            transform=transform)
        nb_classes = 10
    elif args.data_set == 'SUB_CIFAR100':
        subCIFAR100_path = os.path.join(args.data_path, 'subCIFAR100')
        dataset = paddle.vision.datasets.ImageFolder(subCIFAR100_path,
            transform=transform)
        nb_classes = 100
    return dataset, nb_classes


# def build_transform(is_train, args):
#     resize_im = args.input_size > 32
#     if is_train:
#         transform = timm.data.create_transform(input_size=args.input_size,
#             is_training=True, color_jitter=args.color_jitter, auto_augment=
#             args.aa, interpolation=args.train_interpolation, re_prob=args.
#             reprob, re_mode=args.remode, re_count=args.recount)
#         if not resize_im:
#             transform.transforms[0] = paddle.vision.transforms.RandomCrop(args
#                 .input_size, padding=4)
#         return transform
#     t = []
#     if resize_im:
#         size = int(256 / 224 * args.input_size)
#         t.append(paddle.vision.transforms.Resize(size, interpolation=3))
#         t.append(paddle.vision.transforms.CenterCrop(args.input_size))
#         t.append(paddle.vision.transforms.ToTensor())
#         t.append(paddle.vision.transforms.Normalize(IMAGENET_DEFAULT_MEAN,
#         IMAGENET_DEFAULT_STD))
#     return paddle.vision.transforms.Compose(t)

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=args.color_jitter, contrast=args.color_jitter, saturation=args.color_jitter, hue=0.1) if args.color_jitter > 0 else None,
            #transforms.RandomGrayscale(p=0.2) if args.aa == 'rand-m9-mstd0.5' else None,
            transforms.RandomAffine(degrees=10) if args.aa == 'rand-m9-mstd0.5' else None,
            #transforms.RandomErasing(probability=args.reprob, mode=args.remode, max_count=args.recount, num_splits=0) if args.reprob > 0 else None,
            transforms.RandomErasing() if args.reprob > 0 else None,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        transform = [t for t in transform.transforms if t is not None]
        return transforms.Compose(transform)
    else:
        t = []
        if resize_im:
            size = int(256 / 224 * args.input_size)
            t.append(transforms.Resize(size, interpolation='bicubic'))
            t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)