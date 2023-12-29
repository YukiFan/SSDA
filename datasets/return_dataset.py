import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists, return_classlist
from datasets.transforms import ResizeImage, TransformsSimCLR
from datasets.randaugment import RandAugmentMC


def return_dataset(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset

    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val_mask': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'strong': TransformsSimCLR(isCenter=False, crop_size=crop_size),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'self': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=crop_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    }
    source_dataset = Imagelists(image_set_file_s, root=root, transform=data_transforms['train'],
                                strong_labeled_transform=data_transforms['self'])
    target_dataset = Imagelists(image_set_file_t, root=root, transform=data_transforms['train'],
                                strong_labeled_transform=data_transforms['self'])
    target_dataset_val = Imagelists(image_set_file_t_val, root=root, transform=data_transforms['val'])
    target_dataset_unl = Imagelists(image_set_file_unl, root=root,
                                    transform_mask=data_transforms['val_mask'], strong_transform=data_transforms['strong'])
    target_dataset_test = Imagelists(image_set_file_unl, root=root,
                                     transform=data_transforms['test'],
                                     test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))

    return source_dataset, target_dataset, target_dataset_unl, target_dataset_val, target_dataset_test, class_list
