import torchvision.transforms as transforms


class ResizeImage(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class TransformsSimCLR:
    def __init__(self, isCenter, crop_size):
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        if isCenter:
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([color_jitter], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        else:
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(crop_size),
                    transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    transforms.RandomApply([color_jitter], p=0.8),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __call__(self, x):
        x = self.train_transform(x)
        return x
