import torch

from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode


def GetMixupCutmix(*, mixup_alpha, cutmix_alpha, num_classes):
    mixup_cutmix = []
    if mixup_alpha > 0:
        mixup_cutmix.append(v2.MixUp(alpha=mixup_alpha, num_classes=num_classes))
    if cutmix_alpha > 0:
        mixup_cutmix.append(v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes))
    if not mixup_cutmix:
        return None  # nothing happened
    return v2.RandomChoice(mixup_cutmix)


def GetTrTransform(s, mean, std, augment='ra', raMag=9, amSev=3, flip=True, erase=0.0):
    '''
    ra: RandAugment (default magnitude: 9)
    ta: TrivialAugmentWide
    am: AugMix (default severity: 3)
    in: AutoAugmentPolicy.IMAGENET
    cf: AutoAugmentPolicy.CIFAR10
    sv: AutoAugmentPolicy.SVHN
    '''
    assert augment in [None, 'ra', 'ta', 'am', 'in', 'cf', 'sv']
    ipMode = InterpolationMode.BILINEAR
    transform = []
    transform.append(v2.RandomCrop(s, padding=4))
    if flip:
        transform.append(v2.RandomHorizontalFlip())
    if augment is not None:
        if augment == 'ra':
            transform.append(v2.RandAugment(interpolation=ipMode, magnitude=raMag))
        elif augment == 'ta':
            transform.append(v2.TrivialAugmentWide(interpolation=ipMode))
        elif augment == 'am':
            transform.append(v2.AugMix(interpolation=ipMode, severity=amSev))
        elif augment == 'in':
            transform.append(v2.AutoAugment(policy=v2.AutoAugmentPolicy.IMAGENET, interpolation=ipMode))
        elif augment == 'cf':
            transform.append(v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10, interpolation=ipMode))
        elif augment == 'sv':
            transform.append(v2.AutoAugment(policy=v2.AutoAugmentPolicy.SVHN, interpolation=ipMode))
        else:
            raise ValueError(f"Augmentation {augment} is not recognized.")
    transform.append(v2.ToImage())
    transform.append(v2.ToDtype(torch.float, scale=True))
    if erase > 0.0:
        transform.append(v2.RandomErasing(p=erase))
    transform.append(v2.Normalize(mean=mean, std=std))
    return v2.Compose(transform)


def GetVaTransform(s, mean, std, rs=256, crop=False):
    ipMode = InterpolationMode.BILINEAR
    transform = []
    if crop:
        transform.append(v2.Resize(rs, interpolation=ipMode, antialias=True))
        transform.append(v2.CenterCrop(s))
    transform.append(v2.ToImage())
    transform.append(v2.ToDtype(torch.float, scale=True))
    transform.append(v2.Normalize(mean=mean, std=std))
    return v2.Compose(transform)
