from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    RandomBrightnessContrast,
    Resize,
)
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]  # ImageNet values
STD = [0.229, 0.224, 0.225]  # ImageNet values


# ====================================================
# Transforms
# ====================================================
def get_transforms(mode, mean, std, size):

    if mode == "train":
        return Compose(
            [
                Resize(size, size),
                HorizontalFlip(p=0.5),
                RandomBrightnessContrast(
                    p=0.5, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), brightness_by_max=False
                ),
                Normalize(
                    mean=mean,
                    std=std,
                ),
                ToTensorV2(),
            ]
        )

    elif mode == "valid":
        return Compose(
            [
                Resize(size, size),
                Normalize(
                    mean=mean,
                    std=std,
                ),
                ToTensorV2(),
            ]
        )
