from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    RandomBrightnessContrast,
    Resize,
    Rotate,
)
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]  # ImageNet values
STD = [0.229, 0.224, 0.225]  # ImageNet values


# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):

    if data == "train":
        return Compose(
            [
                HorizontalFlip(p=0.5),
                Rotate(p=0.5, limit=(-30, 30)),
                RandomBrightnessContrast(
                    p=0.5, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), brightness_by_max=False
                ),
                Normalize(
                    mean=MEAN,
                    std=STD,
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return Compose(
            [
                Normalize(
                    mean=MEAN,
                    std=STD,
                ),
                ToTensorV2(),
            ]
        )
