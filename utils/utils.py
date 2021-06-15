"""Some useful functions"""

import math
import os
import random
import time

import numpy as np
import torch


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "{} (remain {})".format(asMinutes(s), asMinutes(rs))


def save_model(model, epoch, trainloss, valence, arousal, name):
    """Saves PyTorch model."""
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "train_loss": trainloss,
            "val_valence": valence,
            "val_arousal": arousal,
        },
        os.path.join("weights", name),
    )


def load_model(model, path_to_model):
    cp = torch.load(path_to_model)
    epoch, train_loss, valence, arousal = None, None, None, None
    if "model" in cp:
        model.load_state_dict(cp["model"], strict=False)
    if "epoch" in cp:
        epoch = int(cp["epoch"])
    if "train_loss" in cp:
        train_loss = cp["train_loss"]
    if "val_valence" in cp:
        valence = cp["val_valence"]
    if "val_arousal" in cp:
        arousal = cp["val_arousal"]
    print(
        "Uploading model from the checkpoint...",
        "\nEpoch:",
        epoch,
        "\nTrain Loss:",
        train_loss,
        "\nVal valence:",
        valence,
        "\nMetrics:",
        arousal,
    )
    return cp
