"""Some useful functions"""

import math
import os
import random
import time

import matplotlib.pyplot as plt
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
        "\nVal arousal:",
        arousal,
    )
    return cp


def save_image(input_tensor, y_item, landmarks, fig_path, index, mean, std):
    """Show a single image."""
    image = input_tensor.permute(1, 2, 0).numpy()
    image = std * image + mean
    plt.imshow(image.clip(0, 1))
    plt.plot(*zip(*landmarks), marker="o", color="r", ls="")
    title = str(y_item[0]) + str(", ") + str(y_item[1])
    plt.title(title)
    fig_name = f"{index}.png"
    fig_path = fig_path
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()


def save_batch(dataloader, fig_path, seq_indx, mean, std):
    """Show images for a batch."""
    X_batch, y_batch, landmark_batch = next(iter(dataloader))
    print("Saving batch...")
    X_batch = X_batch[:, seq_indx, :, :, :]
    y_batch = y_batch[:, seq_indx, :]
    landmark_batch = landmark_batch[:, seq_indx, :]
    for index, (x_item, y_item, landmarks) in enumerate(zip(X_batch, y_batch, landmark_batch)):
        save_image(x_item, y_item, landmarks, fig_path, index, mean, std)
    print("Batch has been saved!")
