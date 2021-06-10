import time

import torch
from tqdm import tqdm

from utils.utils import AverageMeter, timeSince

print_freq = 5


def train_one_epoch(epoch, model, device, train_loader, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()
    start = end = time.time()

    # Iterate over dataloader
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        # zero the gradients
        optimizer.zero_grad()
        # distribute data to device
        images, labels = images.to(device), labels.to(device)

        batch_size = images.size(0)

        # print("Input size: ", images.shape)
        # print("Target size: ", labels.shape)

        # Forward
        output = rnn_decoder(cnn_encoder(images))  # output has dim = (batch, number of classes)
        # print("Output size: ", output.shape)
        valence_loss = criterion(output[:, :, 0], labels[:, :, 0])
        # print("Loss size: ", valence_loss.shape)
        arousal_loss = criterion(output[:, :, 1], labels[:, :, 1])
        # Compute loss
        loss = 1 - (valence_loss + arousal_loss) / 2.0
        # Backward
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), batch_size)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % print_freq == 0 or batch_idx == (len(train_loader) - 1):
            print(
                "Epoch: [{Epoch:d}][{Iter:d}/{Len:d}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    Epoch=epoch + 1,
                    Iter=batch_idx,
                    Len=len(train_loader),
                    data_time=data_time,
                    loss=losses,
                    remain=timeSince(start, float(batch_idx + 1) / len(train_loader)),
                )
            )
    return losses.avg


def val_one_epoch(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    valence = AverageMeter()
    arousal = AverageMeter()
    # set models to eval mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()
    start = end = time.time()
    for step, (images, labels) in enumerate(tqdm(valid_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)

        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            output = rnn_decoder(cnn_encoder(images))  # output has dim = (batch, number of classes)
            valence_loss = criterion(output[:, :, 0], labels[:, :, 0])
            arousal_loss = criterion(output[:, :, 1], labels[:, :, 1])
        valence.update(valence_loss.item(), batch_size)
        arousal.update(arousal_loss.item(), batch_size)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{Step:d}/{Len:d}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Valence: {valence.val:.4f}({valence.avg:.4f}) "
                "Arousal: {arousal.val:.4f}({arousal.avg:.4f}) ".format(
                    Step=step,
                    Len=len(valid_loader),
                    data_time=data_time,
                    valence=valence,
                    arousal=arousal,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                )
            )
    return valence.avg, arousal.avg
