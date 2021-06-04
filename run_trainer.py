import os
import time

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import AffWildDataset
from loss import CCCLoss
from models.backbone import ResCNNEncoder
from models.rnn_decoder import DecoderRNN
from train import train_one_epoch, val_one_epoch
from tranforms import get_transforms
from utils.chunk_creator import ChunkCreator
from utils.utils import asMinutes, seed_torch

SAVE_MODEL_PATH = "./logs"

TRAIN_PATH = "./data/dataset/Train_processed"
TRAIN_LABEL_PATH = "./data/annotations_VA/Train_Set"

VAL_PATH = "./data/dataset/Val_processed"
VAL_LABEL_PATH = "./data/annotations_VA/Validation_Set"

SEQ_LEN = 80
BATCH_SIZE = 128
NUM_WORKERS = 8
GPU_ID = 0
EPOCH = 100
LR = 0.001

EARLY_STOPPING = 10


def main():
    device = torch.device(f"cuda:{GPU_ID}")

    # Create train dataset and dataloader
    train_data_chunks = ChunkCreator(path_data=TRAIN_PATH, path_label=TRAIN_LABEL_PATH, seq_len=SEQ_LEN)
    train_data_chunks.form_result_list()
    train_image_paths = train_data_chunks.result_image_paths
    train_labels = train_data_chunks.result_labels

    train_dataset = AffWildDataset(
        image_paths_list=train_image_paths, labels_list=train_labels, transform=get_transforms(data="train")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Create val dataset and dataloader
    val_data_chunks = ChunkCreator(path_data=VAL_PATH, path_label=VAL_LABEL_PATH, seq_len=SEQ_LEN)
    val_data_chunks.form_result_list()
    val_image_paths = val_data_chunks.result_image_paths
    val_labels = val_data_chunks.result_labels

    val_dataset = AffWildDataset(
        image_paths_list=val_image_paths, labels_list=val_labels, transform=get_transforms(data="train")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Create model
    cnn_encoder = ResCNNEncoder().to(device)
    rnn_decoder = DecoderRNN().to(device)

    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())

    model = [cnn_encoder, rnn_decoder]

    criterion = CCCLoss()
    optimizer = Adam(params=crnn_params, lr=LR)

    best_epoch = 0
    best_valence_score = 0.0
    best_arousal_score = 0.0

    count_bad_epochs = 0  # Count epochs that don't improve the score

    # start training
    for epoch in range(EPOCH):
        start_time = time.time()
        avg_train_loss = train_one_epoch(epoch, model, device, train_loader, criterion, optimizer)
        elapsed = time.time() - start_time
        print("Time spent for one epoch: ", asMinutes(elapsed))
        print("Average train loss: ", avg_train_loss)
        print("Validating...")
        avg_val_valence, avg_val_arousal = val_one_epoch(val_loader, model, criterion, device)
        print(f"Average validation Valence: {avg_val_valence}")
        print(f"Average validation Arousal: {avg_val_arousal}")

        best_valence = False
        best_arousal = False

        # Update best score
        if avg_val_valence >= best_valence_score:
            best_valence_score = avg_val_valence
            best_valence = True

        if avg_val_arousal >= best_arousal_score:
            best_arousal_score = avg_val_arousal
            best_arousal = True

        if best_valence and best_arousal:
            torch.save(
                cnn_encoder.state_dict(), os.path.join(SAVE_MODEL_PATH, "cnn_encoder_epoch{}.pth".format(epoch + 1))
            )  # save spatial_encoder
            torch.save(
                rnn_decoder.state_dict(), os.path.join(SAVE_MODEL_PATH, "rnn_decoder_epoch{}.pth".format(epoch + 1))
            )  # save motion_encoder
            best_epoch = epoch + 1
            count_bad_epochs = 0
        else:
            count_bad_epochs += 1
        print(count_bad_epochs)
        # LOGGER.info(f"Number of bad epochs {count_bad_epochs}")
        # Early stopping
        if count_bad_epochs > EARLY_STOPPING:
            # LOGGER.info(f"Stop the training, since the score has not improved for {CFG.early_stopping} epochs!")
            break
    print(
        f"AFTER TRAINING: Epoch {best_epoch}: Best Valence: {best_valence_score:.4f} - \
                    Best Arousal: {best_arousal_score:.4f}"
    )


if __name__ == "__main__":
    main()
