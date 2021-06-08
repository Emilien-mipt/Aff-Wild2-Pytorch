import logging
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import AffWildDataset
from loss import CCCLoss
from models.cnn_encoder import CNNEncoder
from models.rnn_decoder import RNNDecoder
from train import train_one_epoch, val_one_epoch
from transforms import get_transforms
from utils.chunk_creator import ChunkCreator
from utils.utils import seed_torch


def run_trainer(cfg):
    # Create directory for saving logs and weights
    print("Creating dir for saving weights and tensorboard logs")
    os.makedirs("weights")
    os.makedirs("logs")
    print("Directories have been created!")

    # Define logger to save train logs
    logger = logging.getLogger(__name__)
    # Write to tensorboard
    tb = SummaryWriter(os.path.join("logs"))

    # Set seed
    seed = cfg.train_params.seed
    seed_torch(seed=seed)

    # Define paths
    data_path = hydra.utils.to_absolute_path(cfg.dataset.data_path)
    label_path = hydra.utils.to_absolute_path(cfg.dataset.VA_label_path)

    train_data_path = os.path.join(data_path, "Train_processed")
    val_data_path = os.path.join(data_path, "Val_processed")

    train_label_path = os.path.join(label_path, "Train_Set")
    val_label_path = os.path.join(label_path, "Validation_Set")

    # Sequence length
    seq_len = cfg.dataset.seq_len
    logger.info(f"Sequence length: {seq_len}")

    # Set device
    device = torch.device(f"cuda:{cfg.train_params.gpu_id}")
    logger.info(f"Device: {cfg.train_params.gpu_id}")

    # Train params
    batch_size = cfg.train_params.batch_size
    logger.info(f"Batch size: {batch_size}")
    num_workers = cfg.train_params.num_workers
    # Mean and std from ImageNet
    mean = np.array(cfg.dataset.mean)
    std = np.array(cfg.dataset.std)

    # Create train dataset and dataloader
    train_data_chunks = ChunkCreator(path_data=train_data_path, path_label=train_label_path, seq_len=seq_len)
    train_data_chunks.form_result_list()
    train_image_paths = train_data_chunks.result_image_paths
    train_labels = train_data_chunks.result_labels

    train_dataset = AffWildDataset(
        image_paths_list=train_image_paths,
        labels_list=train_labels,
        transform=get_transforms(mode="train", mean=mean, std=std),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # Create val dataset and dataloader
    val_data_chunks = ChunkCreator(path_data=val_data_path, path_label=val_label_path, seq_len=seq_len)
    val_data_chunks.form_result_list()
    val_image_paths = val_data_chunks.result_image_paths
    val_labels = val_data_chunks.result_labels

    val_dataset = AffWildDataset(
        image_paths_list=val_image_paths,
        labels_list=val_labels,
        transform=get_transforms(mode="valid", mean=mean, std=std),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # Model params
    # Encoder params
    fc_hidden1 = cfg.encoder_params.fc_hidden1
    fc_hidden2 = cfg.encoder_params.fc_hidden2
    cnn_drop_out = cfg.encoder_params.drop_out
    embedding_dim = cfg.encoder_params.embedding_dim

    # Decoder params
    h_rnn_layers = cfg.decoder_params.h_rnn_layers  # Number of hidden layers
    h_rnn_nodes = cfg.decoder_params.rnn_nodes  # Number of nodes in the hidden layers
    fc_dim = cfg.decoder_params.fc_dim
    rnn_drop_out = cfg.decoder_params.drop_out
    num_outputs = cfg.decoder_params.num_outputs

    # Create model
    # Define CNN encoder
    cnn_encoder = CNNEncoder(
        fc_hidden1=fc_hidden1, fc_hidden2=fc_hidden2, drop_p=cnn_drop_out, cnn_embed_dim=embedding_dim
    ).to(device)
    # Define RNN decoder
    rnn_decoder = RNNDecoder(
        cnn_embed_dim=embedding_dim,
        h_rnn_layers=h_rnn_layers,
        h_rnn=h_rnn_nodes,
        h_fc_dim=fc_dim,
        drop_p=rnn_drop_out,
        num_outputs=num_outputs,
    ).to(device)

    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())

    model = [cnn_encoder, rnn_decoder]

    criterion = CCCLoss()
    optimizer = Adam(params=crnn_params, lr=cfg.optimizer.lr)

    # start training
    logger.info("Start training...")
    for epoch in range(cfg.train_params.n_epochs):
        start_time = time.time()
        avg_train_loss = train_one_epoch(epoch, model, device, train_loader, criterion, optimizer)
        print("Validating...")
        avg_val_valence, avg_val_arousal = val_one_epoch(val_loader, model, criterion, device)
        elapsed = time.time() - start_time

        cur_lr = optimizer.param_groups[0]["lr"]

        logger.info(f"Current learning rate: {cur_lr}")

        tb.add_scalar("Learning rate", cur_lr, epoch + 1)
        tb.add_scalar("Train Loss", avg_train_loss, epoch + 1)
        tb.add_scalar("Val Valence", avg_val_valence, epoch + 1)
        tb.add_scalar("Val Arousal", avg_val_arousal, epoch + 1)

        logger.info(f"Epoch {epoch + 1} - Avg_train_loss: {avg_train_loss:.4f} time: {elapsed:.0f}s")
        logger.info(f"Epoch: {epoch + 1} - Val_valence: {avg_val_valence:.4f} - Val_arousal: {avg_val_arousal:.4f} ")

        # Save weights
        print("Saving weights...")
        torch.save(
            cnn_encoder.state_dict(), os.path.join("weights", f"best_cnn_encoder_{epoch+1}.pth")
        )  # save spatial_encoder
        torch.save(
            rnn_decoder.state_dict(), os.path.join("weights", f"best_rnn_decoder_{epoch+1}.pth")
        )  # save motion_encoder
    tb.close()


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    run_trainer(cfg)


if __name__ == "__main__":
    run()
