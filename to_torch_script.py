import hydra
import torch
from omegaconf import DictConfig

from models.cnn_encoder import CNNEncoder
from models.rnn_decoder import RNNDecoder


def convert_encoder(model, model_path, model_path_updated, size, device):
    """Save model to torch trace format."""
    # Load checkpoint
    print("Loading model...")
    chk = torch.load(model_path)
    print("Model has been loaded!")
    if "model" in chk:
        print("Loading state dict...")
        model.load_state_dict(chk["model"], strict=False)
        print("State dict has been loaded!")
    isize = size  # Input size
    sample_input = torch.rand(4, 2, 3, isize, isize)
    model.to(device)
    model.eval()
    sample_input = sample_input.to(device)
    print("Tracing the model...")
    traced_model = torch.jit.trace(model, sample_input)
    torch.jit.save(traced_model, model_path_updated)
    print("Model is ready!")
    return traced_model


def convert_decoder(model, model_path, model_path_updated, device):
    """Save model to torch trace format."""
    # Load checkpoint
    print("Loading model...")
    chk = torch.load(model_path)
    print("Model has been loaded!")
    if "model" in chk:
        print("Loading state dict...")
        model.load_state_dict(chk["model"], strict=False)
        print("State dict has been loaded!")

    sample_input = torch.rand(8, 4, 300)
    model.to(device)
    model.eval()
    sample_input = sample_input.to(device)
    print("Tracing the model...")
    traced_model = torch.jit.trace(model, sample_input)
    torch.jit.save(traced_model, model_path_updated)
    print("Model is ready!")
    return traced_model


def main(cfg):
    device = torch.device(f"cuda:{cfg.train_params.gpu_id}")

    # Encoder params
    encoder_path = hydra.utils.to_absolute_path(cfg.encoder_params.chk)
    encoder_output = hydra.utils.to_absolute_path(cfg.encoder_params.torch_script_path)
    fc_hidden1 = cfg.encoder_params.fc_hidden1
    fc_hidden2 = cfg.encoder_params.fc_hidden2
    cnn_drop_out = cfg.encoder_params.drop_out
    embedding_dim = cfg.encoder_params.embedding_dim

    cnn_encoder = CNNEncoder(
        fc_hidden1=fc_hidden1,
        fc_hidden2=fc_hidden2,
        drop_p=cnn_drop_out,
        cnn_embed_dim=embedding_dim,
        pretrain=False,
    ).to(device)

    print("Converting encoder to script format...")
    convert_encoder(cnn_encoder, encoder_path, encoder_output, 96, device)
    print("Encoder has been converted!")

    # Decoder params
    decoder_path = hydra.utils.to_absolute_path(cfg.decoder_params.chk)
    decoder_output = hydra.utils.to_absolute_path(cfg.decoder_params.torch_script_path)
    h_rnn_layers = cfg.decoder_params.h_rnn_layers  # Number of hidden layers
    h_rnn_nodes = cfg.decoder_params.rnn_nodes  # Number of nodes in the hidden layers
    fc_dim = cfg.decoder_params.fc_dim
    rnn_drop_out = cfg.decoder_params.drop_out
    num_outputs = cfg.decoder_params.num_outputs

    # Define RNN decoder
    rnn_decoder = RNNDecoder(
        cnn_embed_dim=embedding_dim,
        h_rnn_layers=h_rnn_layers,
        h_rnn=h_rnn_nodes,
        h_fc_dim=fc_dim,
        drop_p=rnn_drop_out,
        num_outputs=num_outputs,
    ).to(device)

    print("Converting decoder to script format...")
    convert_decoder(rnn_decoder, decoder_path, decoder_output, device)
    print("Decoder has been converted!")


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    run()
