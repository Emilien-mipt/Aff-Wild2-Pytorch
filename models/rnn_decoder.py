import torch.nn as nn


class RNNDecoder(nn.Module):
    """Decoder based on RNN."""

    def __init__(self, cnn_embed_dim=1500, h_rnn_layers=2, h_rnn=128, num_outputs=2):
        super().__init__()
        self.RNN_input_size = cnn_embed_dim
        self.h_RNN_layers = h_rnn_layers  # RNN hidden layers
        self.h_RNN = h_rnn  # RNN hidden nodes
        self.num_outputs = num_outputs  # First value stands for Valence, second - for Arousal
        # Rnn part
        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=self.h_RNN_layers,
            batch_first=True,  # input & output will have batch size as 1st dimension.
            # e.g. (batch, time_step, input_size)
        )
        # Output fc layer
        self.fc1 = nn.Linear(self.h_RNN, self.num_outputs)

    def forward(self, x_RNN):
        # RNN layer
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        # FC layer
        x = self.fc1(RNN_out)

        return x
