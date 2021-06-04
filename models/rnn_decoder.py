import torch.nn as nn
import torch.nn.functional as F


class RNNDecoder(nn.Module):
    def __init__(self, cnn_embed_dim=300, h_rnn_layers=2, h_rnn=128, h_fc_dim=32, drop_p=0.3, num_outputs=2):
        super().__init__()

        self.RNN_input_size = cnn_embed_dim
        self.h_RNN_layers = h_rnn_layers  # RNN hidden layers
        self.h_RNN = h_rnn  # RNN hidden nodes
        self.h_FC_dim = h_fc_dim
        self.drop_p = drop_p
        self.num_outputs = num_outputs  # First value stands for Valence, second - for Arousal

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=self.h_RNN_layers,
            batch_first=True,  # input & output will have batch size as 1st dimension.
            # e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_outputs)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x
