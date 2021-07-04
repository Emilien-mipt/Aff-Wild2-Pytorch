import torch
import torch.nn as nn
from torchvision import models


class CNNEncoder(nn.Module):
    """Encoder based on CNN."""

    def __init__(self, fc_hidden1=1500, drop_p=0.5, pretrain=False):
        """Load the pretrained ResNet and replace top fc layer."""
        super().__init__()

        self.fc_hidden1 = fc_hidden1
        self.drop_p = drop_p

        densenet = models.densenet121(pretrained=True)
        modules = list(densenet.children())[:-1]  # delete the last fc layer.
        self.densenet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(50176, fc_hidden1)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(p=self.drop_p)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.densenet(x_3d[:, t, :, :, :])  # DenseNet
                x = x.view(x.size(0), -1)  # flatten output of conv
                # print("TEST: ", x.shape)

            # FC layers
            x = self.fc1(x)
            x = self.act1(x)
            x = self.dropout(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq
