import torch
import torch.nn as nn
from torchvision import models


class Resnet_Encoder(nn.Module):
    """Encoder based on CNN."""

    def __init__(self, fc_hidden1=1500, drop_p=0.5, pretrain=False, freeze=True):
        """Load the pretrained ResNet and replace top fc layer."""
        super().__init__()

        self.fc_hidden1 = fc_hidden1
        self.drop_p = drop_p

        resnet = models.resnet50(pretrained=pretrain)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(p=self.drop_p)
        self.freeze = freeze

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            if self.freeze:
                with torch.no_grad():
                    x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                    x = x.view(x.size(0), -1)  # flatten output of conv
            else:
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            x = self.fc1(x)
            x = self.act1(x)
            x = self.dropout(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class VGG_Encoder(nn.Module):
    """Encoder based on CNN."""

    def __init__(self, fc_hidden1=4096, drop_p=0.5, pretrain=False, freeze=True):
        """Load the pretrained ResNet and replace top fc layer."""
        super().__init__()

        self.fc_hidden1 = fc_hidden1
        self.drop_p = drop_p

        vgg = models.vgg16(pretrained=pretrain)
        modules = list(vgg.children())[:-1]  # delete the last fc layer.
        self.vgg = nn.Sequential(*modules)
        self.fc1 = nn.Linear(25088 + 136, fc_hidden1)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(p=self.drop_p)
        self.freeze = freeze

    def forward(self, x_3d, lm_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            if self.freeze:
                with torch.no_grad():
                    x = self.vgg(x_3d[:, t, :, :, :])  # VGG
                    img_features = x.view(x.size(0), -1)  # flatten output of conv
            else:
                x = self.vgg(x_3d[:, t, :, :, :])  # VGG
                img_features = x.view(x.size(0), -1)  # flatten output of conv
            # Flatten landmarks
            lm = lm_3d[:, t, :, :]
            lm_features = lm.view(lm.size(0), -1)
            # Fuse output of Conv and landmarks
            fuse_features = torch.cat((img_features, lm_features), 1)

            # FC layers
            fuse_out = self.fc1(fuse_features)
            fuse_out = self.act1(fuse_out)
            out = self.dropout(fuse_out)

            cnn_embed_seq.append(out)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq
