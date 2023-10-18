import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from os import path
import sys
import math

sys.path.append("models")


def get_backbone(
    backbone_model_name, num_class, normal_channel, backbone_pretrained_path=None
):
    model_class = importlib.import_module(backbone_model_name)
    backbone_model = model_class.get_model(num_class, normal_channel)
    if backbone_pretrained_path and path.isfile(backbone_pretrained_path):
        checkpoint = torch.load(backbone_pretrained_path)
        backbone_model.load_state_dict(checkpoint["model_state_dict"])
        print("Successfully loaded pretrained backbone model")

    return backbone_model


class SimpleAttentionModule(nn.Module):
    """
    Simple attention module implementation. Projects inputs to queries, keys and values using
    linear layers. Calculates dot product attention weights and returns weighted sum of values.
    """

    def __init__(self, DX, DQ, DV):
        """
        DX: input feature dimensions
        DQ: dimensions for queries and keys
        DV: dimensions for values
        """
        super(SimpleAttentionModule, self).__init__()

        # Linear layers to project input to queries, keys and values
        self.lin_q = nn.Linear(DX, DQ)
        self.lin_k = nn.Linear(DX, DQ)
        self.lin_v = nn.Linear(DX, DV)

        # Store sqrt(DQ) for normalization
        self.sqrtDQ = math.sqrt(DQ)

    def forward(self, x, return_attention_weights=False):
        """
        x: input features (batch_size, DX)
        return_attention_weights: optionally return attention weights
        Returns:
          y: weighted sum of values (batch_size, DV)
          attn: attention weights (optional, only if return_attention_weights=True)
        """

        # Project input to queries, keys and values
        q = self.lin_q(x)
        k = self.lin_k(x)
        v = self.lin_v(x)

        # Calculate dot product similarity
        e = q @ k.T / self.sqrtDQ

        # Get attention weights
        attn = torch.softmax(e, dim=1)

        # Weighted sum of values
        y = attn @ v

        if return_attention_weights:
            return y, attn

        return y


class Predictor(nn.Module):
    def __init__(self, in_dims, out_dims, use_skip_connection=False):
        super(Predictor, self).__init__()
        self.use_skip_connection = use_skip_connection
        self.attn = nn.Sequential(
            nn.BatchNorm1d(in_dims),
            SimpleAttentionModule(in_dims, in_dims * 4, in_dims)
        )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(in_dims),
            nn.Dropout(0.5),
            #
            nn.Linear(in_dims, 4 * in_dims),
            nn.BatchNorm1d(4 * in_dims),
            nn.ReLU(),
            nn.Dropout(0.5),
            #
            nn.Linear(4 * in_dims, in_dims),
            nn.BatchNorm1d(in_dims),
            nn.ReLU(),
            nn.Dropout(0.5),
            #
            nn.Linear(in_dims, out_dims),
        )

    def forward(self, x):
        main_branch = self.attn(x)
        if self.use_skip_connection:
            return self.fc(main_branch + x)
        else:
            return self.fc(main_branch)


class get_model(nn.Module):
    def __init__(
        self,
        backbone_model_name: str,
        backbone_pretrained_path: str = None,
        backbone_frozen: bool = True,
        backbone_outdims: int = 1024,
        num_class: int = 42,
        normal_channel: bool = True,
        n_out_dims: int = 10,
        use_skip_connection: bool = False,
    ):
        super(get_model, self).__init__()
        self.encoder = get_backbone(
            backbone_model_name,
            num_class,
            normal_channel,
            backbone_pretrained_path,
        )
        if backbone_frozen:
            self.encoder = self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.mlp = Predictor(backbone_outdims, n_out_dims, use_skip_connection)

    def forward(self, xyz):
        x = self.encoder(xyz.transpose(2, 1), encode_only=True)
        outputs = self.mlp(x)
        return outputs


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, include_mae=False):
        mse_loss = F.mse_loss(pred, target)

        if not include_mae:
            return mse_loss

        mae_loss = F.l1_loss(pred, target)
        return mse_loss, mae_loss
