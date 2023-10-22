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

    def __init__(self, in_dims, out_dims):
        """
        in_dims: input feature dimensions
        out_dims: dimensions for queries and keys
        out_dims: dimensions for values
        """
        self.in_dims = in_dims
        self.out_dims = out_dims
        super(SimpleAttentionModule, self).__init__()

        # Linear layers to project input to queries, keys and values
        self.lin_q = nn.Linear(in_dims, out_dims)
        self.lin_k = nn.Linear(in_dims, out_dims)
        self.lin_v = nn.Linear(in_dims, out_dims)

        # Store sqrt(out_dims) for normalization
        self.sqrtout_dims = math.sqrt(out_dims)

    def forward(self, x, return_attention_weights=False):
        """
        x: input features (batch_size, in_dims)
        return_attention_weights: optionally return attention weights
        Returns:
          y: weighted sum of values (batch_size, out_dims)
          attn: attention weights (optional, only if return_attention_weights=True)
        """
        B = x.shape[0] # batch_size

        # Project input to queries, keys and values
        q = self.lin_q(x)  # B x out_dims
        k = self.lin_k(x)  # B x out_dims
        v = self.lin_v(x)  # B x out_dims

        # Calculate dot product similarity
        # e = q @ k.T / self.sqrtout_dims # -> Wrong, it relates elements between batch which is inaccurate in our problem

        e = (
            torch.bmm(q.view(B, self.out_dims, 1), k.view(B, 1, self.out_dims))
            / self.sqrtout_dims
        )

        # Get attention weights
        attn = torch.softmax(e, dim=2)

        # Weighted sum of values
        y = torch.bmm(attn, v.view(B, self.out_dims, 1)).squeeze(dim=-1) # remember the dim, special case when batch_size is 1 which gets squeezed together with the last dim

        if return_attention_weights:
            return y, attn

        return y


class Predictor(nn.Module):
    def __init__(self, in_dims, out_dims, use_skip_connection=False):
        super(Predictor, self).__init__()
        self.use_skip_connection = use_skip_connection
        self.attn = nn.Sequential(
            nn.BatchNorm1d(in_dims),
            SimpleAttentionModule(in_dims, in_dims * 4),
        )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(4 * in_dims),
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
