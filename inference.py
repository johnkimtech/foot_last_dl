import torch
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from data_utils.FootDataLoader import FootDataLoader
import importlib
import sys
import argparse
import matplotlib.pyplot as plt
import csv
import time


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Your program description here.")

    # Define command line arguments
    parser.add_argument("--num_points", type=int, default=6000)
    parser.add_argument("--model", type=str)
    parser.add_argument("--exp_name", type=str, default="model_mlp")
    parser.add_argument("--backbone_model", type=str, default="pointnet2_cls_ssg")
    parser.add_argument("--backbone_outdims", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_normals", type=bool, default=False)
    parser.add_argument("--out_features", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--use_skip_connection",
        action="store_true",
        default=False,
        help="use skip_connection for mlp predictor",
    )

    # Parse arguments from the command line
    return parser.parse_args()

# Main function
def main():
    # Define a function to log strings to a file and print them
    def log_string(str):
        # logger.info(str)
        print(str)

    # Parse command line arguments
    args = parse_args()
    EXP_DIR = Path("log/regression") / args.exp_name
    CKPT_DIR = EXP_DIR / "checkpoints"

    sys.path.append(str(EXP_DIR))
    libpath = f"log.regression.{args.exp_name}"
    model = importlib.import_module(f"{libpath}.{args.model}")

    log_string("Loading model...")
    # Create the regression model
    regressor = model.get_model(
        backbone_model_name=f"{libpath}.{args.backbone_model}",
        backbone_pretrained_path=None,
        backbone_frozen=True,
        backbone_outdims=args.backbone_outdims,
        num_class=42,
        normal_channel=args.use_normals,
        n_out_dims=args.out_features,
        use_skip_connection=args.use_skip_connection,
    )

    try:
        # Load pretrained model if available
        checkpoint = torch.load(
            str(CKPT_DIR / "best_model.pth"), map_location=torch.device(args.device)
        )
        regressor.load_state_dict(checkpoint["model_state_dict"])
        log_string("Use pretrained model")
    except Exception as err:
        # Start training from scratch if no pretrained model is available
        log_string(f"No existing model: {err}. Can't run inference without trained weights. Exitting...")
        return
    

    # Move model to the specified device
    regressor = regressor.to(args.device).eval()
    regressor.encoder.eval()

