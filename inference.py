import sys
import csv
import time
import torch
import logging
import argparse
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from tabulate import tabulate
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_utils.FootDataLoader import FootDataLoader

foot_labels_cols = ["No.", "발 길이", "발볼 둘레", "발등 둘레", "발 뒤꿈치 둘레", "발가락 둘레"]
# foot_labels_cols = ['No', 'Length', 'Ball Circumference', 'Instep Circumference', 'Heel Circumference', 'Toe Circumference']


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
    parser.add_argument("--infer_data_csv", type=str)

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
        log_string(
            f"No existing model: {err}. Can't run inference without trained weights. Exitting..."
        )
        return

    # Move model to the specified device
    regressor = regressor.to(args.device).eval()
    regressor.encoder.eval()

    # Create test datasets
    test_dataset = FootDataLoader(
        num_points=args.num_points,
        use_normals=args.use_normals,
        split="infer",
        infer_data_csv=args.infer_data_csv,
    )
    # Create data loaders
    testDataLoader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10
    )

    np.set_printoptions(precision=3, suppress=True)
    tic = time.perf_counter()

    # Set display options to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)

    results_df = pd.DataFrame(columns=foot_labels_cols)
    with torch.no_grad():
        for batch_id, (points, scale, foot_ids) in enumerate(testDataLoader, 0):
            points = torch.tensor(points, dtype=torch.float32, device=args.device)
            scale = scale.numpy().reshape(scale.shape[0], 1)

            pred = regressor(points)
            pred = pred.cpu().detach().numpy() * scale
            new_df = pd.DataFrame(np.hstack((np.array(foot_ids).reshape(-1,1), np.round(pred, 3))), columns=foot_labels_cols)
            results_df = pd.concat(
                (results_df, new_df)
            )

    toc = time.perf_counter()
    total_time = toc - tic
    results_df.reset_index(drop=True, inplace=True)
    print(tabulate(results_df, headers='keys', tablefmt='grid'))
    print(
        f"Finished in {total_time:0.2f} seconds in total, each sample takes {total_time/len(test_dataset):.3f} sec"
    )


if __name__ == "__main__":
    main()
