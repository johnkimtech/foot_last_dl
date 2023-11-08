import os
import sys
import time
import torch
import logging
import argparse
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from tabulate import tabulate
from torch.utils.data import DataLoader
from data_utils.FootDataset import FootDataset

torch.set_default_tensor_type(torch.FloatTensor)

default_result_headers = ["No.", "발 길이", "발볼 둘레", "발등 둘레", "발 뒤꿈치 둘레", "발가락 둘레"]

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Your program description here.")

    # Define command line arguments
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--infer_data_csv", type=str)
    parser.add_argument("--output_csv_path", type=str, default=None)
    parser.add_argument("--result_headers", type=str, default=default_result_headers)

    # Parse arguments from the command line
    return parser.parse_args()


# Main function
def inference(args):
    # Define a function to log strings to a file and print them
    def log_string(str):
        logger.info(str)
        print(str)

    # Get the current date and time for creating experiment directories
    EXP_DIR = Path("log/regression") / args.exp_name
    CKPT_DIR = EXP_DIR / "checkpoints"
    LOG_DIR = EXP_DIR / "logs"
    CKPT_DIR.mkdir(exist_ok=True, parents=True)
    LOG_DIR.mkdir(exist_ok=True, parents=True)

    # Set up logging
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/%s_test.txt" % (str(LOG_DIR), args.exp_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    try:
        # Load pretrained model if available
        checkpoint = torch.load(
            str(CKPT_DIR / "best_model.pth"), map_location=torch.device(args.device)
        )
        log_string(f"Using pretrained model: {str(CKPT_DIR)}")
    except Exception as err:
        # Start training from scratch if no pretrained model is available
        log_string(f"No existing model: {err}")
        return

    sys.path.append(str(EXP_DIR))
    libpath = f"log.regression.{args.exp_name}"
    model = importlib.import_module(checkpoint["model"], libpath)

    # Create the regression model

    regressor = model.get_model(
        backbone_model_name=checkpoint["backbone_model"],
        backbone_pretrained_path=None,
        backbone_frozen=True,
        backbone_outdims=checkpoint["backbone_outdims"],
        num_class=42,
        normal_channel=checkpoint["use_normals"],
        n_out_dims=checkpoint["out_features"],
        use_skip_connection=checkpoint["use_skip_connection"],
    )
    regressor.load_state_dict(checkpoint["model_state_dict"])

    # Move model to the specified device
    regressor = regressor.to(args.device).eval()

    # Create test datasets
    test_dataset = FootDataset(
        num_points=checkpoint["num_points"],
        use_normals=checkpoint["use_normals"],
        split="infer",
        infer_data_csv=args.infer_data_csv,
    )
    # Create data loaders
    inferDataLoader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10,
        multiprocessing_context=multiprocessing.get_context('spawn')
    )
    log_string("Running Inference...")

    tic = time.perf_counter()

    results_df = pd.DataFrame(columns=args.result_headers)
    with torch.no_grad():
        for batch_id, (points, scale, foot_ids) in tqdm(
            enumerate(inferDataLoader, 0), total=len(inferDataLoader), smoothing=0.8
        ):
            points = torch.tensor(points, dtype=torch.float32, device=args.device)
            scale = scale.numpy().reshape(scale.shape[0], 1)

            pred = regressor(points)
            pred = pred.cpu().detach().numpy() * scale
            new_df = pd.DataFrame(
                np.hstack((np.array(foot_ids).reshape(-1, 1), np.round(pred, 3))),
                columns=args.result_headers or default_result_headers,
            )
            results_df = pd.concat((results_df, new_df))

    toc = time.perf_counter()
    total_time = toc - tic
    results_df.reset_index(drop=True, inplace=True)
    print("*" * 20)
    print("INFERENCE RESULTS:")
    print(tabulate(results_df, headers="keys", tablefmt="simple_grid", showindex=False))
    print(
        f"Finished in {total_time:0.2f} seconds in total, each sample takes {total_time/len(test_dataset):.3f} sec"
    )
    if args.output_csv_path:
        results_df.to_csv(args.output_csv_path, index=False)
        print("Results are saved in:", os.path.abspath(args.output_csv_path))

    return results_df


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    inference(args)
