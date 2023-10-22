import torch
import sys
import time
import logging
import argparse
import importlib
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from data_utils.FootDataset import FootDataset

np.set_printoptions(precision=1, suppress=True)

torch.manual_seed(42)
np.random.seed(42)

# If using GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Your program description here.")

    # Define command line arguments
    parser.add_argument("--dataset_dir", type=str, default="data/3D_All_Foot/oct12")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--print_config",
        action="store_true",
        default=False,
        help="print model configs including architecture and hyperparams",
    )
    parser.add_argument(
        "--print_pred",
        action="store_true",
        default=False,
        help="print predictions",
    )

    # Parse arguments from the command line
    return parser.parse_args()


# Main function
def main():
    # Define a function to log strings to a file and print them
    def log_string(str):
        logger.info(str)
        print(str)

    # Parse command line arguments
    args = parse_args()

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
        log_string("Using pretrained model")
    except Exception as err:
        # Start training from scratch if no pretrained model is available
        log_string(f"No existing model: {err}")
        return

    sys.path.append(str(EXP_DIR))
    libpath = f"log.regression.{args.exp_name}"
    model = importlib.import_module(checkpoint["model"], libpath)

    # Log hyperparameters
    if args.print_config:
        log_string("HYPERPARAMETERS ...")
        log_string(args)

    # Create train and test datasets
    test_dataset = FootDataset(
        args.dataset_dir, checkpoint["num_points"], checkpoint["use_normals"], "test"
    )
    log_string(f"DATASET: {len(test_dataset)=}")

    # Create data loaders
    testDataLoader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10
    )

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

    # Define loss function, optimizer, and learning rate scheduler
    criterion = model.get_loss()

    # Move model to the specified device
    regressor = regressor.to(args.device).eval()

    avg_mse, avg_mae = 0.0, 0.0
    real_mse, real_mae = 0.0, 0.0
    # Training loop
    len_test = len(test_dataset)
    tic = time.perf_counter()
    with torch.no_grad():
        pbar = enumerate(testDataLoader, 0)
        if not args.print_pred:
            pbar = tqdm(pbar, total=len(testDataLoader), smoothing=0.9)
        for batch_id, (points, target, scale) in pbar:
            points = torch.tensor(points, dtype=torch.float32, device=args.device)
            target = torch.tensor(target, dtype=torch.float32, device=args.device)
            target = target[:, : checkpoint["out_features"]]
            scale = scale.numpy().reshape(scale.shape[0], 1)

            pred = regressor(points)
            mse, mae = criterion(pred, target, include_mae=True)
            r_mse, r_mae = criterion(
                pred.cpu() * scale, target.cpu() * scale, include_mae=True
            )
            target = target.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()

            if args.print_pred:
                for t, p in zip(target * scale, pred * scale):
                    print(f"{t} vs {p}")

            batch_size = points.shape[0]
            avg_mse += mse * batch_size / len_test
            avg_mae += mae * batch_size / len_test
            real_mse += r_mse * batch_size / len_test
            real_mae += r_mae * batch_size / len_test

    print(
        f"NORM: Mean Squared Error: {avg_mse.item():.4f}, Mean Absolute Error: {avg_mae.item():.4f}"
    )
    print(
        f"REAL: Mean Squared Error: {real_mse.item():.2f}, Mean Absolute Error: {real_mae.item():.2f}"
    )
    toc = time.perf_counter()
    total_time = toc - tic
    print(
        f"Finished in {total_time:0.2f} seconds in total, each sample takes {total_time/len(test_dataset):.3f} sec"
    )


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
