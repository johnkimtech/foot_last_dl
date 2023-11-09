import torch
import sys
import time
import logging
import argparse
import importlib
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from tabulate import tabulate
from torch.utils.data import DataLoader
from data_utils.FootDataset import FootDataset
from data_utils.helpers import seed_everything_deterministic



np.set_printoptions(precision=1, suppress=True)

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
    parser.add_argument("--seed", type=int, default=None)
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

    if args.seed is not None:
        seed_everything_deterministic(args.seed)

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
    i = 0
    foot_ids = test_dataset.get_foot_ids()
    targets = []
    preds = []
    errors = []
    tic = time.perf_counter()
    with torch.no_grad():
        # pbar = enumerate(testDataLoader, 0)
        # if not args.print_pred:
        pbar = tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9)
        for batch_id, (points, tgt, scale) in pbar:
            points = torch.tensor(points, dtype=torch.float32, device=args.device)
            tgt = torch.tensor(tgt, dtype=torch.float32, device=args.device)
            tgt = tgt[:, : checkpoint["out_features"]]
            scale = scale.view(scale.shape[0], 1).to(args.device)

            prd = regressor(points)
            tgt *= scale
            prd *= scale
            err = (tgt - prd).abs().mean(dim=1)

            tgt = tgt.cpu().detach().numpy().tolist()
            prd = prd.cpu().detach().numpy().tolist()
            err = err.cpu().detach().numpy().tolist()

            targets += np.round(tgt, decimals=2).tolist()
            preds += np.round(prd, decimals=2).tolist()
            errors += np.round(err, decimals=2).tolist()

    if args.print_pred:
        results_df = pd.DataFrame({
            "No.": foot_ids,
            "TARGET": targets,
            "PRED": preds,
            "MAERR": errors
        })
        results_df.sort_values(by='MAERR', inplace=True, ascending=True)
        print(tabulate(results_df, headers="keys", tablefmt="simple_grid", showindex=False, floatfmt=".2f"))


    mean_error = np.mean(errors)

    print(
        f"Mean Absolute Error: {mean_error.item():.2f} (mm)"
    )
    toc = time.perf_counter()
    total_time = toc - tic
    print(
        f"Finished in {total_time:0.2f} seconds in total, each sample takes {total_time/len(test_dataset):.3f} sec"
    )


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
