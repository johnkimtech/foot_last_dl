# Import necessary libraries and modules
import os
import torch
import logging
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from data_utils.FootDataLoader import FootDataLoader

# from models.pointnet2_regression import get_model, get_loss
import importlib
import shutil
import argparse

import matplotlib.pyplot as plt
import csv


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Your program description here.")

    # Define command line arguments
    parser.add_argument("--num_points", type=int, default=6000)
    parser.add_argument("--dataset_dir", type=str, default="data/3D_30/txt/Foot")
    parser.add_argument("--model", type=str)
    parser.add_argument("--exp_name", type=str, default="model_mlp")
    parser.add_argument("--backbone_model", type=str, default="pointnet2_cls_ssg")
    parser.add_argument(
        "--backbone_ckpt",
        type=str,
        default="log/classification/pointnet2_cls_ssg_nonnormal_42c_allleft_new/checkpoints/best_model.pth",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        default=False,
        help="Train pretrained weights of encoder",
    )
    parser.add_argument("--backbone_outdims", type=int, default=256)
    parser.add_argument("--out_features", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="#epochs to log and graph training history",
    )
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--use_normals",
        action="store_true",
        default=False,
        help="use normals input for point cloud data",
    )
    parser.add_argument(
        "--use_lr_scheduler",
        action="store_true",
        default=False,
        help="use lr scheduler",
    )
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

    # Copy necessary files to the experiment directory
    shutil.copy("./models/%s.py" % args.model, str(EXP_DIR))
    shutil.copy(f"./models/{args.backbone_model}.py", str(EXP_DIR))
    shutil.copy("models/pointnet2_utils.py", str(EXP_DIR))
    shutil.copy("./train_regression.py", str(EXP_DIR))

    # Set up logging
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/%s.txt" % (str(LOG_DIR), args.exp_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log hyperparameters
    log_string("HYPERPARAMETERS ...")
    log_string(args)

    # Create train and test datasets
    train_dataset = FootDataLoader(
        args.dataset_dir, args.num_points, args.use_normals, "train"
    )
    test_dataset = FootDataLoader(
        args.dataset_dir, args.num_points, args.use_normals, "test"
    )
    log_string(f"DATASET: {len(train_dataset)=}; {len(test_dataset)=}")

    # Create data loaders
    trainDataLoader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )
    testDataLoader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10
    )

    # Create the regression model
    model = importlib.import_module(f"models.{args.model}")

    regressor = model.get_model(
        backbone_model_name=args.backbone_model,
        backbone_pretrained_path=args.backbone_ckpt,
        backbone_frozen=not args.finetune,
        backbone_outdims=args.backbone_outdims,
        num_class=42,
        normal_channel=args.use_normals,
        n_out_dims=args.out_features,
        use_skip_connection=args.use_skip_connection,
    )

    try:
        # Load pretrained model if available
        checkpoint = torch.load(str(CKPT_DIR / "best_model.pth"))
        start_epoch = checkpoint["epoch"]
        regressor.load_state_dict(checkpoint["model_state_dict"])
        best_val_error = checkpoint["val_error"]
        log_string("Use pretrained model")
    except Exception as err:
        # Start training from scratch if no pretrained model is available
        log_string("No existing model, starting training from scratch...")
        best_val_error = np.inf
        start_epoch = 0

    # Define loss function, optimizer, and learning rate scheduler
    criterion = model.get_loss()
    model_params = (
        regressor.parameters() if args.finetune else regressor.mlp.parameters()
    )
    if args.optim == "adam":
        optimizer = torch.optim.Adam(
            # regressor.parameters(),
            model_params,
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9)

    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    global_epoch = start_epoch
    global_step = 0

    # Move model to the specified device
    regressor = regressor.to(args.device)
    train_losses = []

    # Create lists to store training and validation errors for each epoch
    train_errors = []
    val_errors = []

    # Training loop
    for epoch in range(start_epoch, args.n_epochs):
        regressor = regressor.train()
        log_string("Epoch %d (%d/%s):" % (global_epoch + 1, epoch + 1, args.n_epochs))

        # Train
        if args.use_lr_scheduler:
            scheduler.step()
        for batch_id, (points, target, scale) in tqdm(
            enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9
        ):
            optimizer.zero_grad()

            points = torch.tensor(points, dtype=torch.float32, device=args.device)
            target = torch.tensor(target, dtype=torch.float32, device=args.device)

            target = target[:, : args.out_features]

            pred = regressor(points)
            loss = criterion(pred, target)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            global_step += 1
        train_mse = np.mean(train_losses)

        # Validation
        val_mse_losses = []
        val_mae_losses = []
        with torch.no_grad():
            regressor = regressor.eval()
            for batch_id, (points, target, scale) in tqdm(
                enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9
            ):
                points = torch.tensor(points, dtype=torch.float32, device=args.device)
                target = torch.tensor(target, dtype=torch.float32, device=args.device)
                target = target[:, : args.out_features]

                pred = regressor(points)
                mse, mae = criterion(pred, target, include_mae=True)
                val_mse_losses.append(mse.item())
                val_mae_losses.append(mae.item())
            val_mse = np.mean(val_mse_losses)
            val_mae = np.mean(val_mae_losses)

        # Append errors to the lists
        train_errors.append(train_mse)
        val_errors.append(val_mse)

        # Log training and validation errors
        log_string(
            "Train MSE: %f, Val MSE: %f, Val MAE: %f" % (train_mse, val_mse, val_mae)
        )

        # Save the model if validation error improves
        if val_mse < best_val_error:
            best_val_error = val_mse
            logger.info("Save model...")
            savepath = str(CKPT_DIR) + "/best_model.pth"
            log_string("Saving at %s" % savepath)
            state = {
                "epoch": epoch + 1,
                "train_error": train_mse,
                "val_error": val_mse,
                "model_state_dict": regressor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model": args.model,
                "num_points": args.num_points,
                "backbone_model": args.backbone_model,
                "backbone_outdims": args.backbone_outdims,
                "out_features": args.out_features,
                "use_skip_connection": args.use_skip_connection,
                "use_normals": args.use_normals,
            }
            torch.save(state, savepath)

        log_string("_" * 15)
        if (global_epoch + 1) % args.log_interval == 0:
            # Save training history as a CSV file
            save_history_csv(start_epoch, train_errors, val_errors, LOG_DIR)
            # Plot and save the training history graph
            plot_history(start_epoch, train_errors, val_errors, LOG_DIR)
        global_epoch += 1

    ## END OF TRAINING logging and graphing

    # Save training history as a CSV file
    save_history_csv(start_epoch, train_errors, val_errors, LOG_DIR)
    # Plot and save the training history graph
    plot_history(start_epoch, train_errors, val_errors, LOG_DIR)
    log_string("End of training...")


def save_history_csv(start_epoch, train_errors, val_errors, save_dir):
    history_file_path = os.path.join(save_dir, "training_history.csv")
    with open(history_file_path, mode="w") as history_file:
        history_writer = csv.writer(history_file)
        history_writer.writerow(["Epoch", "Train Error", "Validation Error"])
        for epoch, (train_mse, val_mse) in enumerate(
            zip(train_errors, val_errors), start=start_epoch
        ):
            history_writer.writerow([epoch + 1, train_mse, val_mse])


def plot_history(start_epoch, train_errors, val_errors, save_dir):
    plt.figure()
    plt.plot(
        range(start_epoch, start_epoch + len(train_errors)),
        train_errors,
        label="Train Error",
    )
    plt.plot(
        range(start_epoch, start_epoch + len(val_errors)),
        val_errors,
        label="Validation Error",
    )
    plt.xlabel("Epoch")
    plt.ylabel(f"Error (best={min(val_errors):.4f})")
    plt.legend()
    plt.title("Training History")
    plt.savefig(os.path.join(save_dir, "training_history.png"))


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
