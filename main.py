import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
from typing import Any

from cnn.cnn import CNN
from cnn.loss import loss
from data.distributed_data_parallel import DataDDP
from data.process_data import (
    create_data_df,
    generate_test_train_split,
    EMGTimeSeriesDataset,
)
from utils.arg_parser import parser
from utils.config import update_config

if __name__ == "__main__":
    args = parser.parse_args()
    CONFIG: dict[str, Any] = update_config(args)

    # Initialize DDP
    data_ddp = DataDDP()
    ddp = data_ddp.init_ddp()
    print(data_ddp.device)

    df = create_data_df(CONFIG["dataset_config"]["path"], CONFIG["dataset"])
    print("finished making first df")
    train_df, val_df, test_df = generate_test_train_split(
        df, CONFIG["split"][0], CONFIG["split"][1], CONFIG["split"][2]
    )

    model = CNN(
        time_interval=CONFIG["time_interval"],
        num_channels=CONFIG["dataset_config"]["num_channels"],
        num_classes=len(CONFIG["dataset_config"]["good_classes"]),
    )
    model = model.to(data_ddp.device)

    # Wrap model with DDP if distributed training is enabled
    if ddp:
        model = DDP(model, device_ids=[data_ddp.ddp_local_rank])

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # Train model - only initialize wandb on master process
    if CONFIG["use_wandb"] and data_ddp.master_process:
        wandb.login()
        run = wandb.init(
            project=CONFIG["project_name"], config=CONFIG, name=CONFIG["run_name"]
        )
    else:
        run = None

    # Create PyTorch datasets
    print("making train dataset")
    train_dataset = EMGTimeSeriesDataset(
        df=train_df,
        time_interval=CONFIG["time_interval"],
        good_classes=CONFIG["dataset_config"]["good_classes"],
        num_channels=CONFIG["dataset_config"]["num_channels"],
        class_to_idx=CONFIG["dataset_config"]["class_to_idx"],
        num_samples=CONFIG["train_samples"],
    )

    print("making val dataset")
    val_dataset = EMGTimeSeriesDataset(
        df=val_df,
        time_interval=CONFIG["time_interval"],
        good_classes=CONFIG["dataset_config"]["good_classes"],
        num_channels=CONFIG["dataset_config"]["num_channels"],
        class_to_idx=CONFIG["dataset_config"]["class_to_idx"],
        num_samples=CONFIG["val_samples"],
    )

    # Create samplers for DDP
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp else None

    # Create DataLoaders with multiple workers for efficiency
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        sampler=train_sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=True if data_ddp.device != "cpu" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=True if data_ddp.device != "cpu" else False,
    )

    print("finished making dataloaders")
    for epoch in range(CONFIG["epochs"]):
        if run:
            run.log({"epoch": epoch})

        # Set epoch for distributed sampler
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Training loop
        model.train()
        cum_loss = 0
        correct = 0
        total = 0

        for batch_idx, (batch_inputs, batch_labels) in enumerate(train_loader):
            batch_inputs = batch_inputs.to(data_ddp.device)
            batch_labels = batch_labels.to(data_ddp.device)

            optimizer.zero_grad()

            outputs = model(batch_inputs)
            loss_value = loss(outputs, batch_labels)
            loss_value.backward()
            optimizer.step()

            if data_ddp.master_process:
                print(f"Train loss: {loss_value.item()}")
            if run:
                run.log({"train_loss": loss_value.item()})
            cum_loss += loss_value.item()

            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        avg_train_loss = cum_loss / len(train_loader)
        train_accuracy = correct / total
        if run and data_ddp.master_process:
            run.log(
                {
                    "train_loss_epoch": avg_train_loss,
                    "train_accuracy_epoch": train_accuracy,
                }
            )

        # Validation loop
        model.eval()
        cum_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_batch_inputs, val_batch_labels in val_loader:
                val_batch_inputs = val_batch_inputs.to(data_ddp.device)
                val_batch_labels = val_batch_labels.to(data_ddp.device)

                val_outputs = model(val_batch_inputs)
                val_loss = loss(val_outputs, val_batch_labels)
                cum_loss += val_loss.item()
                if run:
                    run.log({"val_loss": val_loss.item()})

                # Calculate accuracy
                _, predicted = torch.max(val_outputs.data, 1)
                total += val_batch_labels.size(0)
                correct += (predicted == val_batch_labels).sum().item()

        avg_val_loss = cum_loss / len(val_loader)
        val_accuracy = correct / total
        if data_ddp.master_process:
            print(
                f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
            )
        if run and data_ddp.master_process:
            run.log(
                {"val_loss_epoch": avg_val_loss, "val_accuracy_epoch": val_accuracy}
            )

    if run and data_ddp.master_process:
        run.finish()

    # Cleanup DDP
    data_ddp.cleanup_ddp(ddp)
