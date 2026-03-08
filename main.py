import polars as pl
import torch
import torch.optim as optim

import wandb
from cnn.cnn import CNN
from cnn.loss import loss
from data.process_data import (
    create_data_df,
    generate_test_train_split,
    generate_time_series_for_one_result,
)
from utils.arg_parser import parser
from utils.config import update_config

if __name__ == "__main__":
    args = parser.parse_args()
    CONFIG = update_config(args)

    df = create_data_df(CONFIG["dataset_config"]["path"], CONFIG["dataset"])
    train_df, val_df, test_df = generate_test_train_split(
        df, CONFIG["split"][0], CONFIG["split"][1], CONFIG["split"][2]
    )

    model = CNN(
        time_interval=CONFIG["time_interval"],
        num_channels=CONFIG["dataset_config"]["num_channels"],
        num_classes=len(CONFIG["dataset_config"]["good_classes"]),
    )
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # Train model
    if CONFIG["use_wandb"]:
        wandb.login()
        run = wandb.init(
            project=CONFIG["project_name"], config=CONFIG, name=CONFIG["run_name"]
        )
    else:
        run = None

    for epoch in range(CONFIG["epochs"]):
        if run:
            run.log({"epoch": epoch})

        training_data: list[tuple[pl.DataFrame, int]] = []
        for _ in range(CONFIG["train_samples"]):
            training_data.append(
                generate_time_series_for_one_result(
                    train_df,
                    CONFIG["time_interval"],
                    CONFIG["dataset_config"]["good_classes"],
                    CONFIG["dataset_config"]["num_channels"],
                    CONFIG["dataset_config"]["class_to_idx"],
                )
            )

        train_batches: list[list[tuple]] = [
            training_data[i : i + CONFIG["batch_size"]]
            for i in range(0, len(training_data), CONFIG["batch_size"])
        ]

        cum_loss = 0
        correct = 0
        total = 0

        for batch in train_batches:
            batch_labels = torch.tensor([result[1] for result in batch])
            batch_inputs = torch.stack(
                [torch.FloatTensor(result[0].to_numpy()).T for result in batch]
            )

            optimizer.zero_grad()

            outputs = model(batch_inputs)
            loss_value = loss(outputs, batch_labels)
            loss_value.backward()
            optimizer.step()
            print(f"Train loss: {loss_value.item()}")
            if run:
                run.log({"train_loss": loss_value.item()})
            cum_loss += loss_value.item()

            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        avg_train_loss = cum_loss / len(train_batches)
        train_accuracy = correct / total
        if run:
            run.log(
                {
                    "train_loss_epoch": avg_train_loss,
                    "train_accuracy_epoch": train_accuracy,
                }
            )
        # Finished training part of epoch

        val_data: list[tuple[pl.DataFrame, int]] = []
        for _ in range(CONFIG["val_samples"]):
            val_data.append(
                generate_time_series_for_one_result(
                    val_df,
                    CONFIG["time_interval"],
                    CONFIG["dataset_config"]["good_classes"],
                    CONFIG["dataset_config"]["num_channels"],
                    CONFIG["dataset_config"]["class_to_idx"],
                )
            )

        val_batches: list[list[tuple]] = [
            val_data[i : i + CONFIG["batch_size"]]
            for i in range(0, len(val_data), CONFIG["batch_size"])
        ]

        cum_loss = 0
        correct = 0
        total = 0

        for val_batch in val_batches:
            with torch.no_grad():
                val_batch_labels = torch.tensor([result[1] for result in val_batch])
                val_batch_inputs = torch.stack(
                    [torch.FloatTensor(result[0].to_numpy()).T for result in val_batch]
                )

                val_outputs = model(val_batch_inputs)
                val_loss = loss(val_outputs, val_batch_labels)
                cum_loss += val_loss.item()
                if run:
                    run.log({"val_loss": val_loss.item()})

                # Calculate accuracy
                _, predicted = torch.max(val_outputs.data, 1)
                total += val_batch_labels.size(0)
                correct += (predicted == val_batch_labels).sum().item()

        avg_val_loss = cum_loss / len(val_batches)
        val_accuracy = correct / total
        print(
            f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
        )
        if run:
            run.log(
                {"val_loss_epoch": avg_val_loss, "val_accuracy_epoch": val_accuracy}
            )

    if run:
        run.finish()
