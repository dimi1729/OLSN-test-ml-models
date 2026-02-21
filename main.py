import polars as pl
import torch
import torch.optim as optim

from cnn.cnn import CNN
from cnn.loss import loss
from data.process_data import (
    generate_test_train_split,
    generate_time_series_for_one_result,
)

if __name__ == "__main__":
    time_interval = 1024
    df = pl.read_csv("data/EMG-data.csv")
    train_df, val_df, test_df = generate_test_train_split(df, 0.7, 0.15, 0.15)

    model = CNN(time_interval=time_interval)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 2

    for epoch in range(100):
        training_data: list[tuple[pl.DataFrame, int]] = []
        for _ in range(128):
            training_data.append(
                generate_time_series_for_one_result(train_df, time_interval)
            )

        train_batches: list[list[tuple]] = [
            training_data[i : i + batch_size]
            for i in range(0, len(training_data), batch_size)
        ]
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
            print(f"Loss: {loss_value.item()}")
        # Finished training part of epoch

        val_data: list[tuple[pl.DataFrame, int]] = []
        num_val_samples = 16
        cum_loss = 0
        for _ in range(num_val_samples):
            val_data.append(generate_time_series_for_one_result(val_df, time_interval))

        val_batches: list[list[tuple]] = [
            val_data[i : i + batch_size] for i in range(0, len(val_data), batch_size)
        ]
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

                # Calculate accuracy
                _, predicted = torch.max(val_outputs.data, 1)
                total += val_batch_labels.size(0)
                correct += (predicted == val_batch_labels).sum().item()

        avg_val_loss = cum_loss / len(val_batches)
        val_accuracy = correct / total
        print(
            f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
        )
