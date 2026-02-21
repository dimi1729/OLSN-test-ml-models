import polars as pl
import torch
import torch.optim as optim

from cnn.cnn import CNN
from cnn.loss import loss
from data.process_data import generate_time_series_for_one_result

if __name__ == "__main__":
    time_interval = 1024
    df = pl.read_csv("data/EMG-data.csv")
    df = df.drop("label")
    # Later when I do a train test split I will handle this automatically

    model = CNN(time_interval=time_interval)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data: list[tuple[pl.DataFrame, int]] = []
    for _ in range(128):
        data.append(generate_time_series_for_one_result(df, time_interval))

    batch_size = 2
    batches: list[list[tuple]] = [
        data[i : i + batch_size] for i in range(0, len(data), batch_size)
    ]
    # print(f"Batches: {batches}")
    for batch in batches:
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
