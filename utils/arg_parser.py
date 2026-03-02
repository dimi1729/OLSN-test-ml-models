import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--project_name",
    type=str,
    default="OLSN-ML",
    help="""Project name for wandb, should not change because all runs will be
    stored in this project in wandb, change the run name between runs""",
)
parser.add_argument(
    "--run_name", type=str, default="test-run", help="Run name in wandb"
)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument(
    "--epochs", type=int, default=1000, help="Number of epochs to train"
)
parser.add_argument(
    "--train_samples",
    type=int,
    default=128,
    help="Number of samples to train on per epoch",
)
parser.add_argument(
    "--val_samples",
    type=int,
    default=16,
    help="Number of samples to validate on per epoch",
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["kaggle", "mendeley"],
    default="kaggle",
    help="""Which dataset to use. The kaggle dataset refers to this dataset:
    https://www.kaggle.com/datasets/sojanprajapati/emg-signal-for-gesture-recognition/data
    meanwhile the mendeley dataset refers to this one: https://data.mendeley.com/datasets/ckwc76xr2z/2
    Note that as of now the mendeley dataset has not been fully implemented""",
)
