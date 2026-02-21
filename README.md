# OLSN-test-ml-models

This repository is to test machine learning models for reading EMG data to predict different hand positions

## Setup

To use this repository, you will have to download the data, and also set up an environment with uv.

### Data setup

To download the data, go to https://www.kaggle.com/datasets/sojanprajapati/emg-signal-for-gesture-recognition/data and download the csv file. Place it in the data/ directory. The EMG-data_description.txt file describes what the data represents, and more information can be found at the Kaggle link

### uv setup

To get uv, please refer to https://docs.astral.sh/uv/getting-started/installation/. After installing, you can just run `uv sync` in the project folder to get all necessary dependencies. The main code can be run with `uv run main.py`

## TODOs

There are many things that still need to be done for this project. Here is a list for those that want to contribute but do not know where to get started:
 - Improving the CNN: The CNN currently only gets about 50% accuracy on the dataset after 100 epochs, so performance needs to be improved significantly.
 - Data cleaning. Since there is a lot of noise and redundancy in the data since it is being sampled every 1ms, I think performance can be drastically improved by doing some kind of low pass filter before inputting to the CNN
 - Better data logging: Instead of just printing to the terminal we should have some cloud based logging software (I recommend wandb)
 - Actual torch Dataloader and Dataset: Right now I just made some functions but ideally we change this to actual torch dataloaders
 - Random Forest: So far, we have a CNN, but we also plan on
