# OLSN-test-ml-models

This repository is to test machine learning models for reading EMG data to predict different hand positions

## Setup

To use this repository, you will have to download the data, set up an environment with uv, and setup your wandb login.

### Data setup

To download the  "Kaggle" dataset, go to https://www.kaggle.com/datasets/sojanprajapati/emg-signal-for-gesture-recognition/data and download the csv file. Place it in the data/ directory. The EMG-data_description.txt file describes what the data represents, and more information can be found at the Kaggle link

To download the "Mendeley" dataset, go to https://data.mendeley.com/datasets/ckwc76xr2z/2 and in the file system go to filtered > csv and then download all of the csvs there (you do not actually need to download all of them for the code to run, but since each file is a different participant, the train val test split will be done by num of files so make sure you get enough so that after the split you get at least one per class). Make sure to put all of the csvs (without renaming them) into the path data/datasets/mendeley in the project directory.

### uv setup

To get uv, please refer to https://docs.astral.sh/uv/getting-started/installation/. After installing, you can just run `uv sync` in the project folder to get all necessary dependencies. The main code can be run with `uv run main.py`

### wandb login

In order to log our runs, we are using wandb. For documentation and making an account, go to https://docs.wandb.ai/models/models_quickstart#sign-up-and-create-an-api-key. The first time you run main, you will be promped to create an account or use an existing one. The easiest thing to do is create an account at the link, select use an existing account, and then paste your API key. 

## Runs

Once all the setup is done, the code can be run with just `uv run main.py`. Additionally options for training are available, and can be seen in `utils/arg_parser.py`. The recommended way to make runs is to make run files (examples can be seen in `example_runs`) with different names. You can store them in a `runs` directory (in the .gitignore) so that they remain local.

## TODOs

There are many things that still need to be done for this project. Here is a list for those that want to contribute but do not know where to get started:
 - Improving the CNN: The CNN currently only gets about 50% accuracy on the dataset after 100 epochs, so performance needs to be improved significantly.
 - Data cleaning. Since there is a lot of noise and redundancy in the data since it is being sampled every 1ms, I think performance can be drastically improved by doing some kind of low pass filter before inputting to the CNN
 - Actual torch Dataloader and Dataset: Right now I just made some functions but ideally we change this to actual torch dataloaders
 - Random Forest or MLP: So far, we have a CNN, but we also plan on testing on other models.
