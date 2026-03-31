# Example run for if you are on linux or mac
# You can run this by doing "bash example_runs/linux-mac.sh" on linux
# or by doing "zsh example_runs/linux-mac.sh" on mac
#
# The point of running it in a file like this is you can give your runs names and easily
# remember what parameters you used. The 'runs' directory is in the .gitignore, so put your
# run files in there and call the scripts from there

# Single GPU/CPU run
uv run main.py --run_name="example_run" --epochs=500 --val_samples=16

# Multi-GPU run with Distributed Data Parallel (DDP)
# Replace <num_gpus> with the number of GPUs you want to use (e.g., 2, 4, 8)
# uv run torchrun --nproc_per_node=<num_gpus> main.py --run_name="example_run_ddp" --epochs=500 --val_samples=16
