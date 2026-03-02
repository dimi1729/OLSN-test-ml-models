# Example run for if you are on Windows using PowerShell
# You can run this by doing ".\example_runs\windows.ps1" in PowerShell
#
# The point of running it in a file like this is you can give your runs names and easily
# remember what parameters you used. The 'runs' directory is in the .gitignore, so put your
# run files in there and call the scripts from there

uv run main.py --run_name="example_run" --epochs=500 --val_samples=16
