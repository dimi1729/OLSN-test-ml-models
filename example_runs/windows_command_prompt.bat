@echo off
REM Example run for if you are on Windows using Command Prompt
REM You can run this by doing "example_runs\windows.bat" in Command Prompt
REM or by double-clicking this file in Windows Explorer
REM
REM The point of running it in a file like this is you can give your runs names and easily
REM remember what parameters you used. The 'runs' directory is in the .gitignore, so put your
REM run files in there and call the scripts from there

uv run main.py --run_name="example_run" --epochs=500 --val_samples=16
