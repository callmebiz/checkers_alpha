@echo off
rem Stop the long run cleanly: training finishes its iteration and saves
rem (about a minute), then the training window says so. Resume any time
rem with train.bat.
cd /d "%~dp0"
"C:\Users\biz\miniconda3\envs\alpha_checkers2\python.exe" ^
  -m experiments.long_run stop runs\long_10x128
timeout /t 5
