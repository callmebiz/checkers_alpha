@echo off
rem Long run: double-click to start or resume training (see
rem experiments\long_run.py). The first time, it continues the size study's
rem 10x128 network. Progress: runs\long_10x128\STATUS.md. To stop cleanly,
rem double-click train_stop.bat (or press Ctrl+C here).
title Checkers Alpha - training
cd /d "%~dp0"
set "PY=C:\Users\biz\miniconda3\envs\alpha_checkers2\python.exe"
set "RUN=runs\long_10x128"
"%PY%" -m experiments.long_run start %RUN% ^
  --from runs\size_study\runs\10x128 %*
pause
