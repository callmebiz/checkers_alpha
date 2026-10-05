@echo off
rem Checkers Alpha: double-click to play and analyse in the browser.
rem The work is in scripts\start.ps1: build the UI if needed, start the
rem server in this window, open the browser. Close the window to stop.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1"
if errorlevel 1 pause
