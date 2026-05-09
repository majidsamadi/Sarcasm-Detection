@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\activate call .venv\Scripts\activate
python src\error_analysis.py %*
endlocal
