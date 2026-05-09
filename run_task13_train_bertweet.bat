@echo off
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python src\train_bertweet.py --experiment all --overwrite
endlocal
