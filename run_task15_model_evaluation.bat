@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python src\evaluate_models.py %*
