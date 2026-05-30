@echo off
setlocal
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if "%GRADIO_ALLOW_TASK_RUNS%"=="" set GRADIO_ALLOW_TASK_RUNS=1
if "%GRADIO_SERVER_NAME%"=="" set GRADIO_SERVER_NAME=0.0.0.0
if "%GRADIO_SERVER_PORT%"=="" set GRADIO_SERVER_PORT=7860
python app\gradio_app.py
endlocal
