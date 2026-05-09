@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
    python -m venv .venv
)

call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\analyze_stopword_impact.py --config configs\task16_stopword_impact_config.json

endlocal
