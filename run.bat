@echo off
REM Check if .venv exists
if not exist ".venv" (
    echo [ERROR] Virtual environment folder '.venv' not found.
    echo Please create it first using: python -m venv .venv
    pause
    exit /b 1
)

REM Activate the virtual environment
call .venv\Scripts\activate

REM Check if requirements are installed (optional, but good practice if you want to be thorough)
REM pip install -r requirements.txt

REM Run the main script
echo [INFO] Starting KatoSub...
python main.py

REM Deactivate and pause if it crashed
call deactivate
if %errorlevel% neq 0 (
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
)
