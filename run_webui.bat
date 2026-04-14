@echo off
REM Activate the local venv and launch the Wav2Lip WebUI.
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
python webui.py
