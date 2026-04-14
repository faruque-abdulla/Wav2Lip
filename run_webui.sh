#!/usr/bin/env bash
# Launch the Wav2Lip WebUI from the workspace venv.
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi
python webui.py
