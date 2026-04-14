import os
from pathlib import Path
from urllib.request import urlretrieve

MODEL_URL = 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth'
TARGET_PATH = Path(__file__).resolve().parent / 'face_detection' / 'detection' / 'sfd' / 's3fd.pth'

TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)

if TARGET_PATH.exists():
    print(f'Face detection model already exists: {TARGET_PATH}')
else:
    print('Downloading face detection model...')
    urlretrieve(MODEL_URL, TARGET_PATH)
    print(f'Download complete: {TARGET_PATH}')
