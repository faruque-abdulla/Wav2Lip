import argparse
from pathlib import Path
import urllib.request

try:
    import gdown
except ImportError:
    gdown = None

CHECKPOINT_ID = '15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk'
CHECKPOINT_NAME = 'wav2lip_gan.pth'
CHECKPOINT_URL = f'https://drive.google.com/uc?id={CHECKPOINT_ID}'
FACE_MODEL_URL = 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth'
FACE_MODEL_PATH = Path(__file__).resolve().parent / 'face_detection' / 'detection' / 'sfd' / 's3fd.pth'


def download_checkpoint(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / CHECKPOINT_NAME
    if target_path.exists():
        print(f'Checkpoint already exists: {target_path}')
        return target_path
    if gdown is None:
        raise RuntimeError('gdown is not installed. Run `pip install gdown` first.')

    print(f'Downloading Wav2Lip checkpoint to: {target_path}')
    gdown.download(CHECKPOINT_URL, str(target_path), quiet=False)
    if not target_path.exists():
        raise RuntimeError('Checkpoint download failed.')
    print('Download complete.')
    return target_path


def download_face_model():
    FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FACE_MODEL_PATH.exists():
        print(f'Face detector model already exists: {FACE_MODEL_PATH}')
        return FACE_MODEL_PATH

    print(f'Downloading face detector model to: {FACE_MODEL_PATH}')
    urllib.request.urlretrieve(FACE_MODEL_URL, str(FACE_MODEL_PATH))
    if not FACE_MODEL_PATH.exists():
        raise RuntimeError('Face detector download failed.')
    print('Download complete.')
    return FACE_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description='Download required models for local Wav2Lip inference.')
    parser.add_argument('--checkpoint', action='store_true', help='Download the Wav2Lip checkpoint.')
    parser.add_argument('--face', action='store_true', help='Download the face detection model.')
    parser.add_argument('--all', action='store_true', help='Download both checkpoint and face detection model.')
    args = parser.parse_args()

    if not any([args.checkpoint, args.face, args.all]):
        parser.print_help()
        return

    if args.all or args.checkpoint:
        download_checkpoint(Path(__file__).resolve().parent / 'checkpoints')
    if args.all or args.face:
        download_face_model()


if __name__ == '__main__':
    main()
