import os
import sys
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent

DEFAULT_CHECKPOINT = str(HERE / 'checkpoints' / 'wav2lip_gan.pth')


def safe_copy_upload(uploaded_file, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(uploaded_file, str):
        src_path = Path(uploaded_file)
    else:
        src_path = Path(uploaded_file.name)
    target_path = dest_dir / src_path.name
    shutil.copy(str(src_path), str(target_path))
    return str(target_path)


def find_free_port(start_port=7860, max_port=7900):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for port in range(start_port, max_port + 1):
            try:
                sock.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    raise OSError(f'No free port found in range {start_port}-{max_port}')


def run_wav2lip(face_file, audio_file, checkpoint_path, static, fps, resize_factor, crop, rotate, nosmooth, batch_size):
    if face_file is None or audio_file is None:
        return None, 'Please upload both a face/video file and an audio file.'

    if not checkpoint_path:
        return None, 'Please provide the path to a Wav2Lip checkpoint (.pth file).'

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        return None, f'Checkpoint not found: {checkpoint_path}'

    workdir = Path(tempfile.mkdtemp(prefix='wav2lip_'))
    face_path = safe_copy_upload(face_file, workdir)
    audio_path = safe_copy_upload(audio_file, workdir)
    output_path = workdir / 'result.mp4'

    cmd = [
        sys.executable,
        str(HERE / 'inference.py'),
        '--checkpoint_path', str(checkpoint),
        '--face', str(face_path),
        '--audio', str(audio_path),
        '--outfile', str(output_path),
        '--wav2lip_batch_size', str(batch_size),
    ]

    if static:
        cmd.append('--static')
    if nosmooth:
        cmd.append('--nosmooth')
    if rotate:
        cmd.append('--rotate')
    if fps:
        cmd.extend(['--fps', str(fps)])
    if resize_factor and resize_factor > 1:
        cmd.extend(['--resize_factor', str(resize_factor)])
    if crop and crop != '0,0,0,0':
        crop_values = [int(x.strip()) for x in crop.split(',') if x.strip()]
        if len(crop_values) == 4:
            cmd.append('--crop')
            cmd.extend([str(x) for x in crop_values])

    try:
        proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return None, 'Inference timed out. Please try again with smaller video/audio or increase the timeout.'

    if proc.returncode != 0:
        error_msg = proc.stderr or proc.stdout or 'Unknown error'
        return None, f'Wav2Lip failed:\n{error_msg}'

    if not output_path.is_file():
        return None, 'Wav2Lip finished but output file was not created.'

    return str(output_path), 'Success! Download or preview the generated lip-synced video.'


def create_ui():
    with gr.Blocks(title='Wav2Lip WebUI') as demo:
        gr.Markdown('# Wav2Lip WebUI')
        gr.Markdown('Upload a face image/video and audio file, then click Run to generate a lip-synced video.')

        with gr.Row():
            with gr.Column(scale=2):
                face_input = gr.File(label='Face image or source video', file_count='single', type='filepath')
                audio_input = gr.File(label='Audio file (.wav/.mp3)', file_count='single', type='filepath')
                checkpoint_input = gr.Textbox(value=DEFAULT_CHECKPOINT, label='Checkpoint path', placeholder='Path to Wav2Lip checkpoint (.pth)')
                output_video = gr.Video(label='Generated video')
                status = gr.Textbox(label='Status', interactive=False)

            with gr.Column(scale=1):
                static = gr.Checkbox(label='Static image mode', value=False)
                nosmooth = gr.Checkbox(label='Disable smoothing', value=False)
                rotate = gr.Checkbox(label='Rotate input video 90° CW', value=False)
                fps = gr.Number(label='FPS (static image mode)', value=25, precision=0)
                resize_factor = gr.Number(label='Resize factor', value=1, precision=0)
                crop = gr.Textbox(label='Crop (top,bottom,left,right)', value='0,0,0,0')
                batch_size = gr.Number(label='Wav2Lip batch size', value=128, precision=0)
                run_button = gr.Button('Run Wav2Lip')

        run_button.click(
            run_wav2lip,
            inputs=[face_input, audio_input, checkpoint_input, static, fps, resize_factor, crop, rotate, nosmooth, batch_size],
            outputs=[output_video, status],
        )

    return demo


if __name__ == '__main__':
    port = find_free_port(7860, 7900)
    print(f'Launching Wav2Lip WebUI on http://localhost:{port}')
    create_ui().launch(server_name='0.0.0.0', server_port=port, share=False, css='footer {visibility: hidden;}')
