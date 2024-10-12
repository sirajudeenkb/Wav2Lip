from flask import Flask, request, send_file
import os
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure folders
WAV2LIP_FOLDER = '/content/Wav2Lip'
UPLOAD_FOLDER = '/content/uploads'
OUTPUT_FOLDER = '/content/outputs'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def hello():
    return "I am not alive!"

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files or 'audio' not in request.files:
        return 'No video or audio file uploaded', 400

    video_file = request.files['video']
    audio_file = request.files['audio']

    if video_file.filename == '' or audio_file.filename == '':
        return 'No selected file', 400

    if video_file and audio_file:
        video_filename = secure_filename(video_file.filename)
        audio_filename = secure_filename(audio_file.filename)

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)

        video_file.save(video_path)
        audio_file.save(audio_path)

        output_filename = f"output_{video_filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Run Wav2Lip inference
        command = [
            "python", os.path.join(WAV2LIP_FOLDER, "inference.py"),
            "--checkpoint_path", os.path.join(WAV2LIP_FOLDER, "checkpoints/wav2lip.pth"),
            "--face", video_path,
            "--audio", audio_path,
            "--pads", "0", "15", "0", "0",
            "--resize_factor", "1",
            "--nosmooth",
            "--outfile", output_path
        ]

        try:
            subprocess.run(command, check=True, cwd=WAV2LIP_FOLDER)
        except subprocess.CalledProcessError as e:
            return f"Error processing video: {str(e)}", 500

        return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')