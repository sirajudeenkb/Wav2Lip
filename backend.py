from flask import Flask, request, send_file
import os
import subprocess
from werkzeug.utils import secure_filename
from faster_whisper import WhisperModel
from googletrans import Translator
from TTS.api import TTS
import torch

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

# Initialize models
whisper_model = WhisperModel("medium.en", device="cuda", compute_type="float16")
translator = Translator()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True).to("cuda")

# Language mapping
language_mapping = {
    'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
    'Italian': 'it', 'Portuguese': 'pt', 'Polish': 'pl', 'Turkish': 'tr',
    'Russian': 'ru', 'Dutch': 'nl', 'Czech': 'cs', 'Arabic': 'ar',
    'Chinese (Simplified)': 'zh-cn'
}

def extract_audio(video_path):
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_audio.wav')
    ffmpeg_command = f"ffmpeg -i '{video_path}' -acodec pcm_s24le -ar 48000 -q:a 0 -map a -y '{audio_path}'"
    subprocess.run(ffmpeg_command, shell=True, check=True)
    return audio_path

def transcribe_audio(file_path):
    segments, _ = whisper_model.transcribe(file_path)
    return " ".join([segment.text for segment in segments])

def translate_text(text, target_language_code):
    return translator.translate(text, dest=target_language_code).text

def synthesize_voice(text, target_language_code, original_audio_path):
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'synthesized_audio.wav')
    tts.tts_to_file(text,
        speaker_wav=original_audio_path,
        file_path=output_path,
        language=target_language_code
    )
    return output_path

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files or 'target_language' not in request.form:
        return 'Missing video file or target language', 400

    video_file = request.files['video']
    target_language = request.form['target_language']

    if video_file.filename == '':
        return 'No selected file', 400

    if target_language not in language_mapping:
        return 'Invalid target language', 400

    target_language_code = language_mapping[target_language]

    if video_file:
        try:
            # Save and process video
            video_filename = secure_filename(video_file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            video_file.save(video_path)

            # Extract audio
            audio_path = extract_audio(video_path)

            # Transcribe audio
            transcription = transcribe_audio(audio_path)

            # Translate text
            translated_text = translate_text(transcription, target_language_code)

            # Synthesize voice
            synthesized_audio_path = synthesize_voice(translated_text, target_language_code, audio_path)

            # Run Wav2Lip
            output_filename = f"output_{video_filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            wav2lip_command = [
                "python", os.path.join(WAV2LIP_FOLDER, "inference.py"),
                "--checkpoint_path", os.path.join(WAV2LIP_FOLDER, "checkpoints/wav2lip.pth"),
                "--face", video_path,
                "--audio", synthesized_audio_path,
                "--pads", "0", "15", "0", "0",
                "--resize_factor", "1",
                "--nosmooth",
                "--outfile", output_path
            ]
            subprocess.run(wav2lip_command, check=True, cwd=WAV2LIP_FOLDER)

            # Clean up temporary files
            os.remove(video_path)
            os.remove(audio_path)
            os.remove(synthesized_audio_path)

            return send_file(output_path, as_attachment=True)

        except Exception as e:
            return f"Error processing video: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')