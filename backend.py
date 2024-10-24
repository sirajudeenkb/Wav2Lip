from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import subprocess
import logging
from werkzeug.utils import secure_filename
from faster_whisper import WhisperModel
from googletrans import Translator
from TTS.api import TTS
from pathlib import Path
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management class"""
    WAV2LIP_FOLDER = '/content/Wav2Lip'
    UPLOAD_FOLDER = '/content/uploads'
    OUTPUT_FOLDER = '/content/outputs'
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

    # Language mapping
    LANGUAGE_MAPPING = {
        'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
        'Italian': 'it', 'Portuguese': 'pt', 'Polish': 'pl', 'Turkish': 'tr',
        'Russian': 'ru', 'Dutch': 'nl', 'Czech': 'cs', 'Arabic': 'ar',
        'Chinese (Simplified)': 'zh-cn'
    }

class VideoProcessor:
    """Handles video processing operations"""
    def __init__(self):
        self.whisper_model = WhisperModel("medium.en", device="cuda", compute_type="float16")
        self.translator = Translator()
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        
        # Create necessary directories
        for folder in [Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER]:
            Path(folder).mkdir(parents=True, exist_ok=True)

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file"""
        audio_path = os.path.join(tempfile.gettempdir(), 'extracted_audio.wav')
        try:
            ffmpeg_command = [
                'ffmpeg', '-i', video_path, '-acodec', 'pcm_s24le',
                '-ar', '48000', '-q:a', '0', '-map', 'a', '-y', audio_path
            ]
            subprocess.run(ffmpeg_command, check=True, capture_output=True)
            return audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio: {e.stderr.decode()}")
            raise RuntimeError("Audio extraction failed")

    def transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio to text"""
        try:
            segments, _ = self.whisper_model.transcribe(file_path)
            return " ".join([segment.text for segment in segments])
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    def translate_text(self, text: str, target_language_code: str) -> str:
        """Translate text to target language"""
        try:
            return self.translator.translate(text, dest=target_language_code).text
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise

    def synthesize_voice(self, text: str, target_language_code: str, original_audio_path: str) -> str:
        """Synthesize voice from text"""
        output_path = os.path.join(tempfile.gettempdir(), 'synthesized_audio.wav')
        try:
            self.tts.tts_to_file(
                text,
                speaker_wav=original_audio_path,
                file_path=output_path,
                language=target_language_code
            )
            return output_path
        except Exception as e:
            logger.error(f"Voice synthesis failed: {str(e)}")
            raise

    def run_wav2lip(self, video_path: str, audio_path: str, output_path: str):
        """Run Wav2Lip for lip synchronization"""
        try:
            wav2lip_command = [
                "python", os.path.join(Config.WAV2LIP_FOLDER, "inference.py"),
                "--checkpoint_path", os.path.join(Config.WAV2LIP_FOLDER, "checkpoints/wav2lip.pth"),
                "--face", video_path,
                "--audio", audio_path,
                "--pads", "0", "15", "0", "0",
                "--resize_factor", "1",
                "--nosmooth",
                "--outfile", output_path
            ]
            subprocess.run(wav2lip_command, check=True, cwd=Config.WAV2LIP_FOLDER, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Wav2Lip processing failed: {e.stderr.decode()}")
            raise RuntimeError("Lip synchronization failed")

def create_app() -> Flask:
    """Application factory function"""
    app = Flask(__name__)
    # Configure CORS
    CORS(app, resources={
        r"/process_video": {
            "origins": [
                "http://localhost:3000",  # React development server
                "http://localhost:5173",  # Vite development server
                # Add your production frontend URL when deployed
            ],
            "methods": ["POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "max_age": 3600
        },
        r"/health": {
            "origins": "*",  # Allow health checks from anywhere
            "methods": ["GET"]
        }
    })
    
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
    processor = VideoProcessor()

    def allowed_file(filename: str) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    @app.route("/health")
    def health_check():
        return jsonify({"status": "healthy"}), 200

    @app.route('/process_video', methods=['POST'])
    def process_video():
        try:
            # Validate request
            if 'video' not in request.files or 'target_language' not in request.form:
                return jsonify({'error': 'Missing video file or target language'}), 400

            video_file = request.files['video']
            target_language = request.form['target_language']

            if video_file.filename == '' or not allowed_file(video_file.filename):
                return jsonify({'error': 'Invalid file type'}), 400

            if target_language not in Config.LANGUAGE_MAPPING:
                return jsonify({'error': 'Unsupported target language'}), 400

            target_language_code = Config.LANGUAGE_MAPPING[target_language]

            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save and process video
                video_filename = secure_filename(video_file.filename)
                video_path = os.path.join(temp_dir, video_filename)
                video_file.save(video_path)

                # Process video
                audio_path = processor.extract_audio(video_path)
                transcription = processor.transcribe_audio(audio_path)
                translated_text = processor.translate_text(transcription, target_language_code)
                synthesized_audio_path = processor.synthesize_voice(
                    translated_text, target_language_code, audio_path
                )

                # Generate output path
                output_filename = f"output_{video_filename}"
                output_path = os.path.join(Config.OUTPUT_FOLDER, output_filename)

                # Run Wav2Lip
                processor.run_wav2lip(video_path, synthesized_audio_path, output_path)

                # Return processed video
                return send_file(output_path, as_attachment=True)

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return jsonify({'error': 'Internal server error', 'details': str(e)}), 500
        
    @app.after_request
    def after_request(response):
        """Ensure proper CORS headers are set"""
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0')