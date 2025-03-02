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
import torch
import gc
import threading
import time

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
    MODEL_CACHE = {}  # Cache for loaded models
    CACHE_LOCK = threading.Lock()  # Lock for thread-safe cache operations

    # Language mapping
    LANGUAGE_MAPPING = {
        'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
        'Italian': 'it', 'Portuguese': 'pt', 'Polish': 'pl', 'Turkish': 'tr',
        'Russian': 'ru', 'Dutch': 'nl', 'Czech': 'cs', 'Arabic': 'ar',
        'Chinese (Simplified)': 'zh-cn',
        # Added Indian languages
        'Hindi': 'hi', 'Tamil': 'ta'
    }

class VideoProcessor:
    """Handles video processing operations"""
    def __init__(self):
        # Create necessary directories
        for folder in [Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER]:
            Path(folder).mkdir(parents=True, exist_ok=True)
        
        # Lazy loading of models
        self._whisper_model = None
        self._translator = None
        self._tts = None
        
        # Initialize resource monitor
        self._start_resource_monitor()
    
    def _start_resource_monitor(self):
        """Start a background thread to monitor GPU memory"""
        def monitor_resources():
            while True:
                if torch.cuda.is_available():
                    used = torch.cuda.memory_allocated() / (1024 ** 3)
                    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    logger.info(f"GPU Memory: {used:.2f}GB used / {total:.2f}GB total")
                time.sleep(30)  # Check every 30 seconds
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    @property
    def whisper_model(self):
        """Lazy loading of Whisper model"""
        if self._whisper_model is None:
            logger.info("Loading Whisper model...")
            try:
                self._whisper_model = WhisperModel("medium.en", device="cuda", compute_type="float16")
            except Exception as e:
                logger.warning(f"Failed to load Whisper on CUDA: {e}")
                logger.info("Falling back to CPU for Whisper")
                self._whisper_model = WhisperModel("medium.en", device="cpu", compute_type="int8")
        return self._whisper_model
    
    @property
    def translator(self):
        """Lazy loading of translator"""
        if self._translator is None:
            logger.info("Initializing translator...")
            self._translator = Translator()
        return self._translator
    
    @property
    def tts(self):
        """Lazy loading of TTS model"""
        if self._tts is None:
            logger.info("Loading TTS model...")
            try:
                self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
            except Exception as e:
                logger.warning(f"Failed to load TTS on CUDA: {e}")
                logger.info("Falling back to CPU for TTS")
                self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        return self._tts
        
    def _free_gpu_memory(self):
        """Free GPU memory after processing"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleared GPU cache")

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file"""
        audio_path = os.path.join(tempfile.gettempdir(), f'extracted_audio_{int(time.time())}.wav')
        try:
            ffmpeg_command = [
                'ffmpeg', '-i', video_path, '-acodec', 'pcm_s16le',  # Using 16-bit for better compatibility
                '-ar', '16000',  # Lower sample rate for efficiency
                '-ac', '1',  # Convert to mono
                '-q:a', '0', '-map', 'a', '-y', audio_path
            ]
            result = subprocess.run(ffmpeg_command, check=True, capture_output=True)
            logger.info(f"Audio extraction successful: {audio_path}")
            return audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio: {e.stderr.decode()}")
            raise RuntimeError(f"Audio extraction failed: {e.stderr.decode()}")

    def transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio to text"""
        try:
            segments, info = self.whisper_model.transcribe(file_path, beam_size=5)
            transcription = " ".join([segment.text for segment in segments])
            logger.info(f"Transcription completed. Length: {len(transcription)} chars")
            logger.info(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
            return transcription
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription failed: {str(e)}")

    def translate_text(self, text: str, target_language_code: str) -> str:
        """Translate text to target language"""
        # Skip translation if target is already English and source is detected as English
        if target_language_code == 'en' and text.strip() and text.strip()[0].isascii():
            logger.info("Skipping translation as content appears to be in English already")
            return text
        
        try:
            # Break long text into chunks to avoid translation API limits
            max_chunk_size = 1000
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            translated_chunks = []
            
            for chunk in chunks:
                translated = self.translator.translate(chunk, dest=target_language_code).text
                translated_chunks.append(translated)
            
            result = " ".join(translated_chunks)
            logger.info(f"Translation completed to {target_language_code}. Length: {len(result)} chars")
            return result
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise RuntimeError(f"Translation failed: {str(e)}")

    def synthesize_voice(self, text: str, target_language_code: str, original_audio_path: str) -> str:
        """Synthesize voice from text"""
        output_path = os.path.join(tempfile.gettempdir(), f'synthesized_audio_{int(time.time())}.wav')
        try:
            # Break text into sentences to improve TTS quality
            sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
            
            # Generate temporary files for each sentence
            temp_files = []
            for i, sentence in enumerate(sentences):
                if not sentence:
                    continue
                    
                temp_file = os.path.join(tempfile.gettempdir(), f'temp_audio_{i}_{int(time.time())}.wav')
                try:
                    self.tts.tts_to_file(
                        sentence,
                        speaker_wav=original_audio_path,
                        file_path=temp_file,
                        language=target_language_code
                    )
                    temp_files.append(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to synthesize sentence {i}: {str(e)}. Skipping.")
                
            # Concatenate all audio files
            if temp_files:
                concat_command = ['ffmpeg']
                for file in temp_files:
                    concat_command.extend(['-i', file])
                concat_command.extend(['-filter_complex', f'concat=n={len(temp_files)}:v=0:a=1[out]', '-map', '[out]', output_path])
                subprocess.run(concat_command, check=True, capture_output=True)
                
                # Clean up temp files
                for file in temp_files:
                    try:
                        os.remove(file)
                    except:
                        pass
            else:
                raise RuntimeError("No sentences were successfully synthesized")
                
            return output_path
        except Exception as e:
            logger.error(f"Voice synthesis failed: {str(e)}")
            raise RuntimeError(f"Voice synthesis failed: {str(e)}")
        finally:
            # Free GPU memory after synthesis
            self._free_gpu_memory()

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
            logger.info(f"Starting Wav2Lip processing...")
            result = subprocess.run(wav2lip_command, check=True, cwd=Config.WAV2LIP_FOLDER, capture_output=True)
            logger.info(f"Wav2Lip processing completed: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Wav2Lip processing failed: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"Lip synchronization failed: {e.stderr.decode() if e.stderr else str(e)}")

def create_app() -> Flask:
    """Application factory function"""
    app = Flask(__name__)
    # Configure CORS
    CORS(app)
    
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
    processor = VideoProcessor()

    def allowed_file(filename: str) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    @app.route("/health")
    def health_check():
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_available": True,
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
                "memory_reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
            }
        else:
            gpu_info = {"gpu_available": False}
            
        return jsonify({
            "status": "healthy",
            "gpu_info": gpu_info,
            "models_loaded": {
                "whisper": processor._whisper_model is not None,
                "tts": processor._tts is not None
            }
        }), 200

    @app.route('/process_video', methods=['POST'])
    def process_video():
        start_time = time.time()
        temp_files = []  # Track temporary files to clean up
        
        try:
            # Validate request
            if 'video' not in request.files:
                return jsonify({'error': 'Missing video file'}), 400
                
            if 'target_language' not in request.form:
                return jsonify({'error': 'Missing target language'}), 400

            video_file = request.files['video']
            target_language = request.form['target_language']
            logger.info(f"Received video processing request: {video_file.filename} to {target_language}")

            if video_file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
                
            if not allowed_file(video_file.filename):
                return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(Config.ALLOWED_EXTENSIONS)}'}), 400

            if target_language not in Config.LANGUAGE_MAPPING:
                return jsonify({'error': f'Unsupported target language. Supported languages: {", ".join(Config.LANGUAGE_MAPPING.keys())}'}), 400

            target_language_code = Config.LANGUAGE_MAPPING[target_language]
            logger.info(f"Target language code: {target_language_code}")

            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save and process video
                video_filename = secure_filename(video_file.filename)
                video_path = os.path.join(temp_dir, video_filename)
                video_file.save(video_path)
                logger.info(f"Video saved to {video_path}")
                
                # Process video step by step
                logger.info("Step 1: Extracting audio from video")
                audio_path = processor.extract_audio(video_path)
                temp_files.append(audio_path)
                
                logger.info("Step 2: Transcribing audio to text")
                transcription = processor.transcribe_audio(audio_path)
                
                logger.info("Step 3: Translating text")
                translated_text = processor.translate_text(transcription, target_language_code)
                
                logger.info("Step 4: Synthesizing voice")
                synthesized_audio_path = processor.synthesize_voice(
                    translated_text, target_language_code, audio_path
                )
                temp_files.append(synthesized_audio_path)

                # Generate output path with timestamp to avoid conflicts
                output_filename = f"output_{int(time.time())}_{video_filename}"
                output_path = os.path.join(Config.OUTPUT_FOLDER, output_filename)
                
                logger.info("Step 5: Running Wav2Lip for lip synchronization")
                processor.run_wav2lip(video_path, synthesized_audio_path, output_path)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                logger.info(f"Processing completed in {processing_time:.2f} seconds")

                # Return processed video
                return send_file(output_path, as_attachment=True, download_name=output_filename)

        except Exception as e:
            import traceback
            logger.error(f"Error processing video: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Processing failed', 
                'details': str(e),
                'processing_time': time.time() - start_time
            }), 500
        finally:
            # Clean up any temporary files
            for file_path in temp_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {file_path}: {e}")
            
            # Free GPU memory
            processor._free_gpu_memory()

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
