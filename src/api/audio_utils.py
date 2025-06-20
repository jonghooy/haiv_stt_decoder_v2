#!/usr/bin/env python3
"""
Audio Processing Utilities for STT API
Handle various audio format processing and conversion
"""

import base64
import io
import logging
import tempfile
import wave
from pathlib import Path
from typing import Tuple, Optional, Union

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import which

# Configure logging
logger = logging.getLogger(__name__)

# Audio processing constants
SUPPORTED_FORMATS = {
    'pcm_16khz': 'Raw PCM 16kHz',
    'wav': 'WAV',
    'mp3': 'MP3', 
    'flac': 'FLAC',
    'ogg': 'OGG',
    'm4a': 'M4A/AAC',
    'aac': 'AAC',
    'webm': 'WebM',
    'mp4': 'MP4'
}

TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_DURATION = 300.0  # 5 minutes
MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass


class AudioValidator:
    """Validate audio data and parameters"""
    
    @staticmethod
    def validate_base64(audio_data: str) -> bytes:
        """Validate and decode base64 audio data"""
        try:
            decoded = base64.b64decode(audio_data, validate=True)
            if len(decoded) == 0:
                raise AudioProcessingError("Audio data is empty")
            if len(decoded) > MAX_AUDIO_SIZE:
                raise AudioProcessingError(f"Audio data too large: {len(decoded)} bytes (max: {MAX_AUDIO_SIZE})")
            return decoded
        except Exception as e:
            raise AudioProcessingError(f"Invalid base64 audio data: {e}")
    
    @staticmethod
    def validate_sample_rate(sample_rate: int) -> None:
        """Validate sample rate"""
        if not 8000 <= sample_rate <= 48000:
            raise AudioProcessingError(f"Invalid sample rate: {sample_rate} Hz (must be 8000-48000)")
    
    @staticmethod
    def validate_audio_duration(duration: float) -> None:
        """Validate audio duration"""
        if duration <= 0:
            raise AudioProcessingError("Audio duration must be positive")
        if duration > MAX_AUDIO_DURATION:
            raise AudioProcessingError(f"Audio too long: {duration:.1f}s (max: {MAX_AUDIO_DURATION}s)")


class AudioFormatDetector:
    """Detect audio format from binary data"""
    
    # Magic bytes for format detection
    FORMAT_SIGNATURES = {
        b'RIFF': 'wav',
        b'ID3': 'mp3',
        b'\xff\xfb': 'mp3',
        b'\xff\xf3': 'mp3',
        b'\xff\xf2': 'mp3',
        b'fLaC': 'flac',
        b'OggS': 'ogg',
        b'\x00\x00\x00 ftypM4A': 'm4a',
        b'\x00\x00\x00\x18ftypmp42': 'mp4',
        b'\x1a\x45\xdf\xa3': 'webm'
    }
    
    @classmethod
    def detect_format(cls, audio_bytes: bytes) -> Optional[str]:
        """Detect audio format from binary data"""
        for signature, format_name in cls.FORMAT_SIGNATURES.items():
            if audio_bytes.startswith(signature):
                return format_name
        
        # Check for M4A/MP4 more thoroughly
        if b'ftyp' in audio_bytes[:20]:
            if b'M4A' in audio_bytes[:30] or b'mp4' in audio_bytes[:30]:
                return 'm4a'
        
        return None


class AudioConverter:
    """Convert audio between different formats and sample rates"""
    
    def __init__(self):
        """Initialize audio converter"""
        # Check for ffmpeg availability
        self.has_ffmpeg = which("ffmpeg") is not None
        if not self.has_ffmpeg:
            logger.warning("FFmpeg not found. Some audio formats may not be supported.")
    
    def pcm_to_numpy(self, pcm_bytes: bytes, sample_rate: int = TARGET_SAMPLE_RATE, 
                     dtype: str = 'int16') -> np.ndarray:
        """Convert raw PCM bytes to numpy array"""
        try:
            if dtype == 'int16':
                audio_array = np.frombuffer(pcm_bytes, dtype=np.int16)
                # Convert to float32 [-1, 1]
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif dtype == 'float32':
                audio_array = np.frombuffer(pcm_bytes, dtype=np.float32)
            else:
                raise AudioProcessingError(f"Unsupported PCM dtype: {dtype}")
            
            return audio_array
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to convert PCM data: {e}")
    
    def decode_with_soundfile(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Decode audio using soundfile"""
        try:
            audio_io = io.BytesIO(audio_bytes)
            audio_data, sample_rate = sf.read(audio_io, always_2d=False)
            
            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            return audio_data, sample_rate
            
        except Exception as e:
            raise AudioProcessingError(f"Soundfile decoding failed: {e}")
    
    def decode_with_librosa(self, audio_bytes: bytes, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
        """Decode audio using librosa"""
        try:
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                
                audio_data, sample_rate = librosa.load(
                    temp_file.name,
                    sr=target_sr,
                    mono=True
                )
                
                return audio_data, sample_rate
                
        except Exception as e:
            raise AudioProcessingError(f"Librosa decoding failed: {e}")
    
    def decode_with_pydub(self, audio_bytes: bytes, format_hint: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """Decode audio using pydub"""
        try:
            # Detect format if not provided
            if format_hint is None:
                format_hint = AudioFormatDetector.detect_format(audio_bytes)
            
            # Create AudioSegment
            audio_io = io.BytesIO(audio_bytes)
            
            if format_hint == 'mp3':
                audio_segment = AudioSegment.from_mp3(audio_io)
            elif format_hint == 'wav':
                audio_segment = AudioSegment.from_wav(audio_io)
            elif format_hint == 'flac':
                audio_segment = AudioSegment.from_file(audio_io, format="flac")
            elif format_hint == 'ogg':
                audio_segment = AudioSegment.from_ogg(audio_io)
            elif format_hint in ['m4a', 'aac']:
                audio_segment = AudioSegment.from_file(audio_io, format="m4a")
            elif format_hint == 'mp4':
                audio_segment = AudioSegment.from_file(audio_io, format="mp4")
            elif format_hint == 'webm':
                audio_segment = AudioSegment.from_file(audio_io, format="webm")
            else:
                # Try generic
                audio_segment = AudioSegment.from_file(audio_io)
            
            # Convert to mono
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # Get raw audio data
            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Normalize based on sample width
            if audio_segment.sample_width == 1:  # 8-bit
                audio_data /= 128.0
            elif audio_segment.sample_width == 2:  # 16-bit
                audio_data /= 32768.0
            elif audio_segment.sample_width == 4:  # 32-bit
                audio_data /= 2147483648.0
            
            return audio_data, audio_segment.frame_rate
            
        except Exception as e:
            raise AudioProcessingError(f"Pydub decoding failed: {e}")
    
    def resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio_data
        
        try:
            resampled = librosa.resample(
                audio_data,
                orig_sr=orig_sr,
                target_sr=target_sr,
                res_type='kaiser_best'
            )
            return resampled
        except Exception as e:
            raise AudioProcessingError(f"Resampling failed: {e}")
    
    def ensure_mono(self, audio_data: np.ndarray) -> np.ndarray:
        """Ensure audio is mono"""
        if audio_data.ndim == 1:
            return audio_data
        elif audio_data.ndim == 2:
            # Convert stereo to mono by averaging channels
            return np.mean(audio_data, axis=1)
        else:
            raise AudioProcessingError(f"Unsupported audio shape: {audio_data.shape}")
    
    def normalize_audio(self, audio_data: np.ndarray, target_level: float = -20.0) -> np.ndarray:
        """Normalize audio to target RMS level (in dB)"""
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        if rms == 0:
            return audio_data
        
        # Convert target level from dB to linear
        target_rms = 10 ** (target_level / 20.0)
        
        # Apply normalization
        scaling_factor = target_rms / rms
        normalized = audio_data * scaling_factor
        
        # Clip to prevent overflow
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized


class AudioProcessor:
    """Main audio processing class"""
    
    def __init__(self):
        """Initialize audio processor"""
        self.validator = AudioValidator()
        self.detector = AudioFormatDetector()
        self.converter = AudioConverter()
        
        logger.info("AudioProcessor initialized")
    
    def process_audio_data(self, 
                          audio_data: str,
                          audio_format: str,
                          sample_rate: int = TARGET_SAMPLE_RATE,
                          normalize: bool = True) -> Tuple[np.ndarray, int, dict]:
        """
        Process base64 audio data - Only supports PCM 16kHz format
        
        Args:
            audio_data: Base64 encoded audio data (PCM 16-bit, 16kHz)
            audio_format: Expected audio format (only 'pcm_16khz' supported)
            sample_rate: Target sample rate (must be 16000)
            normalize: Whether to normalize audio
            
        Returns:
            Tuple of (audio_array, final_sample_rate, metadata)
        """
        processing_info = {
            'original_format': audio_format,
            'original_size': 0,
            'detected_format': 'pcm_16khz',
            'processing_method': 'pcm_direct',
            'resampled': False,
            'normalized': normalize,
            'duration': 0.0
        }
        
        try:
            # Only support PCM 16kHz
            if audio_format.lower() not in ['pcm_16khz', 'pcm']:
                raise ValueError(f"지원하지 않는 오디오 포맷: {audio_format}. pcm_16khz만 지원됩니다.")
            
            if sample_rate != 16000:
                raise ValueError(f"지원하지 않는 샘플레이트: {sample_rate}. 16000Hz만 지원됩니다.")
            
            # Step 1: Validate and decode base64
            logger.debug("Decoding base64 PCM 16kHz audio data...")
            audio_bytes = self.validator.validate_base64(audio_data)
            processing_info['original_size'] = len(audio_bytes)
            
            # Step 2: Process PCM data
            logger.debug("Processing PCM 16kHz data...")
            audio_array = self.converter.pcm_to_numpy(audio_bytes, sample_rate)
            final_sr = sample_rate
            
            # Step 5: Ensure mono
            audio_array = self.converter.ensure_mono(audio_array)
            
            # Step 6: Validate duration
            duration = len(audio_array) / final_sr
            processing_info['duration'] = duration
            self.validator.validate_audio_duration(duration)
            
            # Step 7: Normalize if requested
            if normalize:
                audio_array = self.converter.normalize_audio(audio_array)
            
            logger.info(f"Audio processed successfully: {duration:.2f}s @ {final_sr}Hz")
            
            return audio_array, final_sr, processing_info
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise AudioProcessingError(f"Audio processing failed: {e}")
    
    def _decode_encoded_audio(self, audio_bytes: bytes, format_hint: str) -> Tuple[np.ndarray, int]:
        """Try multiple decoders to decode audio"""
        errors = []
        
        # Try soundfile first (fastest)
        try:
            return self.converter.decode_with_soundfile(audio_bytes)
        except Exception as e:
            errors.append(f"soundfile: {e}")
        
        # Try pydub (most compatible)
        try:
            return self.converter.decode_with_pydub(audio_bytes, format_hint)
        except Exception as e:
            errors.append(f"pydub: {e}")
        
        # Try librosa (fallback)
        try:
            return self.converter.decode_with_librosa(audio_bytes)
        except Exception as e:
            errors.append(f"librosa: {e}")
        
        # All decoders failed
        error_msg = f"All audio decoders failed: {'; '.join(errors)}"
        raise AudioProcessingError(error_msg)
    
    def create_test_audio(self, duration: float = 2.0, frequency: float = 440.0, 
                         sample_rate: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, str]:
        """
        Create test audio data for testing
        
        Returns:
            Tuple of (audio_array, base64_encoded_audio)
        """
        # Generate sine wave
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        audio_array = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Convert to int16 PCM
        pcm_data = (audio_array * 32767).astype(np.int16).tobytes()
        
        # Encode to base64
        base64_audio = base64.b64encode(pcm_data).decode('utf-8')
        
        return audio_array, base64_audio
    
    def audio_to_wav_base64(self, audio_array: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> str:
        """Convert numpy audio array to base64 encoded WAV"""
        try:
            # Create WAV in memory
            wav_io = io.BytesIO()
            
            # Convert float to int16
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Get WAV bytes and encode to base64
            wav_bytes = wav_io.getvalue()
            return base64.b64encode(wav_bytes).decode('utf-8')
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to create WAV: {e}")


# Global processor instance
_audio_processor = None

def get_audio_processor() -> AudioProcessor:
    """Get global audio processor instance"""
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor()
    return _audio_processor


# Convenience functions
def process_audio_data(audio_data: str, audio_format: str, sample_rate: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int, dict]:
    """Convenience function for processing audio data"""
    processor = get_audio_processor()
    return processor.process_audio_data(audio_data, audio_format, sample_rate)


def create_test_audio(duration: float = 2.0, frequency: float = 440.0) -> Tuple[np.ndarray, str]:
    """Convenience function for creating test audio"""
    processor = get_audio_processor()
    return processor.create_test_audio(duration, frequency)


if __name__ == "__main__":
    # Test the audio processor
    logging.basicConfig(level=logging.INFO)
    
    processor = AudioProcessor()
    
    # Create test audio
    print("Creating test audio...")
    audio_array, base64_audio = processor.create_test_audio(duration=2.0, frequency=440.0)
    
    print(f"Test audio created: {len(audio_array)} samples, {len(base64_audio)} base64 chars")
    
    # Process the test audio
    print("Processing test audio...")
    processed_audio, sr, info = processor.process_audio_data(base64_audio, 'pcm_16khz', 16000)
    
    print(f"Processing info: {info}")
    print(f"Processed audio: {len(processed_audio)} samples @ {sr}Hz")
    
    print("Audio processor test completed successfully!") 