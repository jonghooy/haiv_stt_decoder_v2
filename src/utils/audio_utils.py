import numpy as np
import base64
import io
import wave
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def decode_audio_data(audio_data: bytes, audio_format: str = 'pcm_16khz') -> Tuple[np.ndarray, int]:
    """
    Decode audio data from various formats to numpy array.
    
    Args:
        audio_data: Raw audio bytes
        audio_format: Format of audio data ('pcm_16khz', 'wav', 'mp3', etc.)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        if audio_format.lower() in ['pcm_16khz', 'pcm']:
            # Handle raw PCM data
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Convert to float32 in range [-1, 1]
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array, 16000
            
        elif audio_format.lower() == 'wav':
            # Handle WAV format
            return _decode_wav(audio_data)
            
        elif audio_format.lower() in ['mp3', 'flac', 'm4a']:
            # Handle compressed formats
            return _decode_compressed_audio(audio_data, audio_format)
            
        else:
            logger.warning(f"Unknown audio format: {audio_format}, trying automatic detection")
            return _decode_audio_automatic(audio_data)
            
    except Exception as e:
        logger.error(f"Failed to decode audio with format {audio_format}: {e}")
        raise ValueError(f"Audio processing failed: {e}")

def _decode_wav(audio_data: bytes) -> Tuple[np.ndarray, int]:
    """Decode WAV format audio data."""
    try:
        # Try with wave module first
        with io.BytesIO(audio_data) as audio_buffer:
            with wave.open(audio_buffer, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sampwidth = wav_file.getsampwidth()
                
                # Convert to numpy array
                if sampwidth == 1:
                    dtype = np.uint8
                    audio_array = np.frombuffer(frames, dtype=dtype)
                    audio_array = (audio_array.astype(np.float32) - 128) / 128.0
                elif sampwidth == 2:
                    dtype = np.int16
                    audio_array = np.frombuffer(frames, dtype=dtype)
                    audio_array = audio_array.astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    dtype = np.int32
                    audio_array = np.frombuffer(frames, dtype=dtype)
                    audio_array = audio_array.astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(f"Unsupported sample width: {sampwidth}")
                
                # Handle stereo to mono conversion
                if channels > 1:
                    audio_array = audio_array.reshape(-1, channels)
                    audio_array = np.mean(audio_array, axis=1)
                
                return audio_array, sample_rate
                
    except Exception as e:
        logger.warning(f"Wave module failed: {e}, trying alternative methods")
        
    # Fallback methods
    return _decode_audio_fallback(audio_data)

def _decode_compressed_audio(audio_data: bytes, audio_format: str) -> Tuple[np.ndarray, int]:
    """Decode compressed audio formats like MP3, FLAC, etc."""
    # Try pydub first
    try:
        from pydub import AudioSegment
        
        with io.BytesIO(audio_data) as audio_buffer:
            audio_segment = AudioSegment.from_file(audio_buffer, format=audio_format.lower())
            
            # Convert to mono if stereo
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # Get raw audio data
            raw_data = audio_segment.raw_data
            sample_rate = audio_segment.frame_rate
            
            # Convert to numpy array
            if audio_segment.sample_width == 1:
                audio_array = np.frombuffer(raw_data, dtype=np.uint8)
                audio_array = (audio_array.astype(np.float32) - 128) / 128.0
            elif audio_segment.sample_width == 2:
                audio_array = np.frombuffer(raw_data, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_segment.sample_width == 4:
                audio_array = np.frombuffer(raw_data, dtype=np.int32)
                audio_array = audio_array.astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {audio_segment.sample_width}")
            
            return audio_array, sample_rate
            
    except ImportError:
        logger.warning("pydub not available")
    except Exception as e:
        logger.warning(f"pydub failed: {e}")
    
    # Try soundfile
    try:
        import soundfile as sf
        
        with io.BytesIO(audio_data) as audio_buffer:
            audio_array, sample_rate = sf.read(audio_buffer)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Ensure float32
            audio_array = audio_array.astype(np.float32)
            
            return audio_array, sample_rate
            
    except ImportError:
        logger.warning("soundfile not available")
    except Exception as e:
        logger.warning(f"soundfile failed: {e}")
    
    # Try librosa as last resort
    try:
        import librosa
        
        with io.BytesIO(audio_data) as audio_buffer:
            audio_array, sample_rate = librosa.load(audio_buffer, sr=None, mono=True)
            return audio_array, sample_rate
            
    except ImportError:
        logger.warning("librosa not available")
    except Exception as e:
        logger.warning(f"librosa failed: {e}")
    
    raise ValueError(f"All audio decoders failed for format: {audio_format}")

def _decode_audio_fallback(audio_data: bytes) -> Tuple[np.ndarray, int]:
    """Fallback audio decoding using multiple libraries."""
    errors = []
    
    # Try soundfile
    try:
        import soundfile as sf
        
        with io.BytesIO(audio_data) as audio_buffer:
            audio_array, sample_rate = sf.read(audio_buffer)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            audio_array = audio_array.astype(np.float32)
            return audio_array, sample_rate
            
    except Exception as e:
        errors.append(f"soundfile: {e}")
    
    # Try pydub
    try:
        from pydub import AudioSegment
        
        with io.BytesIO(audio_data) as audio_buffer:
            audio_segment = AudioSegment.from_file(audio_buffer)
            
            # Convert to mono
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            raw_data = audio_segment.raw_data
            sample_rate = audio_segment.frame_rate
            
            if audio_segment.sample_width == 2:
                audio_array = np.frombuffer(raw_data, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0
            else:
                raise ValueError(f"Unsupported sample width: {audio_segment.sample_width}")
            
            return audio_array, sample_rate
            
    except Exception as e:
        errors.append(f"pydub: {e}")
    
    # Try librosa
    try:
        import librosa
        
        with io.BytesIO(audio_data) as audio_buffer:
            audio_array, sample_rate = librosa.load(audio_buffer, sr=None, mono=True)
            return audio_array, sample_rate
            
    except Exception as e:
        errors.append(f"librosa: {e}")
    
    raise ValueError(f"All audio decoders failed: {'; '.join(errors)}")

def _decode_audio_automatic(audio_data: bytes) -> Tuple[np.ndarray, int]:
    """Try to automatically detect and decode audio format."""
    # First, try to detect if it's WAV by checking header
    if len(audio_data) >= 12:
        if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
            logger.info("Detected WAV format from header")
            return _decode_wav(audio_data)
    
    # Try various formats
    formats_to_try = ['wav', 'mp3', 'flac', 'm4a']
    
    for fmt in formats_to_try:
        try:
            return _decode_compressed_audio(audio_data, fmt)
        except Exception:
            continue
    
    # Last resort: try as raw PCM
    try:
        logger.warning("Trying to decode as raw PCM data")
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0
        return audio_array, 16000
    except Exception as e:
        raise ValueError(f"Could not decode audio data: {e}") 