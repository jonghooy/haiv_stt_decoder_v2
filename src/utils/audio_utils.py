import numpy as np
import base64
import io
import wave
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class AudioUtils:
    """오디오 처리 유틸리티 클래스"""
    
    def __init__(self):
        self.supported_pcm_formats = {
            'pcm': 16000,
            'pcm_16khz': 16000,
            'pcm_8khz': 8000,
            'pcm_44khz': 44100,
            'pcm_48khz': 48000
        }
    
    def process_pcm_audio(self, audio_data: bytes, audio_format: str) -> bytes:
        """PCM 오디오 데이터 처리"""
        try:
            # PCM 포맷별 샘플레이트 확인
            sample_rate = self.supported_pcm_formats.get(audio_format.lower(), 16000)
            
            # PCM 데이터를 int16 배열로 변환
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 16kHz로 리샘플링 (Whisper가 16kHz를 선호함)
            if sample_rate != 16000:
                audio_array = self._resample_audio(audio_array, sample_rate, 16000)
                logger.info(f"🔄 {sample_rate}Hz → 16kHz 리샘플링 완료")
            
            # float32로 정규화 [-1, 1]
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # WAV 포맷으로 래핑하여 반환
            return self._wrap_as_wav(audio_float, 16000)
            
        except Exception as e:
            logger.error(f"❌ PCM 처리 실패 ({audio_format}): {e}")
            raise ValueError(f"PCM 처리 실패: {e}")
    
    def _resample_audio(self, audio_array: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """오디오 리샘플링"""
        if from_sr == to_sr:
            return audio_array
        
        # 간단한 선형 보간 리샘플링
        duration = len(audio_array) / from_sr
        new_length = int(duration * to_sr)
        
        # 선형 보간
        old_indices = np.linspace(0, len(audio_array) - 1, new_length)
        resampled = np.interp(old_indices, np.arange(len(audio_array)), audio_array)
        
        return resampled.astype(np.int16)
    
    def _wrap_as_wav(self, audio_float: np.ndarray, sample_rate: int) -> bytes:
        """float32 오디오를 WAV 바이트로 래핑"""
        try:
            # float32를 int16으로 변환
            audio_int16 = (audio_float * 32767).astype(np.int16)
            
            # WAV 포맷으로 래핑
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # 모노
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                
                return wav_buffer.getvalue()
                
        except Exception as e:
            logger.error(f"❌ WAV 래핑 실패: {e}")
            raise

def decode_audio_data(audio_data: bytes, audio_format: str = 'pcm_16khz') -> Tuple[np.ndarray, int]:
    """
    Decode audio data - Only supports PCM 16kHz format.
    
    Args:
        audio_data: Raw audio bytes (PCM 16-bit, 16kHz)
        audio_format: Format of audio data (only 'pcm_16khz' supported)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        if audio_format.lower() in ['pcm_16khz', 'pcm']:
            # Handle raw PCM data (16kHz, 16-bit only)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Convert to float32 in range [-1, 1]
            audio_array = audio_array.astype(np.float32) / 32768.0
            sample_rate = 16000  # 고정 16kHz
            logger.info(f"📻 PCM 16kHz 오디오 처리: {len(audio_array)} samples, {sample_rate}Hz")
            return audio_array, sample_rate
        else:
            raise ValueError(f"지원하지 않는 오디오 포맷: {audio_format}. pcm_16khz만 지원됩니다.")
            
    except Exception as e:
        logger.error(f"PCM 16kHz 오디오 처리 실패: {e}")
        raise ValueError(f"PCM 16kHz 오디오 처리 실패: {e}")

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
                
                logger.info(f"📻 WAV 오디오 감지: {sample_rate}Hz, {channels}ch, {sampwidth*8}bit")
                
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
                    logger.info("🔄 스테레오 → 모노 변환 완료")
                
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