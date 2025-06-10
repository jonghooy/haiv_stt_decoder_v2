#!/usr/bin/env python3
"""
API Data Models
Data models for request/response handling in the STT API
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Any
from enum import Enum
import base64
import time


class AudioFormat(str, Enum):
    """Supported audio formats"""
    PCM_16KHZ = "pcm_16khz"
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    M4A = "m4a"


class LanguageCode(str, Enum):
    """Supported language codes"""
    KOREAN = "ko"
    ENGLISH = "en"
    AUTO = "auto"


class TranscriptionRequest(BaseModel):
    """Request model for audio transcription"""
    
    # Audio data (base64 encoded)
    audio_data: str = Field(
        ..., 
        description="Base64 encoded audio data",
        example="UklGRjTQAABXQVZFZm10IBAAAAABAAEA..."
    )
    
    # Audio format
    audio_format: AudioFormat = Field(
        default=AudioFormat.PCM_16KHZ,
        description="Audio format specification"
    )
    
    # Sample rate (for PCM format)
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz",
        ge=8000,
        le=48000
    )
    
    # Language
    language: LanguageCode = Field(
        default=LanguageCode.KOREAN,
        description="Target language for transcription"
    )
    
    # Processing options
    enable_timestamps: bool = Field(
        default=True,
        description="Include word-level timestamps in response"
    )
    
    enable_confidence: bool = Field(
        default=True,
        description="Include confidence scores in response"
    )
    
    # Quality settings
    beam_size: int = Field(
        default=1,
        description="Beam size for decoding (higher = more accurate but slower)",
        ge=1,
        le=5
    )
    
    temperature: float = Field(
        default=0.0,
        description="Temperature for decoding (0.0 = deterministic)",
        ge=0.0,
        le=1.0
    )
    
    # Session info
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for tracking"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="Optional request ID for tracking"
    )
    
    @validator('audio_data')
    def validate_audio_data(cls, v):
        """Validate base64 audio data"""
        try:
            # Try to decode base64
            decoded = base64.b64decode(v)
            if len(decoded) == 0:
                raise ValueError("Audio data is empty")
            if len(decoded) > 50 * 1024 * 1024:  # 50MB limit
                raise ValueError("Audio data too large (max 50MB)")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 audio data: {e}")
    
    @validator('sample_rate')
    def validate_sample_rate(cls, v, values):
        """Validate sample rate compatibility"""
        audio_format = values.get('audio_format')
        if audio_format == AudioFormat.PCM_16KHZ and v != 16000:
            raise ValueError("PCM_16KHZ format requires 16000 Hz sample rate")
        return v


class WordSegment(BaseModel):
    """Word-level segment with timing"""
    
    word: str = Field(..., description="Transcribed word")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    confidence: Optional[float] = Field(
        default=None, 
        description="Confidence score (0.0-1.0)"
    )


class TranscriptionSegment(BaseModel):
    """Sentence-level transcription segment"""
    
    id: int = Field(..., description="Segment ID")
    text: str = Field(..., description="Transcribed text")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    confidence: Optional[float] = Field(
        default=None,
        description="Average confidence score for segment"
    )
    words: Optional[List[WordSegment]] = Field(
        default=None,
        description="Word-level breakdown"
    )


class ProcessingMetrics(BaseModel):
    """Processing performance metrics"""
    
    total_duration: float = Field(..., description="Total processing time in seconds")
    audio_duration: float = Field(..., description="Audio duration in seconds")
    rtf: float = Field(..., description="Real-time factor (processing_time / audio_duration)")
    model_load_time: Optional[float] = Field(
        default=None,
        description="Model loading time in seconds"
    )
    inference_time: float = Field(..., description="Inference time in seconds")
    queue_wait_time: Optional[float] = Field(
        default=None,
        description="Time spent waiting in queue"
    )


class TranscriptionResponse(BaseModel):
    """Response model for audio transcription"""
    
    # Results
    text: str = Field(..., description="Full transcribed text")
    segments: List[TranscriptionSegment] = Field(
        default_factory=list,
        description="Sentence-level segments"
    )
    
    # Language detection
    language: str = Field(..., description="Detected language")
    language_probability: float = Field(
        ..., 
        description="Language detection confidence"
    )
    
    # Metrics
    metrics: ProcessingMetrics = Field(..., description="Processing metrics")
    
    # Metadata
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID if provided"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID if provided"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Response timestamp"
    )
    model_info: dict = Field(
        default_factory=dict,
        description="Model information"
    )


class ErrorDetail(BaseModel):
    """Error detail information"""
    
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(
        default=None,
        description="Additional error details"
    )


class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: ErrorDetail = Field(..., description="Error information")
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID if provided"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Error timestamp"
    )


class HealthStatus(str, Enum):
    """Health check status values"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck(BaseModel):
    """Health check response"""
    
    status: HealthStatus = Field(..., description="Overall health status")
    timestamp: float = Field(
        default_factory=time.time,
        description="Health check timestamp"
    )
    
    # Component status
    model_status: HealthStatus = Field(..., description="Model health status")
    gpu_status: HealthStatus = Field(..., description="GPU health status")
    memory_status: HealthStatus = Field(..., description="Memory health status")
    
    # Metrics
    model_load_time: Optional[float] = Field(
        default=None,
        description="Model loading time in seconds"
    )
    average_inference_time: Optional[float] = Field(
        default=None,
        description="Average inference time in seconds"
    )
    total_requests: int = Field(
        default=0,
        description="Total number of requests processed"
    )
    memory_usage: dict = Field(
        default_factory=dict,
        description="Memory usage information"
    )
    
    # Version info
    version: str = Field(
        default="1.0.0",
        description="API version"
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Model version"
    )


class BatchTranscriptionRequest(BaseModel):
    """Request model for batch transcription"""
    
    # Audio files (list of base64 encoded data)
    audio_files: List[str] = Field(
        ...,
        description="List of base64 encoded audio files",
        min_items=1,
        max_items=100
    )
    
    # Common settings
    audio_format: AudioFormat = Field(
        default=AudioFormat.WAV,
        description="Audio format for all files"
    )
    language: LanguageCode = Field(
        default=LanguageCode.KOREAN,
        description="Target language for transcription"
    )
    
    # Processing options
    enable_timestamps: bool = Field(default=True)
    enable_confidence: bool = Field(default=True)
    beam_size: int = Field(default=1, ge=1, le=5)
    
    # Batch settings
    max_workers: int = Field(
        default=4,
        description="Maximum number of parallel workers",
        ge=1,
        le=8
    )
    
    # Session info
    batch_id: Optional[str] = Field(
        default=None,
        description="Optional batch ID for tracking"
    )


class BatchTranscriptionResponse(BaseModel):
    """Response model for batch transcription"""
    
    # Results
    results: List[TranscriptionResponse] = Field(
        ...,
        description="Transcription results for each file"
    )
    
    # Batch metrics
    total_files: int = Field(..., description="Total number of files processed")
    successful_files: int = Field(..., description="Number of successfully processed files")
    failed_files: int = Field(..., description="Number of failed files")
    
    # Performance metrics
    total_processing_time: float = Field(..., description="Total batch processing time")
    average_rtf: float = Field(..., description="Average RTF across all files")
    
    # Metadata
    batch_id: Optional[str] = Field(default=None)
    timestamp: float = Field(default_factory=time.time)


# Configuration models
class ModelConfig(BaseModel):
    """Model configuration"""
    
    model_name: str = Field(default="large-v3")
    device: str = Field(default="cuda")
    compute_type: str = Field(default="float16")
    beam_size: int = Field(default=1)
    language: str = Field(default="korean")


class ServerConfig(BaseModel):
    """Server configuration"""
    
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)
    max_concurrent_requests: int = Field(default=10)
    request_timeout: float = Field(default=30.0)
    model_settings: ModelConfig = Field(default_factory=ModelConfig) 