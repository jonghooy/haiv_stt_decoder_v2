#!/usr/bin/env python3
"""
STT Service for Real-time Speech-to-Text using Faster Whisper
Integrates with audio processing utilities and FastAPI server
"""

import asyncio
import logging
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import torch
from faster_whisper import WhisperModel
from dataclasses import dataclass, asdict

# Import our audio processing utilities
from ..utils.audio_utils import decode_audio_data
from ..utils.korean_processor import (
    get_korean_processor, 
    create_korean_optimized_params,
    KoreanProcessingConfig
)
from ..utils.performance_optimizer import (
    get_performance_optimizer,
    get_error_handler,
    OptimizationConfig,
    PerformanceMetrics
)
from ..utils.gpu_optimizer import (
    get_gpu_optimizer,
    apply_rtx_4090_optimizations,
    configure_whisper_for_rtx_4090,
    optimize_inference_for_rtx_4090
)
from ..utils.rtx_4090_config import (
    get_rtx_4090_optimizer,
    apply_rtx_4090_configuration,
    get_rtx_4090_whisper_config,
    get_rtx_4090_inference_params,
    monitor_rtx_4090_performance
)

# Configure logging
logger = logging.getLogger(__name__)

# STT Service constants
DEFAULT_MODEL = "large-v3"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_BEAM_SIZE = 1
DEFAULT_BEST_OF = 1
DEFAULT_PATIENCE = 1.0
DEFAULT_LENGTH_PENALTY = 1.0
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_NO_REPEAT_NGRAM_SIZE = 0
DEFAULT_TEMPERATURE = [0.0]
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.4
DEFAULT_LOG_PROB_THRESHOLD = -1.0
DEFAULT_NO_SPEECH_THRESHOLD = 0.6

# Performance targets
TARGET_RTF = 0.05  # 5% of real-time
TARGET_LATENCY_MS = 1200  # 1.2 seconds for 95% of requests


@dataclass
class STTResult:
    """STT transcription result with metadata"""
    text: str
    language: str
    segments: List[Dict]
    audio_duration: float
    processing_time: float
    rtf: float
    model_info: Dict
    
    @property
    def latency_ms(self) -> float:
        """Processing latency in milliseconds"""
        return self.processing_time * 1000
    
    @property
    def throughput(self) -> float:
        """Throughput: audio duration / processing time"""
        return self.audio_duration / self.processing_time if self.processing_time > 0 else float('inf')
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for JSON response"""
        return {
            'text': self.text,
            'language': self.language,
            'segments': [dict(segment) for segment in self.segments],
            'performance_metrics': {
                'processing_time_ms': self.latency_ms,
                'audio_duration_sec': self.audio_duration,
                'rtf': self.rtf,
                'throughput': self.throughput,
                'meets_targets': self.meets_performance_targets()
            },
            'model_info': self.model_info
        }
    
    def meets_performance_targets(self) -> bool:
        """Check if result meets performance targets"""
        return (self.rtf <= TARGET_RTF and 
                self.latency_ms <= TARGET_LATENCY_MS)
    
    def __str__(self) -> str:
        performance_status = "✅ MEETS TARGETS" if self.meets_performance_targets() else "⚠️ BELOW TARGETS"
        return (f"STTResult(text='{self.text[:50]}...', "
                f"lang={self.language}, "
                f"rtf={self.rtf:.3f}x, "
                f"latency={self.latency_ms:.0f}ms) {performance_status}")


class STTServiceError(Exception):
    """Custom exception for STT service errors"""
    pass


class FasterWhisperSTTService:
    """
    Faster Whisper STT Service for real-time speech-to-text
    Optimized for low latency and high throughput
    """
    
    def __init__(self,
                 model_size: str = DEFAULT_MODEL,
                 device: str = DEFAULT_DEVICE,
                 compute_type: str = DEFAULT_COMPUTE_TYPE,
                 model_path: Optional[str] = None,
                 download_root: Optional[str] = None,
                 local_files_only: bool = False):
        """
        Initialize STT service
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to run on (cuda, cpu, auto)
            compute_type: Computation precision (float16, float32, int8)
            model_path: Path to local model files
            download_root: Directory to cache downloaded models
            local_files_only: Only use local files, don't download
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model_path = model_path
        self.download_root = download_root
        self.local_files_only = local_files_only
        
        # Service state
        self.model: Optional[WhisperModel] = None
        self.is_initialized = False
        self.initialization_error: Optional[str] = None
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="stt-worker")
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_audio_duration': 0.0,
            'total_processing_time': 0.0,
            'avg_rtf': 0.0,
            'avg_latency_ms': 0.0
        }
        
        # Thread lock for stats
        self.stats_lock = threading.Lock()
        
        # Initialize Korean processor with optimized settings
        self.korean_processor = get_korean_processor(
            KoreanProcessingConfig(
                use_korean_vad=True,
                normalize_text=True, 
                enhance_word_boundaries=True,
                filter_confidence_threshold=0.3,
                min_word_length=1,
                enable_post_processing=True
            )
        )
        
        # Initialize performance optimizer
        optimization_config = OptimizationConfig(
            max_worker_threads=6,
            enable_gc_optimization=True,
            gpu_memory_fraction=0.8,
            enable_memory_monitoring=True,
            alert_threshold_rtf=0.1,
            alert_threshold_latency_ms=1500,
            max_retry_attempts=3,
            enable_graceful_degradation=True
        )
        
        self.performance_optimizer = get_performance_optimizer(optimization_config)
        self.error_handler = get_error_handler(optimization_config)
        
        # Initialize GPU optimizer for RTX 4090
        self.gpu_optimizer = get_gpu_optimizer()
        self.rtx_4090_optimizer = get_rtx_4090_optimizer()
        self.gpu_optimizations_applied = False
        self.rtx_4090_optimizations_applied = False
        
        logger.info(f"STT Service initialized with model={model_size}, device={device}, compute_type={compute_type}")
        logger.info("Korean language processor enabled with optimized settings")
        logger.info("Performance optimization and monitoring enabled")
        logger.info("GPU optimizer initialized for RTX 4090 optimizations")
    
    async def initialize(self) -> bool:
        """
        Initialize the STT service asynchronously
        
        Returns:
            bool: True if initialization successful
        """
        if self.is_initialized:
            logger.info("STT Service already initialized")
            return True
        
        try:
            logger.info("Initializing STT Service...")
            
            # Initialize in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.executor, 
                self._initialize_sync
            )
            
            if success:
                self.is_initialized = True
                logger.info("✅ STT Service initialization completed successfully")
                return True
            else:
                logger.error(f"❌ STT Service initialization failed: {self.initialization_error}")
                return False
                
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ STT Service initialization failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _initialize_sync(self) -> bool:
        """Synchronous initialization helper with GPU optimizations"""
        try:
            # Apply RTX 4090 optimizations first
            if self.device.startswith('cuda') or (self.device == "auto" and torch.cuda.is_available()):
                logger.info("Applying comprehensive RTX 4090 optimizations...")
                
                # Apply basic GPU optimizations
                gpu_optimizations = apply_rtx_4090_optimizations()
                self.gpu_optimizations_applied = 'error' not in gpu_optimizations
                
                # Apply advanced RTX 4090 configuration
                rtx_4090_optimizations = apply_rtx_4090_configuration()
                self.rtx_4090_optimizations_applied = not any('error' in result for result in rtx_4090_optimizations.values() if isinstance(result, dict))
                
                if self.gpu_optimizations_applied and self.rtx_4090_optimizations_applied:
                    logger.info("✅ Comprehensive RTX 4090 optimizations applied successfully")
                    # Log optimization details
                    for key, value in gpu_optimizations.items():
                        if key != 'error':
                            logger.info(f"   Basic {key}: {value}")
                    for category, details in rtx_4090_optimizations.items():
                        if isinstance(details, dict) and 'error' not in details:
                            logger.info(f"   Advanced {category}: Applied")
                else:
                    logger.warning("⚠️ Some GPU optimizations failed to apply")
            
            # Initialize Faster Whisper model
            logger.info(f"Loading Faster Whisper model: {self.model_size}")
            start_time = time.time()
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Auto-detected device: {device}")
            else:
                device = self.device
            
            # Base model parameters
            base_model_kwargs = {
                'model_size_or_path': self.model_path or self.model_size,
                'device': device,
                'compute_type': self.compute_type,
                'download_root': self.download_root,
                'local_files_only': self.local_files_only
            }
            
            # Apply RTX 4090 specific model optimizations
            if device.startswith('cuda') and self.rtx_4090_optimizations_applied:
                # Use advanced RTX 4090 configuration
                model_kwargs = get_rtx_4090_whisper_config(base_model_kwargs)
                logger.info("✅ Whisper model configured with advanced RTX 4090 settings")
            elif device.startswith('cuda') and self.gpu_optimizations_applied:
                # Fallback to basic GPU optimization
                model_kwargs = configure_whisper_for_rtx_4090(base_model_kwargs)
                logger.info("✅ Whisper model configured for RTX 4090 (basic)")
            else:
                model_kwargs = base_model_kwargs
            
            # Load model
            self.model = WhisperModel(**model_kwargs)
            
            loading_time = time.time() - start_time
            logger.info(f"✅ Model loaded successfully in {loading_time:.2f}s")
            logger.info(f"   GPU optimizations: {'Applied' if self.gpu_optimizations_applied else 'Not applied'}")
            
            # Test inference to warm up model
            logger.info("Warming up model with test inference...")
            self._warmup_model()
            
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Sync initialization failed: {e}")
            return False
    
    def _warmup_model(self):
        """Warm up the model with a short test audio"""
        try:
            # Create 1-second test audio
            test_audio = np.random.random(16000).astype(np.float32) * 0.1
            
            # Run test transcription
            start_time = time.time()
            segments, info = self.model.transcribe(
                test_audio,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=False,
                language="ko"  # Korean for our use case
            )
            
            # Consume generator to complete transcription
            list(segments)
            
            warmup_time = time.time() - start_time
            logger.info(f"✅ Model warmup completed in {warmup_time:.3f}s")
            
        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {e}")
    
    async def shutdown(self):
        """Shutdown the STT service"""
        logger.info("Shutting down STT Service...")
        
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("Thread pool executor shutdown completed")
        
        self.is_initialized = False
        logger.info("✅ STT Service shutdown completed")
    
    async def transcribe_audio(
        self,
        audio_data: str,  # base64 encoded audio
        audio_format: str = 'pcm_16khz',
        language: Optional[str] = None,
        task: str = 'transcribe',
        beam_size: int = 5,
        word_timestamps: bool = False,
        vad_filter: bool = True,
        vad_parameters: Optional[Dict[str, Any]] = None
    ) -> STTResult:
        """
        Transcribe audio using Faster Whisper model.
        
        Args:
            audio_data: Base64 encoded audio data
            audio_format: Audio format ('pcm_16khz', 'wav', 'mp3', etc.)
            language: Language code ('ko', 'en', None for auto-detect)
            task: 'transcribe' or 'translate'
            beam_size: Beam size for decoding
            word_timestamps: Whether to return word-level timestamps
            vad_filter: Whether to use voice activity detection
            vad_parameters: VAD parameters
            
        Returns:
            STTResult with transcription and metadata
        """
        # Auto-initialize model if not loaded
        if not self.model:
            logger.info("Model not initialized, starting initialization...")
            try:
                initialization_success = await self.initialize()
                if not initialization_success:
                    raise STTServiceError(f"STT model initialization failed: {self.initialization_error}")
            except Exception as e:
                raise STTServiceError(f"STT model initialization failed: {str(e)}")
        
        start_time = time.time()
        
        # Apply pre-inference optimizations
        self.performance_optimizer.pre_inference_optimization()
        
        retry_count = 0
        max_retries = 3
        
        while retry_count <= max_retries:
            try:
                # Decode base64 audio data
                audio_bytes = base64.b64decode(audio_data)
                
                # Use the new audio decoding utility
                audio_array, sample_rate = decode_audio_data(audio_bytes, audio_format)
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                
                audio_duration = len(audio_array) / 16000
                
                # Set up VAD parameters with Korean optimization
                if vad_parameters is None:
                    if language == "ko":
                        # Use Korean-optimized VAD parameters with duration awareness
                        vad_parameters = self.korean_processor.get_korean_vad_parameters(audio_duration)
                        logger.debug(f"Using Korean-optimized VAD parameters for {audio_duration}s audio")
                    else:
                        # Default parameters for other languages
                        vad_parameters = {
                            "threshold": 0.5,
                            "min_speech_duration_ms": 250,
                            "max_speech_duration_s": float('inf'),
                            "min_silence_duration_ms": 2000,
                            "speech_pad_ms": 400,
                        }
                
                # Run transcription in executor
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._transcribe_sync,
                    audio_array,
                    language,
                    task,
                    beam_size,
                    word_timestamps,
                    vad_filter,
                    vad_parameters
                )
                
                processing_time = time.time() - start_time
                rtf = processing_time / audio_duration if audio_duration > 0 else 0
                
                # Apply Korean-specific post-processing if Korean language
                processed_text = result['text']
                processed_segments = result['segments']
                
                if language == "ko":
                    logger.debug("Applying Korean post-processing")
                    processed_text, processed_segments = self.korean_processor.post_process_transcription(
                        processed_text, processed_segments
                    )
                
                # Create performance metrics
                meets_targets = rtf <= 0.05 and processing_time * 1000 <= 1200
                performance_metrics = self.performance_optimizer.create_performance_metrics(
                    rtf=rtf,
                    latency_ms=processing_time * 1000,
                    audio_duration=audio_duration,
                    meets_targets=meets_targets,
                    error_count=0
                )
                
                # Apply post-inference cleanup
                self.performance_optimizer.post_inference_cleanup()
                
                # Create result
                stt_result = STTResult(
                    text=processed_text,
                    language=result['language'],
                    segments=processed_segments,
                    audio_duration=audio_duration,
                    processing_time=processing_time,
                    rtf=rtf,
                    model_info={
                        'model_size': self.model_size,
                        'device': self.device,
                        'compute_type': self.compute_type,
                        'korean_processing': language == "ko",
                        'performance_optimized': True
                    }
                )
                
                # Update statistics
                self._update_stats(audio_duration, processing_time)
                
                return stt_result
                
            except Exception as e:
                retry_count += 1
                
                # Use error handler to determine retry strategy
                should_retry, recovery_message = self.error_handler.handle_error(e, "transcribe_audio")
                
                if should_retry and retry_count <= max_retries:
                    logger.info(f"Retrying transcription (attempt {retry_count}): {recovery_message}")
                    await asyncio.sleep(0.1 * retry_count)  # Exponential backoff
                    continue
                else:
                    # Record failed metrics
                    audio_duration = 0  # Fallback duration
                    processing_time = time.time() - start_time
                    
                    failed_metrics = self.performance_optimizer.create_performance_metrics(
                        rtf=999.0,  # High RTF to indicate failure
                        latency_ms=processing_time * 1000,
                        audio_duration=audio_duration,
                        meets_targets=False,
                        error_count=1
                    )
                    
                    logger.error(f"Transcription failed after {retry_count} attempts: {e}")
                    raise STTServiceError(f"Transcription failed: {e}")
        
        # Should never reach here, but just in case
        raise STTServiceError("Maximum retry attempts exceeded")
    
    def _transcribe_sync(self, audio_array: np.ndarray, *args) -> Dict:
        """Synchronous transcription helper with GPU optimizations"""
        # Unpack arguments
        (language, task, beam_size, word_timestamps, vad_filter, vad_parameters) = args
        
        # Prepare base inference parameters
        base_inference_params = {
            'language': language,
            'task': task,
            'beam_size': beam_size,
            'word_timestamps': word_timestamps,
            'vad_filter': vad_filter,
            'vad_parameters': vad_parameters
        }
        
        # Apply RTX 4090 optimized inference parameters if GPU is available
        if self.rtx_4090_optimizations_applied:
            # Use advanced RTX 4090 inference configuration
            inference_params = get_rtx_4090_inference_params(base_inference_params)
            logger.debug("Using advanced RTX 4090 optimized inference parameters")
        elif self.gpu_optimizations_applied:
            # Fallback to basic GPU optimization
            inference_params = optimize_inference_for_rtx_4090(base_inference_params)
            logger.debug("Using basic RTX 4090 optimized inference parameters")
        else:
            inference_params = base_inference_params
        
        # Get Korean-optimized Whisper parameters if language is Korean
        korean_params = {}
        if language == "ko":
            korean_whisper_params = self.korean_processor.get_korean_whisper_parameters()
            # Apply Korean-specific parameters (only use if not explicitly overridden)
            korean_params.update({
                'patience': korean_whisper_params.get('patience', 1.0),
                'length_penalty': korean_whisper_params.get('length_penalty', 1.0),
                'repetition_penalty': korean_whisper_params.get('repetition_penalty', 1.1),
                'no_repeat_ngram_size': korean_whisper_params.get('no_repeat_ngram_size', 3),
                'temperature': korean_whisper_params.get('temperature', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                'compression_ratio_threshold': korean_whisper_params.get('compression_ratio_threshold', 2.4),
                'log_prob_threshold': korean_whisper_params.get('log_prob_threshold', -1.0),
                'no_speech_threshold': korean_whisper_params.get('no_speech_threshold', 0.6),
                'condition_on_previous_text': korean_whisper_params.get('condition_on_previous_text', True),
            })
            logger.debug(f"Using Korean-optimized Whisper parameters: {list(korean_params.keys())}")
        
        # Merge parameters: GPU optimized + Korean optimized
        final_params = {**inference_params, **korean_params}
        
        # Use GPU optimizer context for optimized inference
        if self.gpu_optimizations_applied:
            with self.gpu_optimizer.optimized_inference_context():
                segments, info = self.model.transcribe(audio_array, **final_params)
        else:
            segments, info = self.model.transcribe(audio_array, **final_params)
        
        # Convert generator to list for segments and extract text
        segments_list = []
        full_text = ""
        
        for segment in segments:
            segment_dict = {
                'id': segment.id,
                'seek': segment.seek,
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'tokens': segment.tokens,
                'temperature': segment.temperature,
                'avg_logprob': segment.avg_logprob,
                'compression_ratio': segment.compression_ratio,
                'no_speech_prob': segment.no_speech_prob
            }
            
            # Add word-level timestamps if available
            if hasattr(segment, 'words') and segment.words:
                segment_dict['words'] = [
                    {
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'probability': word.probability
                    }
                    for word in segment.words
                ]
            
            segments_list.append(segment_dict)
            full_text += segment.text
        
        return {
            'text': full_text.strip(),
            'language': info.language,
            'segments': segments_list
        }
    
    def get_stats(self) -> Dict:
        """Get service statistics"""
        with self.stats_lock:
            return self.stats.copy()
    
    def reset_stats(self):
        """Reset service statistics"""
        with self.stats_lock:
            self.stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_audio_duration': 0.0,
                'total_processing_time': 0.0,
                'avg_rtf': 0.0,
                'avg_latency_ms': 0.0
            }
        logger.info("Statistics reset")
    
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        return (self.is_initialized and 
                self.model is not None)
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        model_info = {
            'model_size': self.model_size,
            'device': self.device,
            'compute_type': self.compute_type,
            'model_path': self.model_path,
            'is_initialized': self.is_initialized,
            'is_healthy': self.is_healthy(),
            'gpu_optimizations_applied': getattr(self, 'gpu_optimizations_applied', False)
        }
        
        # Add GPU performance metrics if available
        if hasattr(self, 'rtx_4090_optimizer') and self.rtx_4090_optimizations_applied:
            # Use advanced RTX 4090 performance monitoring
            gpu_metrics = monitor_rtx_4090_performance()
            model_info['gpu_performance'] = gpu_metrics
            model_info['rtx_4090_optimizations_applied'] = True
        elif hasattr(self, 'gpu_optimizer') and self.gpu_optimizations_applied:
            # Fallback to basic GPU monitoring
            gpu_metrics = self.gpu_optimizer.monitor_gpu_performance()
            model_info['gpu_performance'] = gpu_metrics
            model_info['rtx_4090_optimizations_applied'] = False
        
        return model_info

    def _update_stats(self, audio_duration: float, processing_time: float):
        """Update service statistics."""
        with self.stats_lock:
            self.stats['successful_requests'] += 1
            self.stats['total_audio_duration'] += audio_duration
            self.stats['total_processing_time'] += processing_time
            
            # Calculate running averages
            if self.stats['successful_requests'] > 0:
                self.stats['avg_rtf'] = (self.stats['total_processing_time'] / 
                                       self.stats['total_audio_duration'])
                self.stats['avg_latency_ms'] = (self.stats['total_processing_time'] / 
                                              self.stats['successful_requests'] * 1000)

    async def transcribe_file_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = "ko",
        vad_filter: bool = True,
        beam_size: int = 5,
        word_timestamps: bool = False
    ) -> STTResult:
        """
        Transcribe audio file bytes using Faster Whisper model.
        
        Args:
            audio_bytes: Raw audio file bytes (WAV, MP3, FLAC, etc.)
            language: Language code ('ko', 'en', None for auto-detect)
            vad_filter: Whether to use voice activity detection
            beam_size: Beam size for decoding
            word_timestamps: Whether to return word-level timestamps
            
        Returns:
            STTResult with transcription and metadata
        """
        # Auto-initialize model if not loaded
        if not self.model:
            logger.info("Model not initialized, starting initialization...")
            try:
                initialization_success = await self.initialize()
                if not initialization_success:
                    raise STTServiceError(f"STT model initialization failed: {self.initialization_error}")
            except Exception as e:
                raise STTServiceError(f"STT model initialization failed: {str(e)}")
        
        start_time = time.time()
        
        # Apply pre-inference optimizations
        self.performance_optimizer.pre_inference_optimization()
        
        try:
            # Use the audio decoding utility to handle file bytes
            audio_array, sample_rate = decode_audio_data(audio_bytes, "auto")
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
            audio_duration = len(audio_array) / 16000
            
            # Set up VAD parameters with Korean optimization
            if language == "ko":
                # Use Korean-optimized VAD parameters with duration awareness
                vad_parameters = self.korean_processor.get_korean_vad_parameters(audio_duration)
                logger.debug(f"Using Korean-optimized VAD parameters for {audio_duration}s audio")
            else:
                # Default parameters for other languages
                vad_parameters = {
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "max_speech_duration_s": float('inf'),
                    "min_silence_duration_ms": 2000,
                    "speech_pad_ms": 400,
                }
            
            # Run transcription in executor
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._transcribe_sync,
                audio_array,
                language,
                'transcribe',
                beam_size,
                word_timestamps,
                vad_filter,
                vad_parameters
            )
            
            processing_time = time.time() - start_time
            rtf = processing_time / audio_duration if audio_duration > 0 else 0
            
            # Apply Korean-specific post-processing if Korean language
            processed_text = result['text']
            processed_segments = result['segments']
            
            if language == "ko":
                logger.debug("Applying Korean post-processing")
                processed_text, processed_segments = self.korean_processor.post_process_transcription(
                    processed_text, processed_segments
                )
            
            # Create performance metrics
            meets_targets = rtf <= 0.05 and processing_time * 1000 <= 1200
            performance_metrics = self.performance_optimizer.create_performance_metrics(
                rtf=rtf,
                latency_ms=processing_time * 1000,
                audio_duration=audio_duration,
                meets_targets=meets_targets,
                error_count=0
            )
            
            # Apply post-inference cleanup
            self.performance_optimizer.post_inference_cleanup()
            
            # Create result
            stt_result = STTResult(
                text=processed_text,
                language=result['language'],
                segments=processed_segments,
                audio_duration=audio_duration,
                processing_time=processing_time,
                rtf=rtf,
                model_info={
                    'model_size': self.model_size,
                    'device': self.device,
                    'compute_type': self.compute_type,
                    'korean_processing': language == "ko",
                    'performance_optimized': True
                }
            )
            
            # Update statistics
            self._update_stats(audio_duration, processing_time)
            
            return stt_result
            
        except Exception as e:
            # Record failed metrics
            audio_duration = 0  # Fallback duration
            processing_time = time.time() - start_time
            
            failed_metrics = self.performance_optimizer.create_performance_metrics(
                rtf=999.0,  # High RTF to indicate failure
                latency_ms=processing_time * 1000,
                audio_duration=audio_duration,
                meets_targets=False,
                error_count=1
            )
            
            logger.error(f"File transcription failed: {e}")
            raise STTServiceError(f"File transcription failed: {e}")


# Global STT service instance
_stt_service: Optional[FasterWhisperSTTService] = None


def get_stt_service() -> FasterWhisperSTTService:
    """Get global STT service instance"""
    global _stt_service
    if _stt_service is None:
        _stt_service = FasterWhisperSTTService()
    return _stt_service


async def initialize_stt_service(**kwargs) -> bool:
    """Initialize global STT service"""
    service = get_stt_service()
    return await service.initialize()


async def shutdown_stt_service():
    """Shutdown global STT service"""
    global _stt_service
    if _stt_service:
        await _stt_service.shutdown()
        _stt_service = None


# Convenience functions
async def transcribe_audio_data(audio_data: str, 
                               audio_format: str, 
                               language: Optional[str] = None,
                               **kwargs) -> STTResult:
    """Convenience function for transcribing audio data"""
    service = get_stt_service()
    return await service.transcribe_audio(audio_data, audio_format, language, **kwargs)


if __name__ == "__main__":
    # Test the STT service
    import asyncio
    
    async def test_stt_service():
        logging.basicConfig(level=logging.INFO)
        
        # Initialize service
        service = FasterWhisperSTTService(model_size="tiny", device="auto")
        success = await service.initialize()
        
        if not success:
            print("❌ Service initialization failed")
            return
        
        # Create test audio
        print("Creating test audio...")
        import base64
        audio_array = np.random.random(32000).astype(np.float32) * 0.1  # 2 seconds at 16kHz
        audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Transcribe
        print("Running transcription...")
        try:
            result = await service.transcribe_audio(
                base64_audio, 'pcm_16khz', language='ko'
            )
            
            print(f"✅ Transcription result: {result}")
            print(f"Stats: {service.get_stats()}")
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
        
        # Shutdown
        await service.shutdown()
    
    # Run test
    asyncio.run(test_stt_service()) 