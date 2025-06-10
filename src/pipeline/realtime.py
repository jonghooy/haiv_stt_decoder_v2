#!/usr/bin/env python3
"""
Real-time Audio Processing Pipeline
Optimized for GPU inference with minimal latency
"""

import asyncio
import threading
import time
import numpy as np
from collections import deque
from typing import AsyncGenerator, Callable, Optional, Dict, Any, List
import logging
from dataclasses import dataclass
from enum import Enum
import queue

from core.config import STTConfig, DeviceType, ComputeType
from models.whisper_model import WhisperModelManager

logger = logging.getLogger(__name__)


class StreamStatus(Enum):
    """Stream processing status"""
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class AudioChunk:
    """Audio chunk with metadata"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    duration: float
    chunk_id: int
    is_silent: bool = False


@dataclass
class TranscriptionResult:
    """Real-time transcription result"""
    text: str
    confidence: float
    start_time: float
    end_time: float
    chunk_id: int
    language: str
    is_partial: bool = False
    processing_time: float = 0.0


class AudioBuffer:
    """Thread-safe audio buffer for real-time streaming"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_duration: float = 1.0,
                 overlap_duration: float = 0.2,
                 max_buffer_duration: float = 30.0):
        """
        Initialize audio buffer
        
        Args:
            sample_rate: Audio sample rate
            chunk_duration: Duration of each processing chunk in seconds
            overlap_duration: Overlap between chunks to avoid word cutting
            max_buffer_duration: Maximum buffer duration before dropping old data
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.max_buffer_duration = max_buffer_duration
        
        # Calculate sizes
        self.chunk_size = int(chunk_duration * sample_rate)
        self.overlap_size = int(overlap_duration * sample_rate)
        self.max_buffer_size = int(max_buffer_duration * sample_rate)
        
        # Thread-safe buffer
        self._buffer = deque(maxlen=self.max_buffer_size)
        self._lock = threading.RLock()
        self._chunk_counter = 0
        
        # Statistics
        self.total_samples_added = 0
        self.total_chunks_created = 0
        
        logger.info(f"AudioBuffer initialized: chunk_size={self.chunk_size}, "
                   f"overlap_size={self.overlap_size}, max_buffer_size={self.max_buffer_size}")
    
    def add_audio(self, audio_data: np.ndarray) -> None:
        """Add audio data to buffer"""
        with self._lock:
            # Ensure audio_data is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Add to buffer
            self._buffer.extend(audio_data)
            self.total_samples_added += len(audio_data)
            
            # Limit buffer size
            if len(self._buffer) > self.max_buffer_size:
                excess = len(self._buffer) - self.max_buffer_size
                for _ in range(excess):
                    self._buffer.popleft()
    
    def get_chunk(self) -> Optional[AudioChunk]:
        """Get next audio chunk for processing"""
        with self._lock:
            if len(self._buffer) < self.chunk_size:
                return None
            
            # Extract chunk with overlap
            start_idx = max(0, self.chunk_size - self.overlap_size) if self.total_chunks_created > 0 else 0
            
            if len(self._buffer) < start_idx + self.chunk_size:
                return None
            
            # Get chunk data
            chunk_data = np.array(list(self._buffer)[start_idx:start_idx + self.chunk_size])
            
            # Remove processed data (keep overlap)
            remove_size = self.chunk_size - self.overlap_size
            for _ in range(min(remove_size, len(self._buffer))):
                self._buffer.popleft()
            
            # Create chunk
            chunk = AudioChunk(
                data=chunk_data,
                timestamp=time.time(),
                sample_rate=self.sample_rate,
                duration=self.chunk_duration,
                chunk_id=self._chunk_counter,
                is_silent=self._is_silent(chunk_data)
            )
            
            self._chunk_counter += 1
            self.total_chunks_created += 1
            
            return chunk
    
    def _is_silent(self, audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if audio chunk is silent"""
        return np.max(np.abs(audio_data)) < threshold
    
    def clear(self) -> None:
        """Clear buffer"""
        with self._lock:
            self._buffer.clear()
            self._chunk_counter = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self._lock:
            return {
                "buffer_size": len(self._buffer),
                "buffer_duration": len(self._buffer) / self.sample_rate,
                "total_samples_added": self.total_samples_added,
                "total_chunks_created": self.total_chunks_created,
                "chunk_counter": self._chunk_counter
            }


class StreamProcessor:
    """GPU-optimized real-time stream processor"""
    
    def __init__(self, 
                 model_manager: WhisperModelManager,
                 result_callback: Optional[Callable[[TranscriptionResult], None]] = None):
        """
        Initialize stream processor
        
        Args:
            model_manager: Whisper model manager for inference
            result_callback: Callback for transcription results
        """
        self.model_manager = model_manager
        self.result_callback = result_callback
        
        # Processing queue
        self._processing_queue = queue.Queue(maxsize=100)
        self._result_queue = queue.Queue()
        
        # Worker threads
        self._worker_thread = None
        self._result_thread = None
        self._stop_event = threading.Event()
        
        # Statistics
        self.total_chunks_processed = 0
        self.total_processing_time = 0.0
        self.average_rtf = 0.0
        
        logger.info("StreamProcessor initialized")
    
    def start(self) -> None:
        """Start processing threads"""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        
        self._stop_event.clear()
        
        # Start worker thread for inference
        self._worker_thread = threading.Thread(target=self._process_worker, daemon=True)
        self._worker_thread.start()
        
        # Start result handler thread
        self._result_thread = threading.Thread(target=self._result_worker, daemon=True)
        self._result_thread.start()
        
        logger.info("StreamProcessor started")
    
    def stop(self) -> None:
        """Stop processing threads"""
        self._stop_event.set()
        
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        if self._result_thread:
            self._result_thread.join(timeout=5.0)
        
        logger.info("StreamProcessor stopped")
    
    def submit_chunk(self, chunk: AudioChunk) -> bool:
        """Submit audio chunk for processing"""
        try:
            self._processing_queue.put_nowait(chunk)
            return True
        except queue.Full:
            logger.warning("Processing queue full, dropping chunk")
            return False
    
    def _process_worker(self) -> None:
        """Worker thread for audio processing"""
        logger.info("Processing worker started")
        
        while not self._stop_event.is_set():
            try:
                # Get chunk from queue
                chunk = self._processing_queue.get(timeout=1.0)
                
                # Skip silent chunks for efficiency
                if chunk.is_silent:
                    logger.debug(f"Skipping silent chunk {chunk.chunk_id}")
                    continue
                
                # Process chunk
                start_time = time.time()
                result = self._process_chunk(chunk)
                processing_time = time.time() - start_time
                
                # Update statistics
                self.total_chunks_processed += 1
                self.total_processing_time += processing_time
                self.average_rtf = (self.total_processing_time / self.total_chunks_processed) / chunk.duration
                
                if result:
                    result.processing_time = processing_time
                    self._result_queue.put(result)
                
                # Log performance
                rtf = processing_time / chunk.duration
                logger.debug(f"Processed chunk {chunk.chunk_id} in {processing_time:.3f}s (RTF: {rtf:.3f}x)")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing chunk: {e}", exc_info=True)
        
        logger.info("Processing worker stopped")
    
    def _process_chunk(self, chunk: AudioChunk) -> Optional[TranscriptionResult]:
        """Process single audio chunk"""
        try:
            # Transcribe using GPU-optimized model
            segments, info = self.model_manager.transcribe(
                chunk.data,
                beam_size=1,  # Fast inference
                language="en",  # Fixed language for speed
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(min_silence_duration_ms=300)
            )
            
            # Extract text from segments
            segments_list = list(segments)
            if not segments_list:
                return None
            
            # Combine segments text
            if hasattr(segments_list[0], 'text'):
                text = " ".join([s.text for s in segments_list]).strip()
            elif isinstance(segments_list[0], dict) and 'text' in segments_list[0]:
                text = " ".join([s['text'] for s in segments_list]).strip()
            else:
                logger.warning(f"Unknown segment format: {type(segments_list[0])}")
                return None
            
            if not text or text == ".":  # Skip empty or noise-only results
                return None
            
            # Calculate confidence (average of segment probabilities)
            if hasattr(segments_list[0], 'avg_logprob'):
                confidence = np.mean([max(0, min(1, np.exp(s.avg_logprob))) for s in segments_list])
            else:
                confidence = 0.8  # Default confidence
            
            # Get language info
            if hasattr(info, 'language'):
                language = info.language
            elif isinstance(info, dict) and 'language' in info:
                language = info['language']
            else:
                language = "en"
            
            # Create result
            result = TranscriptionResult(
                text=text,
                confidence=confidence,
                start_time=chunk.timestamp,
                end_time=chunk.timestamp + chunk.duration,
                chunk_id=chunk.chunk_id,
                language=language,
                is_partial=False
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing chunk {chunk.chunk_id}: {e}")
            return None
    
    def _result_worker(self) -> None:
        """Worker thread for result handling"""
        logger.info("Result worker started")
        
        while not self._stop_event.is_set():
            try:
                result = self._result_queue.get(timeout=1.0)
                
                if self.result_callback:
                    self.result_callback(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error handling result: {e}", exc_info=True)
        
        logger.info("Result worker stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "total_chunks_processed": self.total_chunks_processed,
            "total_processing_time": self.total_processing_time,
            "average_rtf": self.average_rtf,
            "queue_size": self._processing_queue.qsize(),
            "result_queue_size": self._result_queue.qsize()
        }


class RealtimeSTT:
    """Main real-time STT engine optimized for GPU performance"""
    
    def __init__(self, 
                 config: Optional[STTConfig] = None,
                 result_callback: Optional[Callable[[TranscriptionResult], None]] = None):
        """
        Initialize real-time STT engine
        
        Args:
            config: STT configuration (uses optimized GPU settings if None)
            result_callback: Callback for transcription results
        """
        # Use optimized GPU configuration if none provided
        if config is None:
            config = STTConfig()
            config.model.device = DeviceType.CUDA
            config.model.compute_type = ComputeType.FLOAT16  # Fastest GPU mode
            config.model.beam_size = 1  # Fast inference
            config.model.local_files_only = True
        
        self.config = config
        self.result_callback = result_callback
        
        # Initialize components
        self.model_manager = WhisperModelManager(config)
        self.audio_buffer = AudioBuffer(
            sample_rate=config.inference.sample_rate,
            chunk_duration=1.0,  # 1 second chunks for real-time
            overlap_duration=0.2,  # 200ms overlap
            max_buffer_duration=30.0  # 30 second buffer
        )
        self.stream_processor = StreamProcessor(
            self.model_manager, 
            self._handle_result
        )
        
        # State management
        self.status = StreamStatus.IDLE
        self._processing_thread = None
        self._stop_event = threading.Event()
        
        # Results storage
        self.results: List[TranscriptionResult] = []
        self._results_lock = threading.Lock()
        
        logger.info("RealtimeSTT engine initialized")
    
    async def start(self) -> None:
        """Start real-time processing"""
        if self.status != StreamStatus.IDLE:
            logger.warning("Engine already started")
            return
        
        logger.info("Starting real-time STT engine...")
        
        # Load model
        model_info = self.model_manager.load_model()
        logger.info(f"Model loaded: {model_info.config.get('model_size_or_path', 'unknown')}")
        
        # Start processing components
        self.stream_processor.start()
        
        # Start chunk processing thread
        self._stop_event.clear()
        self._processing_thread = threading.Thread(target=self._chunk_processor, daemon=True)
        self._processing_thread.start()
        
        self.status = StreamStatus.RECORDING
        logger.info("Real-time STT engine started")
    
    async def stop(self) -> None:
        """Stop real-time processing"""
        if self.status == StreamStatus.IDLE:
            return
        
        logger.info("Stopping real-time STT engine...")
        
        self.status = StreamStatus.IDLE
        self._stop_event.set()
        
        # Stop processing components
        self.stream_processor.stop()
        
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
        
        # Unload model
        self.model_manager.unload_model()
        
        logger.info("Real-time STT engine stopped")
    
    def add_audio(self, audio_data: np.ndarray) -> None:
        """Add audio data for real-time processing"""
        if self.status != StreamStatus.RECORDING:
            logger.warning("Engine not recording")
            return
        
        self.audio_buffer.add_audio(audio_data)
    
    def _chunk_processor(self) -> None:
        """Background thread to process audio chunks"""
        logger.info("Chunk processor started")
        
        while not self._stop_event.is_set():
            try:
                # Get next chunk
                chunk = self.audio_buffer.get_chunk()
                if chunk is None:
                    time.sleep(0.01)  # Small delay to avoid busy waiting
                    continue
                
                # Submit for processing
                if not self.stream_processor.submit_chunk(chunk):
                    logger.warning(f"Failed to submit chunk {chunk.chunk_id}")
                
            except Exception as e:
                logger.error(f"Error in chunk processor: {e}", exc_info=True)
                time.sleep(0.1)
        
        logger.info("Chunk processor stopped")
    
    def _handle_result(self, result: TranscriptionResult) -> None:
        """Handle transcription result"""
        with self._results_lock:
            self.results.append(result)
            
            # Keep only recent results (last 100)
            if len(self.results) > 100:
                self.results = self.results[-100:]
        
        # Call user callback
        if self.result_callback:
            try:
                self.result_callback(result)
            except Exception as e:
                logger.error(f"Error in result callback: {e}")
        
        # Log result
        rtf_text = f"RTF: {result.processing_time/1.0:.3f}x" if result.processing_time > 0 else ""
        logger.info(f"Result [{result.chunk_id}]: '{result.text}' "
                   f"(confidence: {result.confidence:.2f}, {rtf_text})")
    
    def get_recent_results(self, count: int = 10) -> List[TranscriptionResult]:
        """Get recent transcription results"""
        with self._results_lock:
            return self.results[-count:] if self.results else []
    
    def get_full_text(self, separator: str = " ") -> str:
        """Get full transcribed text"""
        with self._results_lock:
            return separator.join([r.text for r in self.results if r.text])
    
    def clear_results(self) -> None:
        """Clear stored results"""
        with self._results_lock:
            self.results.clear()
        self.audio_buffer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        buffer_stats = self.audio_buffer.get_stats()
        processor_stats = self.stream_processor.get_stats()
        
        with self._results_lock:
            result_count = len(self.results)
        
        return {
            "status": self.status.value,
            "model_info": self.model_manager.get_model_info(),
            "buffer": buffer_stats,
            "processor": processor_stats,
            "results": {
                "total_results": result_count,
                "recent_results": min(10, result_count)
            }
        }


# Utility functions for testing and demonstration

def create_test_audio_stream(duration: float = 10.0, 
                           sample_rate: int = 16000) -> np.ndarray:
    """Create test audio stream with speech-like characteristics"""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # Create speech-like signal
    signal = np.zeros(samples)
    frequencies = [200, 400, 800, 1600]  # Speech frequencies
    
    for freq in frequencies:
        amplitude = np.random.uniform(0.1, 0.3)
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add noise
    noise = np.random.normal(0, 0.05, samples)
    signal += noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal.astype(np.float32)


async def demo_realtime_stt():
    """Demonstration of real-time STT"""
    print("🎙️ Real-time STT Demo")
    print("=" * 50)
    
    # Results callback
    def on_result(result: TranscriptionResult):
        print(f"📝 [{result.chunk_id:03d}] {result.text} "
              f"(conf: {result.confidence:.2f}, "
              f"RTF: {result.processing_time/1.0:.3f}x)")
    
    # Initialize engine
    engine = RealtimeSTT(result_callback=on_result)
    
    try:
        # Start engine
        await engine.start()
        print("🚀 Engine started, processing audio...")
        
        # Simulate real-time audio stream
        test_audio = create_test_audio_stream(duration=10.0)
        chunk_size = 1600  # 100ms chunks at 16kHz
        
        for i in range(0, len(test_audio), chunk_size):
            chunk = test_audio[i:i+chunk_size]
            engine.add_audio(chunk)
            
            # Simulate real-time streaming delay
            await asyncio.sleep(0.1)  # 100ms delay
        
        # Wait for processing to complete
        await asyncio.sleep(3.0)
        
        # Show final statistics
        stats = engine.get_stats()
        print("\n📊 Final Statistics:")
        print(f"Status: {stats['status']}")
        print(f"Buffer: {stats['buffer']['total_chunks_created']} chunks created")
        print(f"Processor: {stats['processor']['total_chunks_processed']} chunks processed")
        print(f"Average RTF: {stats['processor']['average_rtf']:.3f}x")
        print(f"Results: {stats['results']['total_results']} transcriptions")
        
        # Show full text
        full_text = engine.get_full_text()
        print(f"\n📜 Full transcription: '{full_text}'")
        
    finally:
        await engine.stop()
        print("🛑 Engine stopped")


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_realtime_stt()) 