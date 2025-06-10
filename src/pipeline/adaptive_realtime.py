#!/usr/bin/env python3
"""
Adaptive Real-time STT Pipeline
Balances performance and latency for real-world usage
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

import sys
sys.path.append('/home/jonghooy/haiv_stt_decoder_v2/src')

from core.config import STTConfig, DeviceType, ComputeType
from models.whisper_model import WhisperModelManager

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing mode based on chunk characteristics"""
    IMMEDIATE = "immediate"  # < 1s - process immediately 
    OPTIMAL = "optimal"      # 1-3s - balance performance/latency
    BATCH = "batch"          # > 3s - optimize for performance


@dataclass
class AudioChunk:
    """Audio chunk with metadata"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    duration: float
    chunk_id: int
    is_silent: bool = False
    is_final: bool = False  # Final chunk (don't wait for more)


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
    processing_mode: ProcessingMode = ProcessingMode.OPTIMAL


class AdaptiveAudioBuffer:
    """Adaptive audio buffer that optimizes based on input patterns"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 max_wait_time: float = 1.5,  # Maximum wait time for optimization
                 min_chunk_duration: float = 0.5,  # Always process if this small
                 optimal_chunk_duration: float = 3.0):  # Target for best performance
        """
        Initialize adaptive audio buffer
        
        Args:
            sample_rate: Audio sample rate
            max_wait_time: Maximum time to wait for optimal chunk size
            min_chunk_duration: Minimum chunk size to process immediately
            optimal_chunk_duration: Target chunk size for best RTF
        """
        self.sample_rate = sample_rate
        self.max_wait_time = max_wait_time
        self.min_chunk_duration = min_chunk_duration
        self.optimal_chunk_duration = optimal_chunk_duration
        
        # Calculate sizes
        self.min_chunk_size = int(min_chunk_duration * sample_rate)
        self.optimal_chunk_size = int(optimal_chunk_duration * sample_rate)
        self.max_buffer_size = int(30.0 * sample_rate)  # 30s max buffer
        
        # Buffer management
        self._buffer = np.zeros(self.max_buffer_size, dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._lock = threading.RLock()
        self._chunk_counter = 0
        self._last_chunk_time = time.time()
        
        # Adaptive parameters
        self._waiting_for_optimal = False
        self._wait_start_time = 0.0
        
        logger.info(f"AdaptiveBuffer: min={min_chunk_duration}s, optimal={optimal_chunk_duration}s, wait={max_wait_time}s")
    
    def add_audio(self, audio_data: np.ndarray, is_final: bool = False) -> None:
        """Add audio data to buffer"""
        with self._lock:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            data_len = len(audio_data)
            end_pos = self._write_pos + data_len
            
            if end_pos <= self.max_buffer_size:
                self._buffer[self._write_pos:end_pos] = audio_data
            else:
                first_part = self.max_buffer_size - self._write_pos
                self._buffer[self._write_pos:] = audio_data[:first_part]
                self._buffer[:data_len - first_part] = audio_data[first_part:]
            
            self._write_pos = end_pos % self.max_buffer_size
            self._last_chunk_time = time.time()
            
            # Mark if this is final chunk
            if is_final:
                self._waiting_for_optimal = False
    
    def get_chunk(self, force_immediate: bool = False) -> Optional[AudioChunk]:
        """Get next audio chunk with adaptive sizing"""
        with self._lock:
            available = self._get_available_samples()
            
            if available == 0:
                return None
            
            # Determine processing strategy
            current_time = time.time()
            
            # Strategy 1: Force immediate (timeout or final chunk)
            if force_immediate and available >= self.min_chunk_size:
                return self._extract_chunk(available, ProcessingMode.IMMEDIATE)
            
            # Strategy 2: Have optimal size - process immediately  
            if available >= self.optimal_chunk_size:
                self._waiting_for_optimal = False
                return self._extract_chunk(self.optimal_chunk_size, ProcessingMode.BATCH)
            
            # Strategy 3: Have reasonable size (1-3s) and not waiting
            reasonable_size = int(1.0 * self.sample_rate)
            if available >= reasonable_size and not self._waiting_for_optimal:
                if available <= self.optimal_chunk_size:
                    return self._extract_chunk(available, ProcessingMode.OPTIMAL)
                else:
                    return self._extract_chunk(self.optimal_chunk_size, ProcessingMode.BATCH)
            
            # Strategy 4: Wait for optimal size if time allows
            if available >= self.min_chunk_size:
                if not self._waiting_for_optimal:
                    self._waiting_for_optimal = True
                    self._wait_start_time = current_time
                    return None  # Wait for more data
                
                # Check if we've waited long enough
                wait_time = current_time - self._wait_start_time
                if wait_time >= self.max_wait_time:
                    self._waiting_for_optimal = False
                    return self._extract_chunk(available, ProcessingMode.IMMEDIATE)
            
            return None
    
    def _get_available_samples(self) -> int:
        """Get number of available samples"""
        if self._write_pos >= self._read_pos:
            return self._write_pos - self._read_pos
        else:
            return self.max_buffer_size - self._read_pos + self._write_pos
    
    def _extract_chunk(self, chunk_size: int, mode: ProcessingMode) -> AudioChunk:
        """Extract chunk from buffer"""
        chunk_data = np.zeros(chunk_size, dtype=np.float32)
        end_pos = self._read_pos + chunk_size
        
        if end_pos <= self.max_buffer_size:
            chunk_data[:] = self._buffer[self._read_pos:end_pos]
        else:
            first_part = self.max_buffer_size - self._read_pos
            chunk_data[:first_part] = self._buffer[self._read_pos:]
            chunk_data[first_part:] = self._buffer[:chunk_size - first_part]
        
        # Update read position (no overlap for simplicity)
        self._read_pos = end_pos % self.max_buffer_size
        
        duration = chunk_size / self.sample_rate
        
        chunk = AudioChunk(
            data=chunk_data,
            timestamp=time.time(),
            sample_rate=self.sample_rate,
            duration=duration,
            chunk_id=self._chunk_counter,
            is_silent=self._is_silent(chunk_data)
        )
        
        self._chunk_counter += 1
        logger.debug(f"Extracted {duration:.1f}s chunk (mode: {mode.value})")
        
        return chunk
    
    def _is_silent(self, audio_data: np.ndarray, threshold: float = 0.003) -> bool:
        """Fast silence detection"""
        return np.max(np.abs(audio_data)) < threshold
    
    def force_flush(self) -> Optional[AudioChunk]:
        """Force flush remaining buffer (for end of stream)"""
        return self.get_chunk(force_immediate=True)


class AdaptiveProcessor:
    """Adaptive processor that optimizes based on chunk characteristics"""
    
    def __init__(self, 
                 model_manager: WhisperModelManager,
                 result_callback: Optional[Callable[[TranscriptionResult], None]] = None):
        """Initialize adaptive processor"""
        self.model_manager = model_manager
        self.result_callback = result_callback
        
        # Processing queue
        self._processing_queue = queue.Queue(maxsize=20)
        self._result_queue = queue.Queue(maxsize=20)
        
        # Worker threads
        self._worker_thread = None
        self._result_thread = None
        self._stop_event = threading.Event()
        
        # Performance tracking
        self.stats = {
            'immediate_mode': {'count': 0, 'total_time': 0.0, 'total_rtf': 0.0},
            'optimal_mode': {'count': 0, 'total_time': 0.0, 'total_rtf': 0.0},
            'batch_mode': {'count': 0, 'total_time': 0.0, 'total_rtf': 0.0}
        }
        self._stats_lock = threading.Lock()
        
        logger.info("AdaptiveProcessor initialized")
    
    def start(self) -> None:
        """Start processing threads"""
        if self._worker_thread:
            return
        
        self._stop_event.clear()
        
        self._worker_thread = threading.Thread(target=self._adaptive_worker, daemon=True)
        self._worker_thread.start()
        
        self._result_thread = threading.Thread(target=self._result_worker, daemon=True)
        self._result_thread.start()
        
        logger.info("AdaptiveProcessor started")
    
    def stop(self) -> None:
        """Stop processing threads"""
        self._stop_event.set()
        
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
        if self._result_thread:
            self._result_thread.join(timeout=2.0)
        
        logger.info("AdaptiveProcessor stopped")
    
    def submit_chunk(self, chunk: AudioChunk, mode: ProcessingMode) -> bool:
        """Submit chunk for adaptive processing"""
        try:
            self._processing_queue.put_nowait((chunk, mode))
            return True
        except queue.Full:
            logger.warning("Processing queue full")
            return False
    
    def _adaptive_worker(self) -> None:
        """Adaptive worker that adjusts parameters based on mode"""
        logger.info("Adaptive worker started")
        
        while not self._stop_event.is_set():
            try:
                chunk, mode = self._processing_queue.get(timeout=0.3)
                
                if chunk.is_silent:
                    continue
                
                start_time = time.perf_counter()
                result = self._process_chunk_adaptive(chunk, mode)
                processing_time = time.perf_counter() - start_time
                
                # Update statistics
                with self._stats_lock:
                    mode_key = f"{mode.value}_mode"
                    if mode_key in self.stats:
                        self.stats[mode_key]['count'] += 1
                        self.stats[mode_key]['total_time'] += processing_time
                        rtf = processing_time / chunk.duration
                        self.stats[mode_key]['total_rtf'] += rtf
                
                if result:
                    result.processing_time = processing_time
                    result.processing_mode = mode
                    try:
                        self._result_queue.put_nowait(result)
                    except queue.Full:
                        logger.warning("Result queue full")
                
                rtf = processing_time / chunk.duration
                logger.debug(f"Processed {chunk.duration:.1f}s chunk (mode: {mode.value}, RTF: {rtf:.3f}x)")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Adaptive worker error: {e}")
        
        logger.info("Adaptive worker stopped")
    
    def _process_chunk_adaptive(self, chunk: AudioChunk, mode: ProcessingMode) -> Optional[TranscriptionResult]:
        """Process chunk with mode-specific optimizations"""
        try:
            # Adaptive parameters based on mode
            if mode == ProcessingMode.IMMEDIATE:
                # Fast processing for small chunks
                beam_size = 1
                vad_filter = False
                compression_ratio_threshold = 1000.0
            elif mode == ProcessingMode.OPTIMAL:
                # Balanced processing for medium chunks
                beam_size = 1
                vad_filter = False
                compression_ratio_threshold = 2.4
            else:  # BATCH mode
                # Optimized processing for large chunks
                beam_size = 1
                vad_filter = False
                compression_ratio_threshold = 2.4
            
            segments, info = self.model_manager.transcribe(
                chunk.data,
                beam_size=beam_size,
                language="en",
                vad_filter=vad_filter,
                temperature=0.0,
                compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                without_timestamps=True,
                word_timestamps=False
            )
            
            # Extract text
            segments_list = list(segments)
            if not segments_list:
                return None
            
            text = ""
            for s in segments_list:
                if hasattr(s, 'text'):
                    text += s.text
                elif isinstance(s, dict) and 'text' in s:
                    text += s['text']
            
            text = text.strip()
            if not text or len(text) < 2:
                return None
            
            return TranscriptionResult(
                text=text,
                confidence=0.8,
                start_time=chunk.timestamp,
                end_time=chunk.timestamp + chunk.duration,
                chunk_id=chunk.chunk_id,
                language="en",
                is_partial=False,
                processing_mode=mode
            )
            
        except Exception as e:
            logger.error(f"Adaptive processing error: {e}")
            return None
    
    def _result_worker(self) -> None:
        """Result handler"""
        while not self._stop_event.is_set():
            try:
                result = self._result_queue.get(timeout=0.3)
                if self.result_callback:
                    self.result_callback(result)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Result handler error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adaptive processing statistics"""
        with self._stats_lock:
            stats_copy = {}
            for mode, data in self.stats.items():
                if data['count'] > 0:
                    avg_time = data['total_time'] / data['count']
                    avg_rtf = data['total_rtf'] / data['count']
                    stats_copy[mode] = {
                        'count': data['count'],
                        'avg_processing_time': avg_time,
                        'avg_rtf': avg_rtf
                    }
                else:
                    stats_copy[mode] = {'count': 0, 'avg_processing_time': 0.0, 'avg_rtf': 0.0}
            
            return stats_copy


class AdaptiveRealtimeSTT:
    """Adaptive real-time STT that balances performance and latency"""
    
    def __init__(self, 
                 config: Optional[STTConfig] = None,
                 result_callback: Optional[Callable[[TranscriptionResult], None]] = None):
        """Initialize adaptive real-time STT"""
        
        if config is None:
            config = STTConfig()
            config.model.device = DeviceType.CUDA
            config.model.compute_type = ComputeType.FLOAT16
            config.model.beam_size = 1
            config.model.local_files_only = True
        
        self.config = config
        self.result_callback = result_callback
        
        # Adaptive components
        self.model_manager = WhisperModelManager(config)
        self.audio_buffer = AdaptiveAudioBuffer(
            max_wait_time=1.5,  # 1.5s max wait for optimization
            min_chunk_duration=0.5,  # Process 0.5s+ immediately if needed
            optimal_chunk_duration=3.0  # Target 3s for best RTF
        )
        self.processor = AdaptiveProcessor(self.model_manager, self._handle_result)
        
        # State
        self.status = "idle"
        self._processing_thread = None
        self._timeout_thread = None
        self._stop_event = threading.Event()
        
        # Results
        self.results: List[TranscriptionResult] = []
        self._results_lock = threading.Lock()
        
        logger.info("AdaptiveRealtimeSTT initialized")
    
    async def start(self) -> None:
        """Start adaptive processing"""
        if self.status != "idle":
            return
        
        logger.info("Starting adaptive real-time STT...")
        
        # Load model
        model_info = self.model_manager.load_model()
        logger.info(f"Model loaded: {model_info.config.get('model_size_or_path', 'unknown')}")
        
        # Start processor
        self.processor.start()
        
        # Start chunk processor
        self._stop_event.clear()
        self._processing_thread = threading.Thread(target=self._adaptive_chunk_processor, daemon=True)
        self._processing_thread.start()
        
        # Start timeout handler
        self._timeout_thread = threading.Thread(target=self._timeout_handler, daemon=True)
        self._timeout_thread.start()
        
        self.status = "recording"
        logger.info("Adaptive real-time STT started")
    
    async def stop(self) -> None:
        """Stop adaptive processing"""
        if self.status == "idle":
            return
        
        logger.info("Stopping adaptive real-time STT...")
        
        self.status = "idle"
        self._stop_event.set()
        
        # Process remaining chunks
        final_chunk = self.audio_buffer.force_flush()
        if final_chunk:
            self.processor.submit_chunk(final_chunk, ProcessingMode.IMMEDIATE)
        
        self.processor.stop()
        
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        if self._timeout_thread:
            self._timeout_thread.join(timeout=2.0)
        
        self.model_manager.unload_model()
        logger.info("Adaptive real-time STT stopped")
    
    def add_audio(self, audio_data: np.ndarray, is_final: bool = False) -> None:
        """Add audio with adaptive processing"""
        if self.status == "recording":
            self.audio_buffer.add_audio(audio_data, is_final)
    
    def _adaptive_chunk_processor(self) -> None:
        """Adaptive chunk processor"""
        logger.info("Adaptive chunk processor started")
        
        while not self._stop_event.is_set():
            try:
                chunk = self.audio_buffer.get_chunk()
                if chunk is None:
                    time.sleep(0.01)
                    continue
                
                # Determine processing mode based on chunk duration
                if chunk.duration <= 1.0:
                    mode = ProcessingMode.IMMEDIATE
                elif chunk.duration <= 3.0:
                    mode = ProcessingMode.OPTIMAL
                else:
                    mode = ProcessingMode.BATCH
                
                self.processor.submit_chunk(chunk, mode)
                
            except Exception as e:
                logger.error(f"Adaptive chunk processor error: {e}")
        
        logger.info("Adaptive chunk processor stopped")
    
    def _timeout_handler(self) -> None:
        """Handle timeouts to force processing"""
        logger.info("Timeout handler started")
        
        while not self._stop_event.is_set():
            try:
                time.sleep(0.5)  # Check every 500ms
                
                # Force processing if waiting too long
                chunk = self.audio_buffer.get_chunk(force_immediate=True)
                if chunk:
                    mode = ProcessingMode.IMMEDIATE
                    self.processor.submit_chunk(chunk, mode)
                
            except Exception as e:
                logger.error(f"Timeout handler error: {e}")
        
        logger.info("Timeout handler stopped")
    
    def _handle_result(self, result: TranscriptionResult) -> None:
        """Handle result"""
        with self._results_lock:
            self.results.append(result)
            if len(self.results) > 50:
                self.results = self.results[-50:]
        
        if self.result_callback:
            try:
                self.result_callback(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        rtf = result.processing_time / (result.end_time - result.start_time) if result.processing_time > 0 else 0.0
        logger.info(f"[{result.chunk_id}] '{result.text}' (mode: {result.processing_mode.value}, RTF: {rtf:.3f}x)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        processor_stats = self.processor.get_stats()
        
        with self._results_lock:
            result_count = len(self.results)
        
        return {
            "status": self.status,
            "processor": processor_stats,
            "results_count": result_count
        }
    
    def get_full_text(self) -> str:
        """Get full transcribed text"""
        with self._results_lock:
            return " ".join([r.text for r in self.results if r.text])


# Test function
async def test_adaptive_realtime():
    """Test adaptive real-time processing with different input patterns"""
    print("🎯 Adaptive Real-time STT Test")
    print("Testing real-world variable input scenarios")
    print("=" * 70)
    
    results = []
    mode_counts = {}
    
    def on_result(result: TranscriptionResult):
        results.append(result)
        mode = result.processing_mode.value
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        duration = result.end_time - result.start_time
        rtf = result.processing_time / duration if result.processing_time > 0 else 0.0
        print(f"📝 [{result.chunk_id:03d}] '{result.text}' (mode: {mode}, RTF: {rtf:.3f}x)")
    
    engine = AdaptiveRealtimeSTT(result_callback=on_result)
    
    try:
        await engine.start()
        
        # Simulate real-world input patterns
        test_scenarios = [
            (1.0, "1-second utterance"),
            (2.5, "2.5-second utterance"),
            (0.8, "short utterance"),
            (4.0, "longer utterance"),
            (3.2, "medium utterance"),
            (0.6, "very short"),
            (5.5, "long utterance")
        ]
        
        print("🎵 Simulating real-world audio input patterns...")
        
        for duration, description in test_scenarios:
            # Create audio
            samples = int(duration * 16000)
            t = np.linspace(0, duration, samples)
            signal = (0.4 * np.sin(2 * np.pi * 440 * t) +
                     0.3 * np.sin(2 * np.pi * 880 * t))
            envelope = np.exp(-1.5 * t) * (1 - np.exp(-5 * t))
            audio = (signal * envelope * 0.8).astype(np.float32)
            
            print(f"🔊 Adding {description} ({duration}s)")
            
            # Stream in small chunks (simulate real-time)
            chunk_size = 800  # 50ms chunks
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                is_final = (i + chunk_size >= len(audio))
                engine.add_audio(chunk, is_final)
                await asyncio.sleep(0.05)  # Real-time delay
            
            await asyncio.sleep(0.5)  # Pause between utterances
        
        # Wait for final processing
        await asyncio.sleep(3.0)
        
        # Results analysis
        stats = engine.get_stats()
        print(f"\n📊 Adaptive Processing Results:")
        print(f"Total results: {len(results)}")
        print(f"Mode distribution: {mode_counts}")
        
        for mode, mode_stats in stats['processor'].items():
            if mode_stats['count'] > 0:
                print(f"{mode}: {mode_stats['count']} chunks, avg RTF: {mode_stats['avg_rtf']:.3f}x")
        
        # Calculate overall performance
        if results:
            all_rtfs = []
            for result in results:
                if result.processing_time > 0:
                    duration = result.end_time - result.start_time
                    rtf = result.processing_time / duration
                    all_rtfs.append(rtf)
            
            if all_rtfs:
                avg_rtf = np.mean(all_rtfs)
                print(f"\nOverall average RTF: {avg_rtf:.3f}x")
                print(f"Target achieved: {'✅ YES' if avg_rtf <= 0.05 else '❌ NO'}")
        
        full_text = engine.get_full_text()
        if full_text:
            print(f"\n📝 Full transcription: '{full_text}'")
        
        return True
        
    finally:
        await engine.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_adaptive_realtime()) 