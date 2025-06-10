#!/usr/bin/env python3
"""
Batch STT Service for High-Performance Transcription
Optimized for 8x real-time processing speed with parallel processing
"""

import asyncio
import logging
import time
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import threading
from pathlib import Path

import numpy as np
import torch
from faster_whisper import WhisperModel

from .stt_service import FasterWhisperSTTService, STTResult, STTServiceError
from ..utils.audio_utils import decode_audio_data
from ..utils.korean_processor import get_korean_processor, KoreanProcessingConfig
from ..utils.performance_optimizer import (
    get_performance_optimizer,
    get_error_handler,
    OptimizationConfig,
    PerformanceMetrics
)

# Configure logging
logger = logging.getLogger(__name__)

# Batch processing constants
BATCH_TARGET_RTF = 0.125  # 8x real-time speed (1/8 = 0.125)
MAX_CONCURRENT_WORKERS = 8
DEFAULT_BATCH_SIZE = 16
GPU_MEMORY_THRESHOLD = 0.9  # Use up to 90% of GPU memory for batch processing


@dataclass
class BatchSTTResult:
    """Batch STT processing result with comprehensive metrics"""
    results: List[STTResult]
    total_files: int
    successful_files: int
    failed_files: int
    total_audio_duration: float
    total_processing_time: float
    average_rtf: float
    peak_memory_usage: float
    concurrent_workers_used: int
    batch_id: str
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        return (self.successful_files / self.total_files * 100) if self.total_files > 0 else 0.0
    
    @property
    def throughput_speedup(self) -> float:
        """Calculate throughput speedup compared to real-time"""
        return 1.0 / self.average_rtf if self.average_rtf > 0 else float('inf')
    
    @property
    def meets_performance_targets(self) -> bool:
        """Check if batch meets 8x real-time performance target"""
        return self.average_rtf <= BATCH_TARGET_RTF
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response"""
        return {
            'results': [result.to_dict() for result in self.results],
            'batch_metrics': {
                'total_files': self.total_files,
                'successful_files': self.successful_files,
                'failed_files': self.failed_files,
                'success_rate_percent': self.success_rate,
                'total_audio_duration': self.total_audio_duration,
                'total_processing_time': self.total_processing_time,
                'average_rtf': self.average_rtf,
                'throughput_speedup': self.throughput_speedup,
                'meets_8x_target': self.meets_performance_targets,
                'peak_memory_usage_gb': self.peak_memory_usage,
                'concurrent_workers': self.concurrent_workers_used
            },
            'batch_id': self.batch_id
        }


class BatchSTTService:
    """
    High-performance batch STT service
    Optimized for 8x real-time processing with parallel execution
    """
    
    def __init__(self, base_stt_service: FasterWhisperSTTService):
        """
        Initialize batch service with base STT service
        
        Args:
            base_stt_service: Initialized FasterWhisperSTTService instance
        """
        self.base_service = base_stt_service
        self.max_workers = min(MAX_CONCURRENT_WORKERS, torch.cuda.device_count() * 2)
        
        # Batch-specific configuration
        self.batch_config = {
            'max_concurrent_batches': 2,
            'optimal_batch_size': DEFAULT_BATCH_SIZE,
            'memory_buffer_ratio': 0.1,  # Reserve 10% memory as buffer
            'dynamic_worker_scaling': True,
            'priority_queue_enabled': True
        }
        
        # Performance monitoring
        self.batch_stats = {
            'total_batches': 0,
            'successful_batches': 0,
            'total_files_processed': 0,
            'average_batch_rtf': 0.0,
            'peak_concurrent_workers': 0,
            'memory_efficiency_ratio': 0.0
        }
        
        self.stats_lock = threading.Lock()
        
        # Initialize Korean processor for batch processing
        korean_config = KoreanProcessingConfig(
            use_korean_vad=True,
            normalize_text=True,
            enhance_word_boundaries=True,
            filter_confidence_threshold=0.25,  # Slightly lower for batch processing
            enable_post_processing=True,
            batch_processing_mode=True
        )
        self.korean_processor = get_korean_processor(korean_config)
        
        # Performance optimizer for batch processing
        batch_optimization_config = OptimizationConfig(
            max_worker_threads=self.max_workers,
            enable_gc_optimization=True,
            gpu_memory_fraction=GPU_MEMORY_THRESHOLD,
            enable_memory_monitoring=True,
            alert_threshold_rtf=BATCH_TARGET_RTF * 1.2,  # 20% tolerance
            alert_threshold_latency_ms=5000,  # Higher tolerance for batch
            max_retry_attempts=2,  # Fewer retries for batch processing
            enable_graceful_degradation=True
        )
        
        self.performance_optimizer = get_performance_optimizer(batch_optimization_config)
        self.error_handler = get_error_handler(batch_optimization_config)
        
        logger.info(f"Batch STT Service initialized with {self.max_workers} max workers")
        logger.info(f"Target RTF: {BATCH_TARGET_RTF:.3f}x (8x real-time speed)")
    
    async def process_batch(self,
                           audio_files: List[str],  # List of base64 encoded audio
                           audio_format: str = 'wav',
                           language: str = 'ko',
                           beam_size: int = 1,  # Lower beam size for batch speed
                           enable_timestamps: bool = True,
                           enable_confidence: bool = True,
                           batch_id: Optional[str] = None) -> BatchSTTResult:
        """
        Process multiple audio files in parallel for high-speed batch transcription
        
        Args:
            audio_files: List of base64 encoded audio data
            audio_format: Audio format for all files
            language: Target language
            beam_size: Beam size for decoding (lower = faster)
            enable_timestamps: Whether to include word timestamps
            enable_confidence: Whether to include confidence scores
            batch_id: Optional batch identifier
            
        Returns:
            BatchSTTResult with comprehensive metrics
        """
        batch_id = batch_id or f"batch_{int(time.time() * 1000)}"
        start_time = time.time()
        
        logger.info(f"Starting batch processing {batch_id} with {len(audio_files)} files")
        logger.info(f"Target: 8x real-time speed (RTF ≤ {BATCH_TARGET_RTF:.3f})")
        
        # Verify service is ready
        if not self.base_service.is_healthy():
            raise STTServiceError("Base STT service is not healthy")
        
        # Pre-processing optimization
        self.performance_optimizer.optimize_pre_inference()
        
        # Calculate optimal concurrency based on file count and memory
        optimal_workers = min(
            self.max_workers,
            len(audio_files),
            max(1, int(self.max_workers * 0.8))  # Conservative estimate
        )
        
        results: List[STTResult] = []
        successful_files = 0
        failed_files = 0
        total_audio_duration = 0.0
        
        try:
            # Create semaphore to limit concurrent processing
            semaphore = asyncio.Semaphore(optimal_workers)
            
            # Process files in parallel with controlled concurrency
            tasks = []
            for i, audio_data in enumerate(audio_files):
                task = self._process_single_file_with_semaphore(
                    semaphore=semaphore,
                    audio_data=audio_data,
                    file_index=i,
                    audio_format=audio_format,
                    language=language,
                    beam_size=beam_size,
                    enable_timestamps=enable_timestamps,
                    enable_confidence=enable_confidence,
                    batch_id=batch_id
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(completed_results):
                if isinstance(result, Exception):
                    logger.error(f"File {i} in batch {batch_id} failed: {result}")
                    failed_files += 1
                    
                    # Create error result placeholder
                    error_result = STTResult(
                        text="",
                        language="unknown",
                        segments=[],
                        audio_duration=0.0,
                        processing_time=0.0,
                        rtf=0.0,
                        model_info={}
                    )
                    results.append(error_result)
                else:
                    results.append(result)
                    successful_files += 1
                    total_audio_duration += result.audio_duration
            
            # Post-processing optimization
            self.performance_optimizer.optimize_post_inference()
            
            # Calculate batch metrics
            total_processing_time = time.time() - start_time
            average_rtf = (
                total_processing_time / total_audio_duration
                if total_audio_duration > 0 else float('inf')
            )
            
            # Get memory usage
            memory_info = self.performance_optimizer.get_current_performance()
            peak_memory_gb = memory_info.gpu_memory_used / (1024**3) if memory_info else 0.0
            
            # Create batch result
            batch_result = BatchSTTResult(
                results=results,
                total_files=len(audio_files),
                successful_files=successful_files,
                failed_files=failed_files,
                total_audio_duration=total_audio_duration,
                total_processing_time=total_processing_time,
                average_rtf=average_rtf,
                peak_memory_usage=peak_memory_gb,
                concurrent_workers_used=optimal_workers,
                batch_id=batch_id
            )
            
            # Update statistics
            self._update_batch_stats(batch_result)
            
            # Log results
            performance_status = "✅ MEETS 8X TARGET" if batch_result.meets_performance_targets else "⚠️ BELOW 8X TARGET"
            logger.info(
                f"Batch {batch_id} completed: {successful_files}/{len(audio_files)} successful, "
                f"RTF={average_rtf:.3f}x ({batch_result.throughput_speedup:.1f}x speedup), "
                f"Workers={optimal_workers} {performance_status}"
            )
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Batch {batch_id} processing failed: {e}")
            self.error_handler.handle_error(e, {"batch_id": batch_id, "file_count": len(audio_files)})
            raise STTServiceError(f"Batch processing failed: {e}")
    
    async def _process_single_file_with_semaphore(self,
                                                 semaphore: asyncio.Semaphore,
                                                 audio_data: str,
                                                 file_index: int,
                                                 audio_format: str,
                                                 language: str,
                                                 beam_size: int,
                                                 enable_timestamps: bool,
                                                 enable_confidence: bool,
                                                 batch_id: str) -> STTResult:
        """Process a single file with semaphore-based concurrency control"""
        async with semaphore:
            try:
                # Korean language optimization
                if language == 'ko':
                    # Use Korean-optimized parameters
                    korean_params = self.korean_processor.get_korean_optimized_params()
                    vad_parameters = korean_params.get('vad_parameters', {})
                else:
                    vad_parameters = None
                
                # Process file with optimized parameters
                result = await self.base_service.transcribe_audio(
                    audio_data=audio_data,
                    audio_format=audio_format,
                    language=language,
                    beam_size=beam_size,
                    word_timestamps=enable_timestamps,
                    vad_filter=True,
                    vad_parameters=vad_parameters
                )
                
                # Korean post-processing if needed
                if language == 'ko' and result.text:
                    result.text = self.korean_processor.normalize_korean_text(result.text)
                
                logger.debug(f"File {file_index} in batch {batch_id}: RTF={result.rtf:.3f}x")
                return result
                
            except Exception as e:
                logger.error(f"Failed to process file {file_index} in batch {batch_id}: {e}")
                raise
    
    def _update_batch_stats(self, batch_result: BatchSTTResult):
        """Update batch processing statistics"""
        with self.stats_lock:
            self.batch_stats['total_batches'] += 1
            if batch_result.failed_files == 0:
                self.batch_stats['successful_batches'] += 1
            
            self.batch_stats['total_files_processed'] += batch_result.total_files
            
            # Update rolling average RTF
            total_batches = self.batch_stats['total_batches']
            current_avg = self.batch_stats['average_batch_rtf']
            self.batch_stats['average_batch_rtf'] = (
                (current_avg * (total_batches - 1) + batch_result.average_rtf) / total_batches
            )
            
            # Update peak workers
            self.batch_stats['peak_concurrent_workers'] = max(
                self.batch_stats['peak_concurrent_workers'],
                batch_result.concurrent_workers_used
            )
    
    def get_batch_stats(self) -> Dict:
        """Get batch processing statistics"""
        with self.stats_lock:
            return dict(self.batch_stats)
    
    def reset_batch_stats(self):
        """Reset batch processing statistics"""
        with self.stats_lock:
            for key in self.batch_stats:
                if isinstance(self.batch_stats[key], (int, float)):
                    self.batch_stats[key] = 0 if isinstance(self.batch_stats[key], int) else 0.0
    
    def is_healthy(self) -> bool:
        """Check if batch service is healthy"""
        return self.base_service.is_healthy()
    
    def get_optimal_batch_size(self, estimated_file_sizes: List[int]) -> int:
        """Calculate optimal batch size based on file sizes and available memory"""
        # Simple heuristic: adjust batch size based on average file size
        avg_file_size_mb = sum(estimated_file_sizes) / len(estimated_file_sizes) / (1024**2)
        
        if avg_file_size_mb < 1:  # Small files
            return min(32, DEFAULT_BATCH_SIZE * 2)
        elif avg_file_size_mb < 5:  # Medium files
            return DEFAULT_BATCH_SIZE
        else:  # Large files
            return max(4, DEFAULT_BATCH_SIZE // 2)


# Global batch service instance
_batch_service: Optional[BatchSTTService] = None


def get_batch_service() -> BatchSTTService:
    """Get the global batch STT service instance"""
    global _batch_service
    if _batch_service is None:
        raise STTServiceError("Batch service not initialized")
    return _batch_service


async def initialize_batch_service(base_stt_service: FasterWhisperSTTService) -> bool:
    """Initialize the global batch STT service"""
    global _batch_service
    try:
        _batch_service = BatchSTTService(base_stt_service)
        logger.info("Batch STT service initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize batch service: {e}")
        return False


async def shutdown_batch_service():
    """Shutdown the global batch STT service"""
    global _batch_service
    if _batch_service:
        logger.info("Shutting down batch STT service...")
        _batch_service = None
        logger.info("Batch STT service shutdown complete")


if __name__ == "__main__":
    # Test batch service
    import asyncio
    from .stt_service import get_stt_service, initialize_stt_service
    
    async def test_batch_service():
        """Test batch processing service"""
        print("Testing Batch STT Service...")
        
        # Initialize base service
        await initialize_stt_service()
        base_service = get_stt_service()
        
        # Initialize batch service
        await initialize_batch_service(base_service)
        batch_service = get_batch_service()
        
        print(f"✅ Batch service initialized with {batch_service.max_workers} workers")
        print(f"🎯 Target: 8x real-time speed (RTF ≤ {BATCH_TARGET_RTF:.3f})")
        
        # Create test batch (dummy data)
        test_audio_files = ["dummy_audio_data"] * 5  # 5 test files
        
        try:
            result = await batch_service.process_batch(
                audio_files=test_audio_files,
                batch_id="test_batch"
            )
            
            print(f"📊 Batch Results:")
            print(f"   Files: {result.successful_files}/{result.total_files}")
            print(f"   RTF: {result.average_rtf:.3f}x")
            print(f"   Speedup: {result.throughput_speedup:.1f}x")
            print(f"   Target met: {result.meets_performance_targets}")
            
        except Exception as e:
            print(f"❌ Batch test failed: {e}")
        
        await shutdown_batch_service()
    
    asyncio.run(test_batch_service()) 