#!/usr/bin/env python3
"""
Batch Processing Pipeline for STT
High-performance batch processing of audio files using Faster Whisper
"""

import asyncio
import concurrent.futures
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Callable, Union, Any, Generator
import logging
import json
import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf

import sys
sys.path.append('/home/jonghooy/haiv_stt_decoder_v2/src')

from core.config import STTConfig, DeviceType, ComputeType
from models.whisper_model import WhisperModelManager

logger = logging.getLogger(__name__)


@dataclass
class AudioFileInfo:
    """Audio file information"""
    filepath: Path
    filename: str
    size_bytes: int
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    format: Optional[str] = None
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class BatchTranscriptionResult:
    """Batch transcription result"""
    file_info: AudioFileInfo
    text: str
    segments: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    timestamp: float
    language: str
    model_info: Dict[str, Any]
    error: Optional[str] = None
    success: bool = True


@dataclass
class BatchProcessingStats:
    """Batch processing statistics"""
    total_files: int = 0
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_audio_duration: float = 0.0
    total_processing_time: float = 0.0
    average_rtf: float = 0.0
    throughput_files_per_minute: float = 0.0
    throughput_hours_per_hour: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)


class AudioFileAnalyzer:
    """Analyze and validate audio files"""
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.webm', '.mp4'}
    
    @classmethod
    def analyze_file(cls, filepath: Path) -> AudioFileInfo:
        """Analyze a single audio file"""
        try:
            if not filepath.exists():
                return AudioFileInfo(
                    filepath=filepath,
                    filename=filepath.name,
                    size_bytes=0,
                    is_valid=False,
                    error_message="File does not exist"
                )
            
            # Check file extension
            if filepath.suffix.lower() not in cls.SUPPORTED_FORMATS:
                return AudioFileInfo(
                    filepath=filepath,
                    filename=filepath.name,
                    size_bytes=filepath.stat().st_size,
                    is_valid=False,
                    error_message=f"Unsupported format: {filepath.suffix}"
                )
            
            # Get basic file info
            stat = filepath.stat()
            
            # Analyze audio properties
            try:
                # Use librosa for audio analysis
                y, sr = librosa.load(str(filepath), sr=None)
                duration = len(y) / sr
                
                return AudioFileInfo(
                    filepath=filepath,
                    filename=filepath.name,
                    size_bytes=stat.st_size,
                    duration=duration,
                    sample_rate=sr,
                    channels=1 if y.ndim == 1 else y.shape[0],
                    format=filepath.suffix.lower(),
                    is_valid=True
                )
                
            except Exception as audio_error:
                return AudioFileInfo(
                    filepath=filepath,
                    filename=filepath.name,
                    size_bytes=stat.st_size,
                    is_valid=False,
                    error_message=f"Audio analysis failed: {audio_error}"
                )
                
        except Exception as e:
            return AudioFileInfo(
                filepath=filepath,
                filename=filepath.name,
                size_bytes=0,
                is_valid=False,
                error_message=f"File analysis failed: {e}"
            )
    
    @classmethod
    def analyze_directory(cls, directory: Path, recursive: bool = True) -> List[AudioFileInfo]:
        """Analyze all audio files in a directory"""
        files = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for filepath in directory.glob(pattern):
            if filepath.is_file() and filepath.suffix.lower() in cls.SUPPORTED_FORMATS:
                file_info = cls.analyze_file(filepath)
                files.append(file_info)
        
        return files


class AudioChunker:
    """Split large audio files into manageable chunks"""
    
    def __init__(self, max_chunk_duration: float = 30.0, overlap_duration: float = 1.0):
        """
        Initialize audio chunker
        
        Args:
            max_chunk_duration: Maximum duration per chunk in seconds
            overlap_duration: Overlap between chunks in seconds
        """
        self.max_chunk_duration = max_chunk_duration
        self.overlap_duration = overlap_duration
    
    def chunk_audio(self, audio_data: np.ndarray, sample_rate: int) -> Generator[np.ndarray, None, None]:
        """
        Split audio into overlapping chunks
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Yields:
            Audio chunks as numpy arrays
        """
        total_duration = len(audio_data) / sample_rate
        
        if total_duration <= self.max_chunk_duration:
            yield audio_data
            return
        
        chunk_samples = int(self.max_chunk_duration * sample_rate)
        overlap_samples = int(self.overlap_duration * sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        start = 0
        while start < len(audio_data):
            end = min(start + chunk_samples, len(audio_data))
            yield audio_data[start:end]
            
            # If this was the last chunk, break
            if end == len(audio_data):
                break
                
            start += step_samples
    
    def merge_results(self, chunk_results: List[Dict[str, Any]], overlap_duration: float) -> Dict[str, Any]:
        """
        Merge transcription results from overlapping chunks
        
        Args:
            chunk_results: List of transcription results from chunks
            overlap_duration: Overlap duration used for chunking
            
        Returns:
            Merged transcription result
        """
        if not chunk_results:
            return {"text": "", "segments": []}
        
        if len(chunk_results) == 1:
            return chunk_results[0]
        
        # Simple merging strategy: concatenate text and adjust timestamps
        merged_text = ""
        merged_segments = []
        cumulative_time = 0.0
        
        for i, result in enumerate(chunk_results):
            if isinstance(result, dict) and 'text' in result:
                chunk_text = result['text'].strip()
                
                if i == 0:
                    # First chunk: use as-is
                    merged_text = chunk_text
                    if 'segments' in result:
                        merged_segments.extend(result['segments'])
                else:
                    # Subsequent chunks: add with time offset
                    if chunk_text:
                        merged_text = merged_text.rstrip() + " " + chunk_text
                    
                    if 'segments' in result:
                        for segment in result['segments']:
                            adjusted_segment = segment.copy()
                            if 'start' in adjusted_segment:
                                adjusted_segment['start'] += cumulative_time
                            if 'end' in adjusted_segment:
                                adjusted_segment['end'] += cumulative_time
                            merged_segments.append(adjusted_segment)
                
                # Update cumulative time (accounting for overlap)
                if i < len(chunk_results) - 1:
                    cumulative_time += self.max_chunk_duration - overlap_duration
        
        return {
            "text": merged_text.strip(),
            "segments": merged_segments
        }


class BatchSTTProcessor:
    """High-performance batch STT processor"""
    
    def __init__(self, 
                 config: Optional[STTConfig] = None,
                 max_workers: Optional[int] = None,
                 max_chunk_duration: float = 30.0,
                 progress_callback: Optional[Callable[[BatchProcessingStats], None]] = None):
        """
        Initialize batch STT processor
        
        Args:
            config: STT configuration
            max_workers: Maximum number of worker threads
            max_chunk_duration: Maximum duration per audio chunk
            progress_callback: Callback for progress updates
        """
        if config is None:
            config = STTConfig()
            config.model.device = DeviceType.CUDA
            config.model.compute_type = ComputeType.FLOAT16
            config.model.beam_size = 1
            config.model.local_files_only = True
        
        self.config = config
        self.max_workers = max_workers or min(4, os.cpu_count() or 1)
        self.progress_callback = progress_callback
        
        # Initialize components
        self.model_manager = WhisperModelManager(config)
        self.chunker = AudioChunker(max_chunk_duration=max_chunk_duration)
        self.analyzer = AudioFileAnalyzer()
        
        # Thread safety
        self._model_lock = threading.RLock()
        self._stats_lock = threading.Lock()
        
        # Statistics
        self.stats = BatchProcessingStats()
        
        logger.info(f"BatchSTTProcessor initialized with {self.max_workers} workers")
    
    def start(self) -> None:
        """Initialize the batch processor"""
        logger.info("Starting batch STT processor...")
        
        with self._model_lock:
            model_info = self.model_manager.load_model()
            logger.info(f"Model loaded: {model_info.config.get('model_size_or_path', 'unknown')}")
        
        logger.info("Batch STT processor ready")
    
    def stop(self) -> None:
        """Cleanup the batch processor"""
        logger.info("Stopping batch STT processor...")
        
        with self._model_lock:
            self.model_manager.unload_model()
        
        logger.info("Batch STT processor stopped")
    
    def process_file(self, file_info: AudioFileInfo) -> BatchTranscriptionResult:
        """Process a single audio file"""
        start_time = time.perf_counter()
        
        try:
            if not file_info.is_valid:
                return BatchTranscriptionResult(
                    file_info=file_info,
                    text="",
                    segments=[],
                    confidence=0.0,
                    processing_time=0.0,
                    timestamp=time.time(),
                    language="",
                    model_info={},
                    error=file_info.error_message,
                    success=False
                )
            
            # Load audio
            audio_data, sample_rate = librosa.load(str(file_info.filepath), sr=16000)
            
            # Process in chunks if necessary
            chunks = list(self.chunker.chunk_audio(audio_data, sample_rate))
            chunk_results = []
            
            for chunk in chunks:
                # Thread-safe model access
                with self._model_lock:
                    segments, info = self.model_manager.transcribe(
                        chunk,
                        beam_size=self.config.model.beam_size,
                        language="en",
                        vad_filter=True,
                        temperature=0.0,
                        compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0,
                        no_speech_threshold=0.6,
                        condition_on_previous_text=False,
                        without_timestamps=False,
                        word_timestamps=False
                    )
                
                # Extract result
                segments_list = list(segments)
                text = " ".join([s.text for s in segments_list if hasattr(s, 'text')])
                
                chunk_results.append({
                    "text": text.strip(),
                    "segments": [{"text": s.text, "start": s.start, "end": s.end} 
                               for s in segments_list if hasattr(s, 'text')]
                })
            
            # Merge chunks
            if len(chunks) > 1:
                merged_result = self.chunker.merge_results(
                    chunk_results, 
                    self.chunker.overlap_duration
                )
            else:
                merged_result = chunk_results[0] if chunk_results else {"text": "", "segments": []}
            
            processing_time = time.perf_counter() - start_time
            
            return BatchTranscriptionResult(
                file_info=file_info,
                text=merged_result["text"],
                segments=merged_result["segments"],
                confidence=0.8,  # Default confidence
                processing_time=processing_time,
                timestamp=time.time(),
                language="en",
                model_info={"model": "faster-whisper-large-v3"},
                success=True
            )
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            error_msg = f"Processing failed: {e}"
            logger.error(f"Error processing {file_info.filename}: {error_msg}")
            
            return BatchTranscriptionResult(
                file_info=file_info,
                text="",
                segments=[],
                confidence=0.0,
                processing_time=processing_time,
                timestamp=time.time(),
                language="",
                model_info={},
                error=error_msg,
                success=False
            )
    
    def process_batch(self, 
                     file_infos: List[AudioFileInfo],
                     output_dir: Optional[Path] = None,
                     save_individual: bool = True,
                     save_summary: bool = True) -> List[BatchTranscriptionResult]:
        """
        Process a batch of audio files
        
        Args:
            file_infos: List of audio file information
            output_dir: Output directory for results
            save_individual: Save individual transcription files
            save_summary: Save batch summary
            
        Returns:
            List of transcription results
        """
        logger.info(f"Starting batch processing of {len(file_infos)} files")
        
        # Initialize statistics
        with self._stats_lock:
            self.stats = BatchProcessingStats()
            self.stats.total_files = len(file_infos)
            self.stats.start_time = time.time()
            self.stats.total_audio_duration = sum(
                f.duration for f in file_infos if f.duration is not None
            )
        
        results = []
        
        # Process files with progress tracking
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_file, file_info): file_info 
                for file_info in file_infos
            }
            
            # Process results with progress bar
            with tqdm(total=len(file_infos), desc="Processing files") as pbar:
                for future in as_completed(future_to_file):
                    file_info = future_to_file[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update statistics
                        with self._stats_lock:
                            self.stats.processed_files += 1
                            if result.success:
                                self.stats.successful_files += 1
                                self.stats.total_processing_time += result.processing_time
                            else:
                                self.stats.failed_files += 1
                                if result.error:
                                    self.stats.errors.append(f"{file_info.filename}: {result.error}")
                            
                            # Calculate RTF and throughput
                            if self.stats.successful_files > 0:
                                processed_duration = sum(
                                    r.file_info.duration or 0 
                                    for r in results if r.success and r.file_info.duration
                                )
                                if processed_duration > 0 and self.stats.total_processing_time > 0:
                                    self.stats.average_rtf = self.stats.total_processing_time / processed_duration
                                
                                elapsed_time = time.time() - self.stats.start_time
                                if elapsed_time > 0:
                                    self.stats.throughput_files_per_minute = (
                                        self.stats.processed_files / elapsed_time * 60
                                    )
                                    if processed_duration > 0:
                                        self.stats.throughput_hours_per_hour = (
                                            processed_duration / 3600 / elapsed_time * 3600
                                        )
                        
                        # Progress callback
                        if self.progress_callback:
                            with self._stats_lock:
                                self.progress_callback(self.stats)
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'Success': f"{self.stats.successful_files}/{self.stats.processed_files}",
                            'RTF': f"{self.stats.average_rtf:.3f}x" if self.stats.average_rtf > 0 else "N/A"
                        })
                        
                    except Exception as e:
                        logger.error(f"Failed to process {file_info.filename}: {e}")
                        
                        # Create error result
                        error_result = BatchTranscriptionResult(
                            file_info=file_info,
                            text="",
                            segments=[],
                            confidence=0.0,
                            processing_time=0.0,
                            timestamp=time.time(),
                            language="",
                            model_info={},
                            error=str(e),
                            success=False
                        )
                        results.append(error_result)
                        
                        with self._stats_lock:
                            self.stats.processed_files += 1
                            self.stats.failed_files += 1
                            self.stats.errors.append(f"{file_info.filename}: {e}")
                        
                        pbar.update(1)
        
        # Finalize statistics
        with self._stats_lock:
            self.stats.end_time = time.time()
        
        # Save results if requested
        if output_dir:
            self._save_results(results, output_dir, save_individual, save_summary)
        
        logger.info(f"Batch processing completed: {self.stats.successful_files}/{self.stats.total_files} files successful")
        
        return results
    
    def _save_results(self, 
                     results: List[BatchTranscriptionResult],
                     output_dir: Path,
                     save_individual: bool,
                     save_summary: bool) -> None:
        """Save batch processing results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual transcriptions
        if save_individual:
            for result in results:
                if result.success:
                    filename = result.file_info.filepath.stem + "_transcription.txt"
                    output_file = output_dir / filename
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result.text)
        
        # Save batch summary
        if save_summary:
            summary_file = output_dir / "batch_summary.json"
            
            summary_data = {
                "statistics": {
                    "total_files": self.stats.total_files,
                    "successful_files": self.stats.successful_files,
                    "failed_files": self.stats.failed_files,
                    "total_audio_duration": self.stats.total_audio_duration,
                    "total_processing_time": self.stats.total_processing_time,
                    "average_rtf": self.stats.average_rtf,
                    "throughput_files_per_minute": self.stats.throughput_files_per_minute,
                    "throughput_hours_per_hour": self.stats.throughput_hours_per_hour,
                    "start_time": self.stats.start_time,
                    "end_time": self.stats.end_time,
                    "errors": self.stats.errors
                },
                "results": [
                    {
                        "filename": r.file_info.filename,
                        "text": r.text,
                        "processing_time": r.processing_time,
                        "success": r.success,
                        "error": r.error
                    }
                    for r in results
                ]
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    def get_stats(self) -> BatchProcessingStats:
        """Get current processing statistics"""
        with self._stats_lock:
            return self.stats


# Convenience functions
def process_directory(directory: Union[str, Path],
                     output_dir: Optional[Union[str, Path]] = None,
                     recursive: bool = True,
                     max_workers: Optional[int] = None,
                     config: Optional[STTConfig] = None) -> List[BatchTranscriptionResult]:
    """
    Process all audio files in a directory
    
    Args:
        directory: Input directory containing audio files
        output_dir: Output directory for results
        recursive: Process subdirectories recursively
        max_workers: Maximum number of worker threads
        config: STT configuration
        
    Returns:
        List of transcription results
    """
    directory = Path(directory)
    output_dir = Path(output_dir) if output_dir else directory / "transcriptions"
    
    # Analyze files
    analyzer = AudioFileAnalyzer()
    file_infos = analyzer.analyze_directory(directory, recursive=recursive)
    
    if not file_infos:
        logger.warning(f"No audio files found in {directory}")
        return []
    
    logger.info(f"Found {len(file_infos)} audio files")
    
    # Process files
    processor = BatchSTTProcessor(config=config, max_workers=max_workers)
    
    try:
        processor.start()
        results = processor.process_batch(
            file_infos=file_infos,
            output_dir=output_dir,
            save_individual=True,
            save_summary=True
        )
        return results
    finally:
        processor.stop()


def process_file_list(file_paths: List[Union[str, Path]],
                     output_dir: Optional[Union[str, Path]] = None,
                     max_workers: Optional[int] = None,
                     config: Optional[STTConfig] = None) -> List[BatchTranscriptionResult]:
    """
    Process a list of audio files
    
    Args:
        file_paths: List of audio file paths
        output_dir: Output directory for results
        max_workers: Maximum number of worker threads
        config: STT configuration
        
    Returns:
        List of transcription results
    """
    output_dir = Path(output_dir) if output_dir else Path("transcriptions")
    
    # Analyze files
    analyzer = AudioFileAnalyzer()
    file_infos = [analyzer.analyze_file(Path(fp)) for fp in file_paths]
    
    # Process files
    processor = BatchSTTProcessor(config=config, max_workers=max_workers)
    
    try:
        processor.start()
        results = processor.process_batch(
            file_infos=file_infos,
            output_dir=output_dir,
            save_individual=True,
            save_summary=True
        )
        return results
    finally:
        processor.stop()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample directory (for testing)
    print("🔄 Batch Processing Pipeline Demo")
    print("To use: python -c \"from batch_processor import process_directory; process_directory('path/to/audio/files')\"") 