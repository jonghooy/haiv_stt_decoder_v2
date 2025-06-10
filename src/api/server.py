#!/usr/bin/env python3
"""
FastAPI Server for Real-time STT
High-performance real-time speech-to-text API server
"""

import asyncio
import base64
import io
import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import psutil

import sys
sys.path.append('/home/jonghooy/haiv_stt_decoder_v2/src')

from api.models import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionSegment,
    WordSegment,
    ProcessingMetrics,
    ErrorResponse,
    ErrorDetail,
    HealthCheck,
    HealthStatus,
    BatchTranscriptionRequest,
    BatchTranscriptionResponse,
    AudioFormat,
    LanguageCode
)
from core.config import STTConfig, DeviceType, ComputeType
from models.whisper_model import WhisperModelManager
from pipeline.realtime_optimized import UltraOptimizedRealtimeSTT
from stt_service import get_stt_service, initialize_stt_service, shutdown_stt_service, STTServiceError
from batch_service import get_batch_service, initialize_batch_service, shutdown_batch_service, BatchSTTResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
model_manager: Optional[WhisperModelManager] = None
realtime_engine: Optional[UltraOptimizedRealtimeSTT] = None
server_stats = {
    "start_time": time.time(),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_processing_time": 0.0,
    "total_audio_duration": 0.0
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("Starting STT API server...")
    
    try:
        # Initialize STT service
        logger.info("Initializing STT service...")
        success = await initialize_stt_service()
        
        if not success:
            raise RuntimeError("Failed to initialize STT service")
        
        logger.info("✅ STT service initialized successfully")
        
        # Initialize batch service
        logger.info("Initializing batch STT service...")
        batch_success = await initialize_batch_service(get_stt_service())
        
        if not batch_success:
            logger.warning("⚠️ Batch service initialization failed - batch endpoints will be disabled")
        else:
            logger.info("✅ Batch STT service initialized successfully")
        
        logger.info("STT API server ready to accept requests")
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down STT API server...")
    
    try:
        await shutdown_batch_service()
        await shutdown_stt_service()
        logger.info("✅ Server shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="Real-time Speech-to-Text API",
    description="High-performance real-time speech transcription using Faster Whisper",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)


# Request tracking middleware
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Update stats
    processing_time = time.time() - start_time
    server_stats["total_requests"] += 1
    server_stats["total_processing_time"] += processing_time
    
    if response.status_code < 400:
        server_stats["successful_requests"] += 1
    else:
        server_stats["failed_requests"] += 1
    
    # Add performance headers
    response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    
    return response


# Helper functions
def decode_audio_data(audio_data: str, audio_format: AudioFormat, sample_rate: int) -> np.ndarray:
    """Decode base64 audio data to numpy array"""
    try:
        # Decode base64
        audio_bytes = base64.b64decode(audio_data)
        
        if audio_format == AudioFormat.PCM_16KHZ:
            # Raw PCM data
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
        else:
            # Use soundfile for other formats
            audio_io = io.BytesIO(audio_bytes)
            audio_array, sr = sf.read(audio_io)
            
            # Resample if necessary
            if sr != sample_rate:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=sample_rate)
        
        # Ensure mono
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        return audio_array
        
    except Exception as e:
        raise ValueError(f"Failed to decode audio data: {e}")


def create_error_response(error_code: str, message: str, request_id: Optional[str] = None, details: Optional[dict] = None) -> ErrorResponse:
    """Create standardized error response"""
    return ErrorResponse(
        error=ErrorDetail(
            code=error_code,
            message=message,
            details=details
        ),
        request_id=request_id,
        timestamp=time.time()
    )


# API Routes

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Real-time Speech-to-Text API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    try:
        # Check STT service status
        stt_service = get_stt_service()
        model_status = HealthStatus.HEALTHY
        if not stt_service.is_healthy():
            model_status = HealthStatus.UNHEALTHY
        
        # Check GPU status
        gpu_status = HealthStatus.HEALTHY
        if torch.cuda.is_available():
            try:
                torch.cuda.get_device_properties(0)
            except:
                gpu_status = HealthStatus.DEGRADED
        else:
            gpu_status = HealthStatus.DEGRADED
        
        # Check memory status
        memory = psutil.virtual_memory()
        memory_status = HealthStatus.HEALTHY
        if memory.percent > 90:
            memory_status = HealthStatus.DEGRADED
        elif memory.percent > 95:
            memory_status = HealthStatus.UNHEALTHY
        
        # Overall status
        overall_status = HealthStatus.HEALTHY
        if any(s == HealthStatus.UNHEALTHY for s in [model_status, memory_status]):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in [model_status, gpu_status, memory_status]):
            overall_status = HealthStatus.DEGRADED
        
        # Calculate metrics
        average_inference_time = None
        if server_stats["successful_requests"] > 0:
            average_inference_time = server_stats["total_processing_time"] / server_stats["successful_requests"]
        
        # Memory usage
        memory_usage = {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used
        }
        
        if torch.cuda.is_available():
            try:
                memory_usage["gpu_total"] = torch.cuda.get_device_properties(0).total_memory
                memory_usage["gpu_allocated"] = torch.cuda.memory_allocated(0)
                memory_usage["gpu_cached"] = torch.cuda.memory_reserved(0)
            except:
                pass
        
        return HealthCheck(
            status=overall_status,
            model_status=model_status,
            gpu_status=gpu_status,
            memory_status=memory_status,
            average_inference_time=average_inference_time,
            total_requests=server_stats["total_requests"],
            memory_usage=memory_usage,
            model_version="faster-whisper-large-v3"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status=HealthStatus.UNHEALTHY,
            model_status=HealthStatus.UNHEALTHY,
            gpu_status=HealthStatus.UNHEALTHY,
            memory_status=HealthStatus.UNHEALTHY
        )


@app.post("/infer/utterance", response_model=TranscriptionResponse)
async def transcribe_utterance(request: TranscriptionRequest):
    """
    Transcribe a single audio utterance
    
    This is the main endpoint for real-time speech transcription.
    Accepts audio data and returns transcription with timestamps.
    """
    start_time = time.time()
    request_id = request.request_id or str(uuid.uuid4())
    
    try:
        logger.info(f"Processing transcription request {request_id}")
        
        # Get STT service
        stt_service = get_stt_service()
        
        # Validate service is ready
        if not stt_service.is_healthy():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=create_error_response(
                    "SERVICE_NOT_READY",
                    "STT service is not ready",
                    request_id
                ).dict()
            )
        
        # Map language codes
        language_map = {
            LanguageCode.KOREAN: "ko",
            LanguageCode.ENGLISH: "en",
            LanguageCode.AUTO: None
        }
        language = language_map.get(request.language, "ko")
        
        # Map audio format
        format_map = {
            AudioFormat.PCM_16KHZ: "pcm_16khz",
            AudioFormat.WAV: "wav",
            AudioFormat.MP3: "mp3",
            AudioFormat.FLAC: "flac",
            AudioFormat.M4A: "m4a"
        }
        audio_format = format_map.get(request.audio_format, "pcm_16khz")
        
        logger.info(f"Request {request_id}: format={audio_format}, language={language}")
        
        # Perform transcription using STT service
        try:
            result = await stt_service.transcribe_audio(
                audio_data=request.audio_data,
                audio_format=audio_format,
                language=language,
                word_timestamps=request.enable_timestamps,
                beam_size=request.beam_size
            )
        except STTServiceError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=create_error_response(
                    "STT_ERROR",
                    str(e),
                    request_id
                ).dict()
            )
        
        # Update stats with actual audio duration
        server_stats["total_audio_duration"] += result.audio_duration
        
        # Create response segments
        response_segments = []
        for i, segment in enumerate(result.segments):
            # Create word segments if requested
            words = None
            if request.enable_timestamps and 'words' in segment and segment['words']:
                words = [
                    WordSegment(
                        word=word.get('word', ''),
                        start=word.get('start', 0.0),
                        end=word.get('end', 0.0),
                        confidence=word.get('probability') if request.enable_confidence else None
                    )
                    for word in segment['words']
                ]
            
            response_segments.append(
                TranscriptionSegment(
                    id=i,
                    text=segment['text'].strip(),
                    start=segment.get('start', 0.0),
                    end=segment.get('end', result.audio_duration),
                    confidence=segment.get('avg_logprob') if request.enable_confidence else None,
                    words=words
                )
            )
        
        # Calculate metrics
        total_time = time.time() - start_time
        
        metrics = ProcessingMetrics(
            total_duration=total_time,
            audio_duration=result.audio_duration,
            rtf=result.rtf,
            inference_time=result.processing_time,
            queue_wait_time=0.0  # Single-threaded for now
        )
        
        # Create response
        response = TranscriptionResponse(
            text=result.text,
            segments=response_segments,
            language=result.language,
            language_probability=1.0,  # Default confidence
            metrics=metrics,
            request_id=request_id,
            session_id=request.session_id,
            model_info={
                "model": f"faster-whisper-{result.model_info['model_size']}",
                "device": result.model_info['device'],
                "compute_type": result.model_info['compute_type'],
                "performance": {
                    "rtf": result.rtf,
                    "latency_ms": result.latency_ms,
                    "meets_targets": result.meets_performance_targets()
                }
            }
        )
        
        performance_status = "✅ MEETS TARGETS" if result.meets_performance_targets() else "⚠️ BELOW TARGETS"
        logger.info(f"Request {request_id} completed: RTF={result.rtf:.3f}x, "
                   f"Latency={result.latency_ms:.0f}ms, text_length={len(result.text)} {performance_status}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request {request_id} failed: {e}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                "PROCESSING_ERROR",
                f"Internal server error: {str(e)}",
                request_id,
                {"traceback": traceback.format_exc()}
            ).dict()
        )


@app.post("/infer/batch", response_model=BatchTranscriptionResponse)
async def infer_batch(request: BatchTranscriptionRequest):
    """
    High-performance batch inference for 8x real-time transcription speed
    
    Optimized for processing multiple audio files in parallel with maximum throughput.
    Target: 8x real-time speed (RTF ≤ 0.125x)
    """
    batch_id = request.batch_id or str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"Starting high-performance batch {batch_id} with {len(request.audio_files)} files")
        logger.info(f"🎯 Target: 8x real-time speed (RTF ≤ 0.125x)")
        
        # Check if batch service is available
        try:
            batch_service = get_batch_service()
        except STTServiceError:
            logger.warning("Batch service not available, falling back to sequential processing")
            return await transcribe_batch_fallback(request)
        
        # Validate batch service health
        if not batch_service.is_healthy():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=create_error_response(
                    "BATCH_SERVICE_UNHEALTHY",
                    "Batch processing service is not healthy",
                    batch_id
                ).dict()
            )
        
        # Process batch using high-performance service
        batch_result = await batch_service.process_batch(
            audio_files=request.audio_files,
            audio_format=request.audio_format.value,
            language=request.language.value,
            beam_size=request.beam_size,
            enable_timestamps=request.enable_timestamps,
            enable_confidence=request.enable_confidence,
            batch_id=batch_id
        )
        
        # Convert results to API response format
        api_results = []
        for stt_result in batch_result.results:
            # Convert STTResult to TranscriptionResponse
            segments = []
            for i, segment_dict in enumerate(stt_result.segments):
                words = []
                if 'words' in segment_dict:
                    for word_dict in segment_dict['words']:
                        words.append(WordSegment(
                            word=word_dict['word'],
                            start=word_dict['start'],
                            end=word_dict['end'],
                            confidence=word_dict.get('confidence', 0.0)
                        ))
                
                segments.append(TranscriptionSegment(
                    id=i,
                    text=segment_dict['text'],
                    start=segment_dict['start'],
                    end=segment_dict['end'],
                    confidence=segment_dict.get('confidence', 0.0),
                    words=words if words else None
                ))
            
            api_result = TranscriptionResponse(
                text=stt_result.text,
                segments=segments,
                language=stt_result.language,
                language_probability=1.0,
                metrics=ProcessingMetrics(
                    total_duration=stt_result.processing_time,
                    audio_duration=stt_result.audio_duration,
                    rtf=stt_result.rtf,
                    inference_time=stt_result.processing_time
                ),
                request_id=f"{batch_id}_{len(api_results)}",
                model_info=stt_result.model_info
            )
            api_results.append(api_result)
        
        # Create final response
        response = BatchTranscriptionResponse(
            results=api_results,
            total_files=batch_result.total_files,
            successful_files=batch_result.successful_files,
            failed_files=batch_result.failed_files,
            total_processing_time=batch_result.total_processing_time,
            average_rtf=batch_result.average_rtf,
            batch_id=batch_id
        )
        
        # Log performance results
        performance_status = "✅ MEETS 8X TARGET" if batch_result.meets_performance_targets else "⚠️ BELOW 8X TARGET"
        logger.info(
            f"Batch {batch_id} completed: {batch_result.successful_files}/{batch_result.total_files} successful, "
            f"RTF={batch_result.average_rtf:.3f}x ({batch_result.throughput_speedup:.1f}x speedup), "
            f"Workers={batch_result.concurrent_workers_used} {performance_status}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch {batch_id} processing failed: {e}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                "BATCH_PROCESSING_ERROR",
                f"High-performance batch processing failed: {str(e)}",
                batch_id,
                {"traceback": traceback.format_exc()}
            ).dict()
        )


@app.post("/batch/transcribe", response_model=BatchTranscriptionResponse)
async def transcribe_batch_fallback(request: BatchTranscriptionRequest):
    """
    Fallback batch transcription endpoint (sequential processing)
    
    Used when high-performance batch service is unavailable.
    """
    batch_id = request.batch_id or str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"Processing fallback batch {batch_id} with {len(request.audio_files)} files")
        
        # Validate STT service is ready
        stt_service = get_stt_service()
        if not stt_service.is_healthy():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=create_error_response(
                    "STT_SERVICE_UNHEALTHY",
                    "Speech recognition service is not healthy",
                    batch_id
                ).dict()
            )
        
        # Process files sequentially
        results = []
        successful_files = 0
        failed_files = 0
        
        for i, audio_data in enumerate(request.audio_files):
            try:
                # Create individual request
                individual_request = TranscriptionRequest(
                    audio_data=audio_data,
                    audio_format=request.audio_format,
                    language=request.language,
                    enable_timestamps=request.enable_timestamps,
                    enable_confidence=request.enable_confidence,
                    beam_size=request.beam_size,
                    request_id=f"{batch_id}_{i}"
                )
                
                # Process individual file
                result = await transcribe_utterance(individual_request)
                results.append(result)
                successful_files += 1
                
            except Exception as e:
                logger.error(f"Failed to process file {i} in batch {batch_id}: {e}")
                failed_files += 1
                
                # Create error result
                error_result = TranscriptionResponse(
                    text="",
                    segments=[],
                    language="unknown",
                    language_probability=0.0,
                    metrics=ProcessingMetrics(
                        total_duration=0.0,
                        audio_duration=0.0,
                        rtf=0.0,
                        inference_time=0.0
                    ),
                    request_id=f"{batch_id}_{i}",
                    model_info={}
                )
                results.append(error_result)
        
        # Calculate batch metrics
        total_processing_time = time.time() - start_time
        total_audio_duration = sum(r.metrics.audio_duration for r in results)
        average_rtf = (
            sum(r.metrics.rtf * r.metrics.audio_duration for r in results) / total_audio_duration
            if total_audio_duration > 0 else 0.0
        )
        
        response = BatchTranscriptionResponse(
            results=results,
            total_files=len(request.audio_files),
            successful_files=successful_files,
            failed_files=failed_files,
            total_processing_time=total_processing_time,
            average_rtf=average_rtf,
            batch_id=batch_id
        )
        
        logger.info(f"Fallback batch {batch_id} completed: {successful_files}/{len(request.audio_files)} successful")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fallback batch {batch_id} failed: {e}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                "BATCH_PROCESSING_ERROR",
                f"Batch processing failed: {str(e)}",
                batch_id,
                {"traceback": traceback.format_exc()}
            ).dict()
        )


@app.get("/stats", response_model=dict)
async def get_server_stats():
    """Get server statistics"""
    uptime = time.time() - server_stats["start_time"]
    
    stats = {
        "uptime_seconds": uptime,
        "total_requests": server_stats["total_requests"],
        "successful_requests": server_stats["successful_requests"],
        "failed_requests": server_stats["failed_requests"],
        "success_rate": (
            server_stats["successful_requests"] / server_stats["total_requests"]
            if server_stats["total_requests"] > 0 else 0.0
        ),
        "average_processing_time": (
            server_stats["total_processing_time"] / server_stats["total_requests"]
            if server_stats["total_requests"] > 0 else 0.0
        ),
        "total_audio_duration": server_stats["total_audio_duration"],
        "average_rtf": (
            server_stats["total_processing_time"] / server_stats["total_audio_duration"]
            if server_stats["total_audio_duration"] > 0 else 0.0
        ),
        "requests_per_minute": (
            server_stats["total_requests"] / uptime * 60
            if uptime > 0 else 0.0
        )
    }
    
    return stats


# Development server
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="STT API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=False
    ) 