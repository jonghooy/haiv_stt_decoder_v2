#!/usr/bin/env python3
"""
GPU Optimized STT Server
cuDNN을 완전히 활성화하고 RTX 4090 최적화를 적용한 STT API 서버
"""

import sys
import os
sys.path.append('/home/jonghooy/haiv_stt_decoder_v2')

import asyncio
import time
import torch
import torchaudio
import numpy as np
import logging
import base64
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# GPU 최적화 설정 - cuDNN 완전 활성화
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from src.api.stt_service import FasterWhisperSTTService

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic 모델
class TranscriptionRequest(BaseModel):
    audio_data: str  # base64 인코딩된 오디오 데이터
    language: Optional[str] = "ko"
    audio_format: Optional[str] = "pcm_16khz"

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    rtf: float
    processing_time: float
    audio_duration: float
    gpu_optimized: bool
    model_load_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    model_loaded: bool
    cudnn_enabled: bool
    gpu_name: Optional[str] = None

# FastAPI 앱 생성
app = FastAPI(
    title="GPU Optimized STT API",
    description="cuDNN 활성화된 고성능 음성 인식 API 서버",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 STT 서비스
stt_service: Optional[FasterWhisperSTTService] = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 STT 서비스 초기화"""
    global stt_service
    try:
        logger.info("🚀 GPU Optimized STT Server 시작 중...")
        logger.info(f"cuDNN 활성화 상태: {torch.backends.cudnn.enabled}")
        logger.info(f"cuDNN 벤치마크 모드: {torch.backends.cudnn.benchmark}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA 버전: {torch.version.cuda}")
            logger.info(f"PyTorch 버전: {torch.__version__}")
        
        # STT 서비스 생성 및 즉시 초기화
        stt_service = FasterWhisperSTTService()
        logger.info("📦 STT 모델 로딩 중...")
        
        # 모델을 미리 로드하여 첫 번째 요청 지연 제거
        start_time = time.time()
        await stt_service.initialize()
        load_time = time.time() - start_time
        
        logger.info(f"✅ STT 서비스 초기화 완료 - 모델 로딩 시간: {load_time:.2f}초")
        logger.info(f"GPU 사용 가능: {torch.cuda.is_available()}")
        
    except Exception as e:
        logger.error(f"❌ 서버 초기화 실패: {e}")
        raise

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "GPU Optimized STT API Server", 
        "status": "running",
        "features": {
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크 엔드포인트"""
    model_loaded = False
    if stt_service is not None:
        model_loaded = hasattr(stt_service, 'model') and stt_service.model is not None
        
    return HealthResponse(
        status="healthy",
        gpu_available=torch.cuda.is_available(),
        model_loaded=model_loaded,
        cudnn_enabled=torch.backends.cudnn.enabled,
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """오디오 전사 엔드포인트 (JSON)"""
    if stt_service is None:
        raise HTTPException(status_code=500, detail="STT 서비스가 초기화되지 않았습니다")
    
    try:
        start_time = time.time()
        
        # 오디오 전사 실행
        result = await stt_service.transcribe_audio(
            audio_data=request.audio_data,
            audio_format=request.audio_format,
            language=request.language
        )
        
        processing_time = time.time() - start_time
        
        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            rtf=result.rtf,
            processing_time=processing_time,
            audio_duration=result.audio_duration,
            gpu_optimized=torch.cuda.is_available() and torch.backends.cudnn.enabled
        )
        
    except Exception as e:
        logger.error(f"전사 실패: {e}")
        raise HTTPException(status_code=500, detail=f"전사 실패: {str(e)}")

@app.post("/transcribe/file", response_model=TranscriptionResponse)
async def transcribe_file(
    audio: UploadFile = File(...),
    language: str = Form("ko"),
    vad_filter: bool = Form(False)
):
    """오디오 파일 전사 엔드포인트"""
    if stt_service is None:
        raise HTTPException(status_code=500, detail="STT 서비스가 초기화되지 않았습니다")
    
    try:
        start_time = time.time()
        
        # 파일 읽기
        audio_bytes = await audio.read()
        
        # 전사 실행
        result = await stt_service.transcribe_file_bytes(
            audio_bytes=audio_bytes,
            language=language,
            vad_filter=vad_filter
        )
        
        processing_time = time.time() - start_time
        
        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            rtf=result.rtf,
            processing_time=processing_time,
            audio_duration=result.audio_duration,
            gpu_optimized=torch.cuda.is_available() and torch.backends.cudnn.enabled
        )
        
    except Exception as e:
        logger.error(f"파일 전사 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 전사 실패: {str(e)}")

if __name__ == "__main__":
    # 서버 실행
    uvicorn.run(
        "gpu_optimized_stt_server:app",
        host="0.0.0.0",
        port=8001,  # 다른 포트 사용
        log_level="info",
        reload=False
    ) 