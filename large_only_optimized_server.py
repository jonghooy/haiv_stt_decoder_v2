#!/usr/bin/env python3
"""
Large Only Optimized STT Server
Large-v3 모델 전용 극한 최적화 STT 서버
"""

import asyncio
import logging
import time
import torch
import numpy as np
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
import argparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 로컬 모듈 임포트
from src.api.stt_service import FasterWhisperSTTService
from src.utils.audio_utils import AudioUtils

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Data Models ====================

class STTRequest(BaseModel):
    audio_data: str  # base64 encoded
    language: str = "ko"
    audio_format: str = "pcm_16khz"  # 지원 포맷: pcm_16khz 만 지원
    vad_enabled: bool = True  # VAD(Voice Activity Detection) 사용 여부

class STTWithKeywordsRequest(BaseModel):
    audio_data: str  # base64 encoded
    language: str = "ko"
    audio_format: str = "pcm_16khz"
    keywords: List[str] = []  # 부스팅할 키워드 목록
    keyword_boost: float = 2.0  # 키워드 부스팅 강도 (1.0-5.0)
    vad_enabled: bool = True  # VAD(Voice Activity Detection) 사용 여부

class STTResponse(BaseModel):
    text: str
    language: str
    rtf: float
    processing_time: float
    confidence: float = 0.0
    audio_duration: float = 0.0
    audio_format: str = "pcm_16khz"
    vad_enabled: bool = True  # VAD 사용 여부

class KeywordBoostResponse(BaseModel):
    text: str
    language: str
    rtf: float
    processing_time: float
    confidence: float = 0.0
    keywords_detected: List[str] = []  # 감지된 키워드 목록
    boost_applied: bool = False  # 부스팅 적용 여부
    audio_duration: float = 0.0
    audio_format: str = "pcm_16khz"
    vad_enabled: bool = True  # VAD 사용 여부

class HealthResponse(BaseModel):
    status: str
    gpu_info: Dict[str, Any]
    model_info: Dict[str, Any]
    optimization_status: Dict[str, Any]

# ==================== Main STT Service ====================

class LargeOnlyOptimizedSTTService:
    """Large-v3 모델 전용 극한 최적화 STT 서비스"""
    
    def __init__(self):
        self.model = None
        self.gpu_info = None
        self._setup_gpu_optimizations()
        
    def _setup_gpu_optimizations(self):
        """극한 GPU 최적화 설정"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA가 사용 불가능합니다")
            
        logger.info("🚀 Large 모델 전용 극한 GPU 최적화 적용 중...")
        
        # GPU 메모리 설정 (PyTorch 2.5+ 호환)
        try:
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(0.95)  # GPU 메모리 95% 사용
                logger.info("✅ CUDA 메모리 fraction 설정 완료")
            else:
                # PyTorch 2.5+ 호환 메모리 설정
                torch.cuda.memory.set_per_process_memory_fraction(0.95)
                logger.info("✅ CUDA 프로세스별 메모리 fraction 설정 완료")
        except Exception as e:
            logger.warning(f"⚠️ CUDA 메모리 fraction 설정 실패: {e}")
            # 기본 메모리 정리만 수행
            torch.cuda.empty_cache()
            
        torch.backends.cudnn.benchmark = True  # cuDNN 벤치마크 활성화
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 활성화
        torch.backends.cudnn.allow_tf32 = True
        
        # CUDA 메모리 풀 최적화 (PyTorch 2.5+ 호환)
        try:
            if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, 'set_memory_pool_limit'):
                torch.cuda.memory.set_memory_pool_limit(0.95)
                logger.info("✅ CUDA 메모리 풀 제한 설정 완료")
            else:
                # PyTorch 2.5+ 호환 메모리 최적화
                torch.cuda.empty_cache()
                logger.info("✅ CUDA 메모리 캐시 정리 완료")
        except Exception as e:
            logger.warning(f"⚠️ CUDA 메모리 설정 실패: {e}")
            torch.cuda.empty_cache()  # 최소한 메모리 정리는 수행
                
        # Mixed precision 활성화
        torch.backends.cuda.enable_flash_sdp(True)
        
        # GPU 정보 수집
        self.gpu_info = {
            "device": torch.cuda.get_device_name(),
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__
        }
        
        logger.info("✅ 극한 GPU 최적화 완료")
        
    async def _load_large_model(self):
        """Large-v3 모델 로딩"""
        try:
            logger.info("📦 Large-v3 모델 로딩 중 (float16 최적화)...")
            
            self.model = FasterWhisperSTTService(
                model_size="large-v3",
                device="cuda",
                compute_type="float16"
            )
            
            # STT 서비스 초기화
            logger.info("🔧 STT 서비스 초기화 중...")
            initialized = await self.model.initialize()
            if not initialized:
                raise RuntimeError("STT 서비스 초기화 실패")
            
            logger.info("✅ Large-v3 모델 로드 및 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ Large-v3 모델 로드 실패: {e}")
            raise
            
    async def _warmup_model(self):
        """모델 웜업"""
        try:
            logger.info("🔥 모델 웜업 시작...")
            
            # 더미 오디오 데이터 생성 (1초, 16kHz)
            dummy_audio = np.random.randn(16000).astype(np.float32) * 0.1  # 작은 볼륨
            
            # 모델의 내부 transcribe 메서드를 직접 사용 (가장 안전함)
            start_time = time.time()
            # STT 서비스의 내부 모델에 직접 접근
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'transcribe'):
                # FasterWhisper 모델의 transcribe 메서드 직접 호출
                segments, info = self.model.model.transcribe(
                    dummy_audio,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    vad_filter=False,
                    language="ko"
                )
                # 결과 소비
                list(segments)
            else:
                logger.info("웜업을 위한 직접 모델 접근 불가, 웜업 건너뜀")
                
            warmup_time = time.time() - start_time
            
            logger.info(f"✅ 모델 웜업 완료 ({warmup_time:.3f}초)")
            
        except Exception as e:
            logger.warning(f"⚠️ 모델 웜업 실패 (비중요): {e}")
            # 웜업 실패는 서버 시작을 막지 않음
            pass

    async def transcribe(self, audio_data: bytes, language: str = "ko", audio_format: str = "pcm_16khz", vad_enabled: bool = True) -> Dict[str, Any]:
        """오디오 전사 처리"""
        start_time = time.time()
        
        try:
            # PCM 16kHz 전용 - 오디오 길이 계산
            if audio_format == "pcm_16khz":
                # 16kHz, 16bit (2 bytes per sample)
                audio_duration = len(audio_data) / (16000 * 2)
            else:
                # PCM 16kHz만 지원
                raise ValueError(f"지원하지 않는 오디오 포맷: {audio_format}. PCM 16kHz만 지원됩니다.")
            
            # PCM 16kHz 바이트 데이터를 numpy 배열로 변환
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # FasterWhisper 모델에 직접 전사 요청
            if not self.model or not hasattr(self.model, 'model'):
                raise ValueError("STT 모델이 초기화되지 않았습니다")
                
            segments, info = self.model.model.transcribe(
                audio_array,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                language=language,
                vad_filter=vad_enabled
            )
            
            # 결과를 텍스트로 변환
            result_text = " ".join([segment.text for segment in segments])
            
            # STTResult 형태로 변환
            class SimpleResult:
                def __init__(self, text):
                    self.text = text
                    
            result = SimpleResult(result_text)
            
            processing_time = time.time() - start_time
            rtf = processing_time / max(audio_duration, 0.001)
            
            # 결과 텍스트 추출
            result_text = result.text if hasattr(result, 'text') else str(result)
            # 텍스트가 너무 길면 앞부분만 표시
            display_text = result_text[:50] + "..." if len(result_text) > 50 else result_text
            
            logger.info(
                f"✅ 전사 완료 - VAD: {vad_enabled}, 오디오: {audio_duration:.3f}초, "
                f"처리시간: {processing_time:.3f}초, RTF: {rtf:.4f}, 텍스트: \"{display_text}\""
            )
            
            return {
                "text": result.text if hasattr(result, 'text') else str(result),
                "language": language,
                "rtf": rtf,
                "processing_time": processing_time,
                "confidence": result.model_info.get("confidence", 0.0) if hasattr(result, 'model_info') else 0.0,
                "audio_duration": audio_duration,
                "audio_format": audio_format,
                "vad_enabled": vad_enabled
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 전사 실패: {e}, 처리시간: {processing_time:.3f}초")
            raise HTTPException(status_code=500, detail=f"전사 처리 실패: {str(e)}")

    def get_status(self) -> Dict[str, Any]:
        """서비스 상태 반환"""
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
        
        return {
            "status": "healthy" if self.model else "loading",
            "gpu_info": {
                **self.gpu_info,
                "memory_allocated": f"{gpu_memory_allocated:.2f}GB",
                "memory_reserved": f"{gpu_memory_reserved:.2f}GB"
            },
            "model_info": {
                "model": "large-v3",
                "compute_type": "float16",
                "device": "cuda",
                "loaded": self.model is not None
            },
            "optimization_status": {
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
                "memory_fraction": 0.95
            }
        }

# ==================== FastAPI App ====================

def create_app() -> FastAPI:
    """FastAPI 앱 생성"""
    app = FastAPI(
        title="Large Only Optimized STT Server",
        description="Large-v3 모델 전용 극한 최적화 STT 서버",
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
    
    # STT 서비스 인스턴스
    stt_service = None
    
    @app.on_event("startup")
    async def startup_event():
        nonlocal stt_service
        try:
            logger.info("🚀 Large Only 극한 최적화 STT 서버 시작 중...")
            
            # GPU 정보 출력
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA 버전: {torch.version.cuda}")
            logger.info(f"PyTorch 버전: {torch.__version__}")
            logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # STT 서비스 초기화
            stt_service = LargeOnlyOptimizedSTTService()
            await stt_service._load_large_model()
            await stt_service._warmup_model()
            
            logger.info("✅ 서버 시작 완료")
            
        except Exception as e:
            logger.error(f"❌ 서버 시작 실패: {e}")
            raise

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """헬스 체크"""
        if not stt_service:
            raise HTTPException(status_code=503, detail="서비스가 초기화되지 않았습니다")
        return stt_service.get_status()

    @app.post("/transcribe", response_model=STTResponse)
    async def transcribe_audio(request: STTRequest):
        """오디오 전사"""
        if not stt_service:
            raise HTTPException(status_code=503, detail="STT 서비스가 초기화되지 않았습니다")
        
        try:
            # Base64 디코딩
            audio_bytes = base64.b64decode(request.audio_data)
            
            # 전사 처리
            result = await stt_service.transcribe(
                audio_data=audio_bytes,
                language=request.language,
                audio_format=request.audio_format,
                vad_enabled=request.vad_enabled
            )
            
            return STTResponse(**result)
            
        except Exception as e:
            logger.error(f"❌ 전사 요청 처리 실패: {e}")
            raise HTTPException(status_code=500, detail=f"전사 처리 실패: {str(e)}")

    @app.post("/transcribe/keywords", response_model=KeywordBoostResponse)
    async def transcribe_with_keywords(request: STTWithKeywordsRequest):
        """키워드 부스팅 전사 (현재는 기본 전사와 동일)"""
        if not stt_service:
            raise HTTPException(status_code=503, detail="STT 서비스가 초기화되지 않았습니다")
        
        try:
            # Base64 디코딩
            audio_bytes = base64.b64decode(request.audio_data)
            
            # 전사 처리
            result = await stt_service.transcribe(
                audio_data=audio_bytes,
                language=request.language,
                audio_format=request.audio_format,
                vad_enabled=request.vad_enabled
            )
            
            # 키워드 부스팅 응답 형태로 변환
            return KeywordBoostResponse(
                text=result["text"],
                language=result["language"],
                rtf=result["rtf"],
                processing_time=result["processing_time"],
                confidence=result["confidence"],
                keywords_detected=[],  # 현재는 키워드 감지 미구현
                boost_applied=False,  # 현재는 부스팅 미구현
                audio_duration=result["audio_duration"],
                audio_format=result["audio_format"],
                vad_enabled=result["vad_enabled"]
            )
            
        except Exception as e:
            logger.error(f"❌ 키워드 부스팅 전사 요청 처리 실패: {e}")
            raise HTTPException(status_code=500, detail=f"전사 처리 실패: {str(e)}")

    return app

# Gunicorn이 찾을 수 있도록 앱 인스턴스를 전역 스코프에서 생성합니다.
app = create_app()

# ==================== Main ====================

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Large Only Optimized STT Server")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8003, help="서버 포트")
    
    args = parser.parse_args()
    
    # 서버 실행
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,  # 단일 워커로 실행 (GPU 공유 문제 방지)
        loop="asyncio"
    )

if __name__ == "__main__":
    main() 