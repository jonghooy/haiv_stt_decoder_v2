#!/usr/bin/env python3
"""
GPU Optimized STT Server with Queueing System
cuDNN을 완전히 활성화하고 RTX 4090 최적화를 적용한 STT API 서버
20개 이하 클라이언트를 위한 지능형 큐잉 시스템 포함
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
import uuid
import json
import tempfile
import shutil
import zipfile
from collections import defaultdict, deque, Counter
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Callable, Set
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import re

# GPU 최적화 설정 - Large-v3 모델 전용 극한 최적화
def setup_extreme_gpu_optimizations():
    """Large-v3 모델 전용 극한 GPU 최적화"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA가 사용 불가능합니다")
        
    logger.info("🚀 Large-v3 모델 전용 극한 GPU 최적화 적용 중...")
    
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
        
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # cuDNN 벤치마크 활성화
    torch.backends.cudnn.deterministic = False
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
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("✅ Flash Attention SDP 활성화 완료")
    except Exception as e:
        logger.warning(f"⚠️ Flash Attention SDP 활성화 실패: {e}")
    
    logger.info("✅ 극한 GPU 최적화 완료")

async def warmup_large_model(stt_service):
    """Large-v3 모델 웜업 (첫 요청 지연 최소화)"""
    try:
        logger.info("🔥 Large-v3 모델 웜업 시작...")
        
        # 더미 오디오 데이터 생성 (1초, 16kHz)
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.1  # 작은 볼륨
        
        # 모델의 내부 transcribe 메서드를 직접 사용 (가장 안전함)
        start_time = time.time()
        # STT 서비스의 내부 모델에 안전하게 접근
        model = None
        
        # 1. 직접 model 속성이 있는 경우 (NeMo 등)
        if hasattr(stt_service, 'model') and hasattr(stt_service.model, 'transcribe'):
            model = stt_service.model
        # 2. WhisperSTTServiceAdapter의 경우
        elif hasattr(stt_service, 'whisper_service') and hasattr(stt_service.whisper_service, 'model'):
            model = stt_service.whisper_service.model
        # 3. 기타 어댑터 패턴
        elif hasattr(stt_service, 'service') and hasattr(stt_service.service, 'model'):
            model = stt_service.service.model
        
        if model and hasattr(model, 'transcribe'):
            # FasterWhisper 모델의 transcribe 메서드 직접 호출 (Large-v3 최적화)
            segments, info = model.transcribe(
                dummy_audio,
                beam_size=5,  # Large-v3에 최적화된 beam_size
                best_of=5,    # Large-v3에 최적화된 best_of
                temperature=0.0,
                vad_filter=False,
                language="ko"
            )
            # 결과 소비
            list(segments)
        else:
            logger.info("웜업을 위한 직접 모델 접근 불가, 웜업 건너뜀")
            
        warmup_time = time.time() - start_time
        
        logger.info(f"✅ Large-v3 모델 웜업 완료 ({warmup_time:.3f}초)")
        
    except Exception as e:
        logger.warning(f"⚠️ 모델 웜업 실패 (비중요): {e}")
        # 웜업 실패는 서버 시작을 막지 않음
        pass

# STT 서비스 관련 imports
from src.api.stt_service import FasterWhisperSTTService
from src.api.base_stt_service import BaseSTTService
from src.api.stt_factory import STTServiceFactory

from src.api.post_processing_correction import (
    get_post_processing_corrector,
    apply_keyword_correction,
    CorrectionResult
)
from src.api.models import (
    KeywordRegistrationRequest,
    KeywordCorrectionRequest,
    KeywordCorrectionResponse,
    TranscriptionWithCorrection,
    KeywordStatsResponse,
    ProcessingMetrics
)

# 로깅 설정 - 디버깅을 위해 DEBUG 레벨로 설정, 파일과 콘솔 동시 출력
import logging.handlers

# 로그 디렉토리 생성
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 로그 파일 경로
log_file = log_dir / "stt_server_debug.log"

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 포맷터 설정
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 파일 핸들러 (회전 로그 파일)
file_handler = logging.handlers.RotatingFileHandler(
    log_file, 
    maxBytes=50*1024*1024,  # 50MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# 콘솔 핸들러
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 콘솔은 INFO 레벨만
console_handler.setFormatter(formatter)

# 핸들러 추가
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 루트 로거도 설정
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().addHandler(file_handler)

# 극한 GPU 최적화 실행 (logger 정의 후)
setup_extreme_gpu_optimizations()

# ============================================================================
# 모델 선택을 위한 전역 변수 및 설정
# ============================================================================

# 서버 설정을 위한 전역 변수들 (argparse로 설정됨)
SERVER_MODEL_TYPE = "whisper"  # 기본값: whisper
SERVER_MODEL_NAME = "large-v3"  # 기본값: large-v3
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8004

# ============================================================================
# 지능형 큐잉 시스템 구현
# ============================================================================

class RequestPriority(Enum):
    """요청 우선순위"""
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RequestStatus(Enum):
    """요청 상태"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class QueuedRequest:
    """큐에 들어간 요청"""
    request_id: str
    client_id: str
    audio_data: str
    language: str
    audio_format: str
    priority: RequestPriority
    created_at: datetime
    timeout_at: datetime
    status: RequestStatus = RequestStatus.QUEUED
    processing_started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """요청 ID가 없으면 자동 생성"""
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
    
    def __lt__(self, other):
        """우선순위 큐에서 비교를 위한 메서드"""
        if not isinstance(other, QueuedRequest):
            return NotImplemented
        # 우선순위가 같으면 생성 시간으로 비교 (먼저 생성된 것이 우선)
        if self.priority.value == other.priority.value:
            return self.created_at < other.created_at
        return self.priority.value < other.priority.value

@dataclass
class QueueStats:
    """큐 통계"""
    total_requests: int = 0
    queued_requests: int = 0
    processing_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    average_wait_time: float = 0.0
    average_processing_time: float = 0.0
    current_throughput: float = 0.0

# Helper functions
def logprob_to_confidence(avg_logprob: float) -> float:
    """
    Whisper의 avg_logprob을 신뢰도 점수(0.0-1.0)로 변환 (개선된 버전)
    
    Args:
        avg_logprob: Whisper 세그먼트의 평균 로그 확률
        
    Returns:
        0.0-1.0 범위의 신뢰도 점수
    """
    if avg_logprob is None:
        return 0.0
    
    # avg_logprob을 더 세밀하게 구간별로 매핑
    # 실제 Whisper 데이터를 기반으로 조정된 구간들
    if avg_logprob >= -0.1:
        # 매우 높은 신뢰도: -0.1 이상
        return min(0.99, 0.92 + (avg_logprob + 0.1) * 0.7)
    elif avg_logprob >= -0.3:
        # 높은 신뢰도: -0.3 ~ -0.1
        return 0.85 + (avg_logprob + 0.3) * 0.35  # -0.3~-0.1 -> 0.85~0.92
    elif avg_logprob >= -0.6:
        # 중상 신뢰도: -0.6 ~ -0.3
        return 0.75 + (avg_logprob + 0.6) * 0.33  # -0.6~-0.3 -> 0.75~0.85
    elif avg_logprob >= -1.0:
        # 중간 신뢰도: -1.0 ~ -0.6
        return 0.60 + (avg_logprob + 1.0) * 0.375  # -1.0~-0.6 -> 0.60~0.75
    elif avg_logprob >= -1.5:
        # 중하 신뢰도: -1.5 ~ -1.0
        return 0.45 + (avg_logprob + 1.5) * 0.30   # -1.5~-1.0 -> 0.45~0.60
    elif avg_logprob >= -2.5:
        # 낮은 신뢰도: -2.5 ~ -1.5
        return 0.25 + (avg_logprob + 2.5) * 0.20   # -2.5~-1.5 -> 0.25~0.45
    else:
        # 매우 낮은 신뢰도: -2.5 이하
        return max(0.05, 0.25 + (avg_logprob + 2.5) * 0.08)


def calculate_segment_confidence_from_words(words: List[dict], avg_logprob: float) -> float:
    """
    단어 레벨 신뢰도를 기반으로 세그먼트 신뢰도 계산
    
    Args:
        words: 단어별 신뢰도 정보가 포함된 리스트
        avg_logprob: 세그먼트의 평균 로그 확률 (fallback용)
        
    Returns:
        계산된 세그먼트 신뢰도
    """
    if not words:
        # 단어 정보가 없으면 avg_logprob 기반으로 계산
        return logprob_to_confidence(avg_logprob)
    
    # 단어별 신뢰도 수집
    word_confidences = []
    for word in words:
        if isinstance(word, dict) and 'confidence' in word:
            confidence = word['confidence']
            if confidence is not None and confidence > 0:
                word_confidences.append(confidence)
    
    if not word_confidences:
        # 유효한 단어 신뢰도가 없으면 avg_logprob 사용
        return logprob_to_confidence(avg_logprob)
    
    # 가중 평균 계산 (낮은 신뢰도에 더 높은 가중치)
    # 이는 전체 세그먼트의 신뢰도를 보수적으로 평가하기 위함
    sorted_confidences = sorted(word_confidences)
    
    if len(sorted_confidences) == 1:
        return sorted_confidences[0]
    
    # 하위 30%, 중위 40%, 상위 30%로 가중치 부여
    n = len(sorted_confidences)
    lower_30_idx = max(1, int(n * 0.3))
    upper_70_idx = max(lower_30_idx + 1, int(n * 0.7))
    
    lower_30 = sorted_confidences[:lower_30_idx]
    middle_40 = sorted_confidences[lower_30_idx:upper_70_idx]
    upper_30 = sorted_confidences[upper_70_idx:]
    
    # 가중 평균 (낮은 신뢰도에 더 높은 가중치)
    weighted_sum = 0.0
    total_weight = 0.0
    
    # 하위 30%: 가중치 0.5
    if lower_30:
        weighted_sum += sum(lower_30) * 0.5
        total_weight += len(lower_30) * 0.5
    
    # 중위 40%: 가중치 0.3  
    if middle_40:
        weighted_sum += sum(middle_40) * 0.3
        total_weight += len(middle_40) * 0.3
        
    # 상위 30%: 가중치 0.2
    if upper_30:
        weighted_sum += sum(upper_30) * 0.2
        total_weight += len(upper_30) * 0.2
    
    if total_weight > 0:
        final_confidence = weighted_sum / total_weight
        # avg_logprob와의 조합 (70% word-based, 30% logprob-based)
        logprob_confidence = logprob_to_confidence(avg_logprob)
        return final_confidence * 0.7 + logprob_confidence * 0.3
    else:
        return logprob_to_confidence(avg_logprob)


class IntelligentSTTQueue:
    """지능형 STT 처리 큐"""
    
    def __init__(self, 
                 max_concurrent: int = 8,
                 max_queue_size: int = 50,
                 default_timeout: int = 60,
                 priority_timeout: int = 30):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        self.priority_timeout = priority_timeout
        
        # 큐 관리
        self.priority_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.processing_requests: Dict[str, QueuedRequest] = {}
        self.completed_requests: Dict[str, QueuedRequest] = {}
        self.request_futures: Dict[str, asyncio.Future] = {}
        
        # 동시성 제어
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.processing_lock = asyncio.Lock()
        
        # 통계 및 모니터링
        self.stats = QueueStats()
        self.start_time = datetime.now()
        self.last_throughput_calc = datetime.now()
        self.recent_completions: List[datetime] = []
        
        # 백그라운드 작업
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """큐 시스템 시작"""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_requests())
        self._monitor_task = asyncio.create_task(self._monitor_performance())
        logger.info(f"🔄 STT 큐 시스템 시작 - 최대 동시 처리: {self.max_concurrent}개")
    
    async def stop(self):
        """큐 시스템 중지"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._monitor_task:
            self._monitor_task.cancel()
        
        # 모든 대기 중인 요청 취소
        while not self.priority_queue.empty():
            try:
                _, request = self.priority_queue.get_nowait()
                request.status = RequestStatus.CANCELLED
                if request.request_id in self.request_futures:
                    future = self.request_futures[request.request_id]
                    if not future.done():
                        future.cancel()
            except asyncio.QueueEmpty:
                break
        
        logger.info("🛑 STT 큐 시스템 중지")
    
    async def submit_request(self, 
                           audio_data: str,
                           language: str = "ko",
                           audio_format: str = "pcm_16khz",
                           client_id: str = None,
                           priority: RequestPriority = RequestPriority.MEDIUM,
                           timeout: Optional[int] = None) -> str:
        """요청을 큐에 제출"""
        
        # 큐 크기 확인
        current_queue_size = self.priority_queue.qsize()
        if current_queue_size >= self.max_queue_size:
            raise HTTPException(status_code=503, detail=f"Queue is full ({current_queue_size}/{self.max_queue_size})")
        
        # 요청 생성
        request_id = str(uuid.uuid4())
        client_id = client_id or f"client_{uuid.uuid4().hex[:8]}"
        timeout_seconds = timeout or (self.priority_timeout if priority == RequestPriority.HIGH else self.default_timeout)
        
        request = QueuedRequest(
            request_id=request_id,
            client_id=client_id,
            audio_data=audio_data,
            language=language,
            audio_format=audio_format,
            priority=priority,
            created_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=timeout_seconds)
        )
        
        # Future 생성
        future = asyncio.Future()
        self.request_futures[request_id] = future
        
        # 우선순위 큐에 추가 (우선순위가 높을수록 먼저 처리)
        await self.priority_queue.put((priority.value, request))
        
        # 통계 업데이트
        async with self.processing_lock:
            self.stats.total_requests += 1
            self.stats.queued_requests += 1
        
        logger.info(f"📝 요청 큐 추가 - ID: {request_id}, 클라이언트: {client_id}, 우선순위: {priority.name}, 큐 크기: {current_queue_size + 1}")
        
        return request_id
    
    async def get_request_result(self, request_id: str) -> Dict[str, Any]:
        """요청 결과 대기 및 반환"""
        if request_id not in self.request_futures:
            raise HTTPException(status_code=404, detail="Request not found")
        
        future = self.request_futures[request_id]
        
        try:
            # 결과 대기
            result = await future
            
            # 완료된 요청 정리
            if request_id in self.request_futures:
                del self.request_futures[request_id]
            
            return result
        
        except asyncio.CancelledError:
            raise HTTPException(status_code=408, detail="Request was cancelled")
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Request timed out")
    
    async def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """요청 상태 조회"""
        # 처리 중인 요청 확인
        if request_id in self.processing_requests:
            request = self.processing_requests[request_id]
            wait_time = (datetime.now() - request.created_at).total_seconds()
            processing_time = (datetime.now() - request.processing_started_at).total_seconds() if request.processing_started_at else 0
            
            return {
                "request_id": request_id,
                "status": request.status.value,
                "priority": request.priority.name,
                "wait_time": wait_time,
                "processing_time": processing_time,
                "estimated_remaining_time": max(0, request.timeout_at.timestamp() - datetime.now().timestamp())
            }
        
        # 완료된 요청 확인
        if request_id in self.completed_requests:
            request = self.completed_requests[request_id]
            total_time = (request.completed_at - request.created_at).total_seconds()
            
            return {
                "request_id": request_id,
                "status": request.status.value,
                "priority": request.priority.name,
                "total_time": total_time,
                "completed_at": request.completed_at.isoformat(),
                "result": request.result,
                "error_message": request.error_message
            }
        
        # 큐에서 대기 중인 요청 확인
        queue_position = await self._get_queue_position(request_id)
        if queue_position is not None:
            return {
                "request_id": request_id,
                "status": RequestStatus.QUEUED.value,
                "queue_position": queue_position,
                "estimated_wait_time": self._estimate_wait_time(queue_position)
            }
        
        raise HTTPException(status_code=404, detail="Request not found")
    
    async def cancel_request(self, request_id: str) -> bool:
        """요청 취소"""
        # Future가 있으면 취소
        if request_id in self.request_futures:
            future = self.request_futures[request_id]
            if not future.done():
                future.cancel()
                del self.request_futures[request_id]
                return True
        
        # 처리 중인 요청은 취소할 수 없음
        if request_id in self.processing_requests:
            return False
        
        return False
    
    async def get_queue_stats(self) -> QueueStats:
        """큐 통계 반환"""
        async with self.processing_lock:
            # 실시간 통계 업데이트
            self.stats.queued_requests = self.priority_queue.qsize()
            self.stats.processing_requests = len(self.processing_requests)
            
            # 처리량 계산
            now = datetime.now()
            time_diff = (now - self.last_throughput_calc).total_seconds()
            if time_diff >= 10:  # 10초마다 처리량 업데이트
                recent_count = len([t for t in self.recent_completions if (now - t).total_seconds() <= 60])
                self.stats.current_throughput = recent_count / 60.0  # 분당 처리량
                self.last_throughput_calc = now
            
            return self.stats
    
    async def process_requests(self, stt_service):
        """백그라운드에서 요청 처리"""
        while self._running:
            try:
                # 세마포어로 동시 처리 수 제한
                async with self.semaphore:
                    # 큐에서 요청 가져오기
                    try:
                        _, request = await asyncio.wait_for(self.priority_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    
                    # 타임아웃 확인
                    if datetime.now() > request.timeout_at:
                        request.status = RequestStatus.TIMEOUT
                        await self._complete_request(request, None, "Request timed out")
                        continue
                    
                    # 처리 시작
                    request.status = RequestStatus.PROCESSING
                    request.processing_started_at = datetime.now()
                    
                    async with self.processing_lock:
                        self.processing_requests[request.request_id] = request
                        self.stats.queued_requests = max(0, self.stats.queued_requests - 1)
                        self.stats.processing_requests += 1
                    
                    logger.info(f"🔄 요청 처리 시작 - ID: {request.request_id}")
                    
                    # 실제 STT 처리
                    try:
                        result = await self._process_stt_request(request, stt_service)
                        await self._complete_request(request, result, None)
                        
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"❌ 요청 처리 실패 - ID: {request.request_id}, 오류: {error_msg}")
                        await self._complete_request(request, None, error_msg)
            
            except Exception as e:
                logger.error(f"❌ 큐 처리 중 오류: {e}")
                await asyncio.sleep(1)
    
    async def _process_stt_request(self, request: QueuedRequest, stt_service) -> Dict[str, Any]:
        """실제 STT 처리"""
        start_time = time.time()
        
        # Base64 디코딩
        audio_bytes = base64.b64decode(request.audio_data)
        
        # STT 처리 - 올바른 메서드명 사용
        result = await stt_service.transcribe_file_bytes(
            audio_bytes, 
            language=request.language
        )
        
        processing_time = time.time() - start_time
        
        # STTResult 객체에서 필요한 정보 추출
        return {
            "text": result.text,
            "language": result.language,
            "rtf": result.rtf,
            "processing_time": processing_time,
            "audio_duration": result.audio_duration,
            "gpu_optimized": True,
            "queue_processed": True,
            "request_id": request.request_id,
            "client_id": request.client_id,
            "segments": result.segments
        }
    
    async def _complete_request(self, request: QueuedRequest, result: Optional[Dict[str, Any]], error_message: Optional[str]):
        """요청 완료 처리"""
        request.completed_at = datetime.now()
        request.result = result
        request.error_message = error_message
        request.status = RequestStatus.COMPLETED if result else RequestStatus.FAILED
        
        # 처리 중 목록에서 제거
        async with self.processing_lock:
            if request.request_id in self.processing_requests:
                del self.processing_requests[request.request_id]
                self.stats.processing_requests = max(0, self.stats.processing_requests - 1)
        
        # 완료 목록에 추가
        self.completed_requests[request.request_id] = request
        
        # 통계 업데이트
        if result:
            self.stats.completed_requests += 1
            self.recent_completions.append(datetime.now())
        else:
            self.stats.failed_requests += 1
        
        # Future 완료
        if request.request_id in self.request_futures:
            future = self.request_futures[request.request_id]
            if not future.done():
                if result:
                    future.set_result(result)
                else:
                    future.set_exception(Exception(error_message or "Unknown error"))
        
        logger.info(f"✅ 요청 완료 - ID: {request.request_id}, 상태: {request.status.value}")
    
    async def _cleanup_expired_requests(self):
        """만료된 요청 정리"""
        while self._running:
            try:
                await asyncio.sleep(30)  # 30초마다 정리
                
                now = datetime.now()
                expired_ids = []
                
                # 완료된 요청 중 1시간 이상 된 것 제거
                for request_id, request in list(self.completed_requests.items()):
                    if (now - request.completed_at).total_seconds() > 3600:  # 1시간
                        expired_ids.append(request_id)
                
                for request_id in expired_ids:
                    del self.completed_requests[request_id]
                
                # 최근 완료 기록 정리 (1시간 이상)
                self.recent_completions = [t for t in self.recent_completions if (now - t).total_seconds() <= 3600]
                
                if expired_ids:
                    logger.info(f"🧹 만료된 요청 {len(expired_ids)}개 정리 완료")
                
            except Exception as e:
                logger.error(f"❌ 요청 정리 중 오류: {e}")
    
    async def _monitor_performance(self):
        """성능 모니터링"""
        while self._running:
            try:
                await asyncio.sleep(60)  # 1분마다 모니터링
                
                stats = await self.get_queue_stats()
                logger.info(f"📊 큐 성능 모니터링:")
                logger.info(f"   대기 중: {stats.queued_requests}개")
                logger.info(f"   처리 중: {stats.processing_requests}개")
                logger.info(f"   완료: {stats.completed_requests}개")
                logger.info(f"   실패: {stats.failed_requests}개")
                logger.info(f"   처리량: {stats.current_throughput:.2f} 요청/분")
                
            except Exception as e:
                logger.error(f"❌ 성능 모니터링 중 오류: {e}")
    
    async def _get_queue_position(self, request_id: str) -> Optional[int]:
        """큐에서의 위치 확인"""
        # 실제 구현에서는 큐 내용을 확인해야 하지만,
        # PriorityQueue는 내용을 직접 확인할 수 없으므로 추정
        return self.priority_queue.qsize()
    
    def _estimate_wait_time(self, queue_position: int) -> float:
        """대기 시간 추정"""
        if self.stats.current_throughput > 0:
            return queue_position / (self.stats.current_throughput / 60.0)  # 초 단위
        return queue_position * 30  # 기본 30초/요청으로 추정

# 전역 큐 인스턴스
stt_queue: Optional[IntelligentSTTQueue] = None

# Pydantic 모델
class TranscriptionRequest(BaseModel):
    audio_data: str  # base64 인코딩된 오디오 데이터
    language: Optional[str] = "ko"
    audio_format: Optional[str] = "pcm_16khz"

class WordSegment(BaseModel):
    word: str
    start: float
    end: float
    confidence: Optional[float] = None

class SegmentInfo(BaseModel):
    id: int
    text: str
    start: float
    end: float
    confidence: Optional[float] = None
    words: Optional[List[WordSegment]] = None

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    rtf: float
    processing_time: float
    audio_duration: float
    gpu_optimized: bool
    model_load_time: Optional[float] = None
    segments: Optional[List[SegmentInfo]] = None

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    model_loaded: bool
    cudnn_enabled: bool
    gpu_name: Optional[str] = None
    gpu_info: Optional[Dict[str, Any]] = None
    optimization_status: Optional[Dict[str, Any]] = None

# 큐잉 시스템 관련 Pydantic 모델
class QueuedTranscriptionRequest(BaseModel):
    audio_data: str
    language: Optional[str] = "ko"
    audio_format: Optional[str] = "pcm_16khz"
    client_id: Optional[str] = None
    priority: Optional[str] = "medium"  # high, medium, low
    timeout: Optional[int] = None

class QueuedTranscriptionResponse(BaseModel):
    request_id: str
    status: str
    message: str
    estimated_wait_time: Optional[float] = None
    queue_position: Optional[int] = None

class QueueStatusResponse(BaseModel):
    request_id: str
    status: str
    priority: Optional[str] = None
    wait_time: Optional[float] = None
    processing_time: Optional[float] = None
    total_time: Optional[float] = None
    queue_position: Optional[int] = None
    estimated_wait_time: Optional[float] = None
    estimated_remaining_time: Optional[float] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class QueueStatsResponse(BaseModel):
    total_requests: int
    queued_requests: int
    processing_requests: int
    completed_requests: int
    failed_requests: int
    timeout_requests: int
    average_wait_time: float
    average_processing_time: float
    current_throughput: float
    queue_capacity: int
    max_concurrent: int

# 배치 처리 관련 모델들
class BatchTranscriptionRequest(BaseModel):
    """배치 STT 처리 요청"""
    language: Optional[str] = "ko"
    vad_filter: Optional[bool] = False
    enable_word_timestamps: Optional[bool] = True
    enable_confidence: Optional[bool] = True
    client_id: Optional[str] = None
    priority: Optional[str] = "medium"  # high, medium, low
    
    # 키워드 부스팅 관련 필드
    call_id: Optional[str] = None  # 키워드가 등록된 call_id
    enable_keyword_boosting: Optional[bool] = False  # 키워드 부스팅 활성화
    keywords: Optional[List[str]] = None  # 직접 키워드 리스트 제공
    keyword_boost_factor: Optional[float] = 2.0  # 키워드 부스팅 강도 (1.0-5.0)

class BatchFileInfo(BaseModel):
    """처리된 파일 정보"""
    filename: str
    size_bytes: int
    duration_seconds: float
    processing_time_seconds: float
    text: str
    language: str
    confidence: float
    segments: Optional[List[SegmentInfo]] = None

class BatchTranscriptionResponse(BaseModel):
    """배치 STT 처리 응답"""
    batch_id: str
    status: str
    message: str
    total_files: int
    processed_files: int
    failed_files: int
    total_duration: float
    total_processing_time: float
    created_at: str
    download_url: Optional[str] = None
    files: Optional[List[BatchFileInfo]] = None

class BatchStatusResponse(BaseModel):
    """배치 처리 상태 응답"""
    batch_id: str
    status: str  # processing, completed, failed, cancelled
    progress: float  # 0.0 - 1.0
    total_files: int
    processed_files: int
    failed_files: int
    estimated_remaining_time: Optional[float] = None
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

# FastAPI 앱 생성
app = FastAPI(
    title="Large-v3 극한 최적화 STT API",
    description="Large-v3 모델 전용 극한 GPU 최적화 음성 인식 API 서버 - float16, TF32, 95% GPU 메모리 활용",
    version="3.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 배치 처리 시스템 구현
# ============================================================================

import os
import json
import tempfile
import shutil
import io
from typing import Dict, List
from fastapi.responses import FileResponse
import zipfile

@dataclass
class BatchJob:
    """배치 작업 정보"""
    batch_id: str
    status: str = "processing"  # processing, completed, failed, cancelled
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_duration: float = 0.0
    total_processing_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_files: List[BatchFileInfo] = field(default_factory=list)
    temp_dir: Optional[str] = None
    download_path: Optional[str] = None

class BatchProcessor:
    """배치 STT 처리 관리자"""
    
    def __init__(self):
        self.batch_jobs: Dict[str, BatchJob] = {}
        self.processing_queue = asyncio.Queue()
        self.max_concurrent_batches = 2  # 동시 처리 가능한 배치 수
        self.current_processing = 0
        self.batch_temp_dir = "/tmp/stt_batch_processing"
        
        # 임시 디렉토리 생성
        os.makedirs(self.batch_temp_dir, exist_ok=True)
    
    async def submit_batch(self, files: List[UploadFile], request: BatchTranscriptionRequest) -> str:
        """배치 작업 제출"""
        batch_id = str(uuid.uuid4())
        
        # 임시 디렉토리 생성
        temp_dir = os.path.join(self.batch_temp_dir, batch_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # 배치 작업 생성
        batch_job = BatchJob(
            batch_id=batch_id,
            total_files=len(files),
            temp_dir=temp_dir
        )
        
        self.batch_jobs[batch_id] = batch_job
        
        # 파일들을 임시 디렉토리에 저장
        saved_files = []
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_files.append((file.filename, file_path, len(content)))
        
        # 백그라운드에서 배치 처리 시작
        asyncio.create_task(self._process_batch(batch_id, saved_files, request))
        
        return batch_id
    
    async def _process_batch(self, batch_id: str, files: List[tuple], request: BatchTranscriptionRequest):
        """배치 처리 실행"""
        batch_job = self.batch_jobs[batch_id]
        
        try:
            self.current_processing += 1
            logger.info(f"🔄 배치 {batch_id} 처리 시작 (파일 수: {len(files)})")
            
            # 시작 진행 상황 전송
            await progress_notifier.update_progress(batch_id, {
                "status": "processing",
                "progress": 0.0,
                "total_files": batch_job.total_files,
                "processed_files": 0,
                "failed_files": 0,
                "estimated_remaining_time": None,
                "current_file": None
            })
            
            start_time = time.time()
            
            # 🚀 병렬 처리로 개선: RTX 4090 GPU 성능 활용
            # GPU 메모리와 처리 능력에 따라 동시 처리할 수 있는 파일 수 결정
            max_concurrent = min(4, len(files))  # RTX 4090: 최대 4개 파일 동시 처리
            
            logger.info(f"🚀 병렬 처리 시작: {len(files)}개 파일을 최대 {max_concurrent}개씩 동시 처리")
            
            # 세마포어로 동시 처리 제한
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_single_file(filename: str, file_path: str, file_size: int, file_index: int):
                """단일 파일 처리 (세마포어로 동시 처리 제한)"""
                async with semaphore:
                    try:
                        logger.info(f"🔄 파일 처리 시작: {filename} ({file_index + 1}/{len(files)})")
                        
                        # 오디오 파일 처리
                        file_info = await self._process_audio_file(
                            filename, file_path, file_size, request
                        )
                        
                        # Thread-safe 업데이트
                        batch_job.result_files.append(file_info)
                        batch_job.processed_files += 1
                        batch_job.total_duration += file_info.duration_seconds
                        batch_job.total_processing_time += file_info.processing_time_seconds
                        
                        logger.info(f"✅ 파일 처리 완료: {filename} ({batch_job.processed_files}/{batch_job.total_files}) RTF: {file_info.processing_time_seconds/file_info.duration_seconds:.3f}")
                        
                        # 진행 상황 업데이트
                        await progress_notifier.update_progress(batch_id, {
                            "status": "processing",
                            "progress": batch_job.processed_files / batch_job.total_files,
                            "total_files": batch_job.total_files,
                            "processed_files": batch_job.processed_files,
                            "failed_files": batch_job.failed_files,
                            "current_file": f"처리 완료: {filename}",
                            "total_duration": batch_job.total_duration,
                            "total_processing_time": batch_job.total_processing_time,
                            "estimated_remaining_time": self._estimate_remaining_time(
                                start_time, batch_job.processed_files, len(files)
                            )
                        })
                        
                        return file_info
                        
                    except Exception as e:
                        batch_job.failed_files += 1
                        logger.error(f"❌ 파일 처리 실패: {filename} - {str(e)}")
                        
                        # 실패 상황 업데이트
                        await progress_notifier.update_progress(batch_id, {
                            "status": "processing",
                            "progress": (batch_job.processed_files + batch_job.failed_files) / batch_job.total_files,
                            "total_files": batch_job.total_files,
                            "processed_files": batch_job.processed_files,
                            "failed_files": batch_job.failed_files,
                            "current_file": f"{filename} (실패)",
                            "last_error": str(e),
                            "estimated_remaining_time": self._estimate_remaining_time(
                                start_time, batch_job.processed_files + batch_job.failed_files, len(files)
                            )
                        })
                        return None
            
            # 모든 파일을 병렬로 처리
            tasks = [
                process_single_file(filename, file_path, file_size, i)
                for i, (filename, file_path, file_size) in enumerate(files)
            ]
            
            # asyncio.gather로 병렬 실행 및 결과 수집
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리 (예외 처리 포함)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ 파일 처리 중 예외 발생: {files[i][0]} - {result}")
                    batch_job.failed_files += 1
            
            # 결과 파일 생성 진행 상황
            await progress_notifier.update_progress(batch_id, {
                "status": "creating_package",
                "progress": 0.95,
                "total_files": batch_job.total_files,
                "processed_files": batch_job.processed_files,
                "failed_files": batch_job.failed_files,
                "current_file": "결과 패키지 생성 중...",
                "total_duration": batch_job.total_duration,
                "total_processing_time": batch_job.total_processing_time
            })
            
            # 결과 파일 생성
            result_zip_path = await self._create_result_package(batch_job)
            batch_job.download_path = result_zip_path
            batch_job.status = "completed"
            batch_job.completed_at = datetime.now()
            
            # 완료 상황 전송
            await progress_notifier.update_progress(batch_id, {
                "status": "completed",
                "progress": 1.0,
                "total_files": batch_job.total_files,
                "processed_files": batch_job.processed_files,
                "failed_files": batch_job.failed_files,
                "total_duration": batch_job.total_duration,
                "total_processing_time": batch_job.total_processing_time,
                "completed_at": batch_job.completed_at.isoformat(),
                "download_available": True
            })
            
            logger.info(f"✅ 배치 {batch_id} 처리 완료")
            
        except Exception as e:
            batch_job.status = "failed"
            batch_job.error_message = str(e)
            batch_job.completed_at = datetime.now()
            
            # 실패 상황 전송
            await progress_notifier.update_progress(batch_id, {
                "status": "failed",
                "progress": batch_job.processed_files / batch_job.total_files if batch_job.total_files > 0 else 0.0,
                "total_files": batch_job.total_files,
                "processed_files": batch_job.processed_files,
                "failed_files": batch_job.failed_files,
                "error_message": str(e),
                "completed_at": batch_job.completed_at.isoformat()
            })
            
            logger.error(f"❌ 배치 {batch_id} 처리 실패: {e}")
        
        finally:
            self.current_processing -= 1
            # 배치 완료 후 연결 정리 (비동기로)
            asyncio.create_task(progress_notifier.cleanup_batch(batch_id))
    
    def _estimate_remaining_time(self, start_time: float, completed: int, total: int) -> Optional[float]:
        """남은 시간 추정"""
        if completed == 0:
            return None
        
        elapsed_time = time.time() - start_time
        avg_time_per_file = elapsed_time / completed
        remaining_files = total - completed
        
        return avg_time_per_file * remaining_files
    
    async def _process_audio_file(self, filename: str, file_path: str, file_size: int, 
                                request: BatchTranscriptionRequest) -> BatchFileInfo:
        """개별 오디오 파일 처리"""
        start_time = time.time()
        
        # 오디오 파일을 읽어서 STT 처리
        with open(file_path, "rb") as f:
            audio_content = f.read()
        
        # 오디오 길이 계산
        try:
            # librosa 또는 torchaudio로 오디오 길이 계산
            try:
                import librosa
                audio_data, sr = librosa.load(file_path, sr=16000)
                duration = len(audio_data) / sr
            except ImportError:
                # librosa가 없으면 torchaudio 사용
                import torchaudio
                waveform, sample_rate = torchaudio.load(file_path)
                duration = waveform.shape[1] / sample_rate
        except Exception:
            # 대략적인 추정 (평균 비트레이트 기준)
            duration = file_size / (16000 * 2)  # 16kHz, 16bit 기준
        
        # STT 서비스를 통한 전사
        # 파일을 base64로 인코딩하여 기존 STT 서비스 사용
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        
        # 파일 확장자에 따른 오디오 포맷 결정
        file_ext = os.path.splitext(filename)[1].lower()
        audio_format = "wav"  # 기본값
        if file_ext in ['.mp3', '.m4a']:
            audio_format = "mp3"
        elif file_ext in ['.flac']:
            audio_format = "flac"
        elif file_ext in ['.ogg', '.webm']:
            audio_format = "ogg"
        
        # STT 처리 요청 생성
        transcription_request = TranscriptionRequest(
            audio_data=audio_base64,
            language=request.language,
            audio_format=audio_format
        )
        
        # 실제 STT 처리 (기존 서비스 재사용)
        result = await self._transcribe_for_batch(transcription_request, request)
        
        processing_time = time.time() - start_time
        
        # 파일 정보 생성
        file_info = BatchFileInfo(
            filename=filename,
            size_bytes=file_size,
            duration_seconds=duration,
            processing_time_seconds=processing_time,
            text=result.get("text", ""),
            language=result.get("language", request.language),
            confidence=result.get("confidence", 0.0),
            segments=result.get("segments", [])
        )
        
        return file_info
    
    async def _transcribe_for_batch(self, request: TranscriptionRequest, batch_request: Optional[BatchTranscriptionRequest] = None) -> Dict[str, Any]:
        """배치 처리용 STT (환각 필터 및 키워드 부스팅 적용)"""
        if not stt_service:
            raise HTTPException(status_code=503, detail="STT 서비스가 초기화되지 않았습니다")
        
        # 환각 필터 인스턴스 생성
        hallucination_filter = HallucinationFilter()
        
        # 키워드 부스팅 관련 변수
        boosting_applied = False
        boosted_text = None
        keyword_stats = {}
        temp_call_id = None
        
        try:
            # base64 디코딩
            audio_bytes = base64.b64decode(request.audio_data)
            
            # numpy 배열로 변환
            try:
                # librosa로 다양한 오디오 포맷 처리
                import librosa
                audio_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            except ImportError:
                # librosa가 없으면 기본 처리
                if request.audio_format == "wav":
                    # WAV 파일은 단순 PCM으로 가정
                    audio_np = np.frombuffer(audio_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    # PCM 16kHz로 가정
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # STT 처리 - 서비스 종류에 따라 다른 방식 사용
            start_time = time.time()
            
            # NeMo 서비스인지 확인
            from src.api.nemo_stt_service import NeMoSTTService
            if isinstance(stt_service, NeMoSTTService):
                # NeMo 서비스 사용
                logger.info("🤖 NeMo 서비스로 배치 전사 처리 중...")
                logger.info(f"🔍 호출 파라미터 - audio_format: {request.audio_format}, language: {request.language}")
                logger.info(f"🔍 audio_data 길이: {len(request.audio_data)} chars")
                print(f"🚨 NeMo transcribe_audio 호출 직전! audio_format={request.audio_format}")
                
                try:
                    result = await stt_service.transcribe_audio(
                        request.audio_data,
                        audio_format=request.audio_format,
                        language=request.language
                    )
                    print("🚨 NeMo transcribe_audio 호출 성공!")
                    logger.info("✅ NeMo transcribe_audio 호출 성공!")
                except Exception as e:
                    print(f"🚨 NeMo transcribe_audio 호출 실패: {e}")
                    logger.error(f"❌ NeMo transcribe_audio 호출 실패: {e}")
                    raise
                
                # NeMo 결과를 Whisper 형식으로 변환
                full_text = result.text
                processing_time = time.time() - start_time
                audio_duration = len(audio_np) / 16000
                
                # 기본 세그먼트 생성 (NeMo는 segments를 따로 제공하지 않을 수 있음)
                segments_list = [{
                    "text": full_text,
                    "start": 0.0,
                    "end": audio_duration,
                    "confidence": result.confidence if hasattr(result, 'confidence') else 0.8
                }]
                
                # Whisper 스타일 info 객체 모방
                class NeMoInfo:
                    def __init__(self, language):
                        self.language = language
                
                info = NeMoInfo(request.language)
                
            else:
                # Whisper 서비스 사용
                logger.info("🎤 Whisper 서비스로 배치 전사 처리 중...")
                
                # 안전한 모델 접근
                model = None
                if hasattr(stt_service, 'model') and hasattr(stt_service.model, 'transcribe'):
                    model = stt_service.model
                elif hasattr(stt_service, 'whisper_service') and hasattr(stt_service.whisper_service, 'model'):
                    model = stt_service.whisper_service.model
                elif hasattr(stt_service, 'service') and hasattr(stt_service.service, 'model'):
                    model = stt_service.service.model
                
                if model and hasattr(model, 'transcribe'):
                    segments, info = model.transcribe(
                        audio_np,
                        language=request.language,
                        word_timestamps=True,
                        beam_size=5,
                        best_of=5,
                        temperature=0.0
                    )
                    segments_list = list(segments)
                else:
                    # 모델 직접 접근이 불가능한 경우 서비스 메서드 사용
                    logger.info("🔄 모델 직접 접근 불가, 서비스 메서드 사용")
                    # STT 서비스의 transcribe 메서드 사용
                    audio_bytes = (audio_np * 32768).astype(np.int16).tobytes()
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                    
                    result = await stt_service.transcribe_audio(
                        audio_data=audio_b64,
                        audio_format="pcm_16khz",
                        language=request.language
                    )
                    
                    # 결과를 segments 형태로 변환
                    segments_list = []
                    if hasattr(result, 'segments') and result.segments:
                        segments_list = result.segments
                    else:
                        # 단일 세그먼트로 처리
                        class MockSegment:
                            def __init__(self, text, start, end):
                                self.text = text
                                self.start = start
                                self.end = end
                                self.avg_logprob = -0.5
                                self.words = []
                        
                        segments_list = [MockSegment(result.text, 0.0, len(audio_np) / 16000.0)]
                    
                    # info 객체 생성
                    class MockInfo:
                        def __init__(self, language):
                            self.language = language
                    
                    info = MockInfo(request.language)
            
            # 신뢰도 계산 및 환각 필터 적용
            avg_confidence = 0.0
            segment_infos = []
            filtered_text_parts = []
            total_hallucination_count = 0
            
            for i, segment in enumerate(segments_list):
                # 세그먼트가 딕셔너리인지 객체인지 확인
                if isinstance(segment, dict):
                    # NeMo 결과 (딕셔너리)
                    segment_text = segment.get("text", "")
                    segment_start = segment.get("start", 0.0)
                    segment_end = segment.get("end", audio_duration)
                    segment_confidence = segment.get("confidence", 0.8)
                    words = segment.get("words", [])
                else:
                    # Whisper 결과 (객체)
                    segment_text = segment.text
                    segment_start = segment.start
                    segment_end = segment.end
                    
                    # 단어 정보 수집
                    words = []
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            # word.probability를 그대로 사용 (0-1 범위)
                            word_confidence = word.probability if hasattr(word, 'probability') and word.probability else 0.0
                            word_dict = {
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "confidence": word_confidence
                            }
                            words.append(word_dict)
                    
                    # 개선된 신뢰도 계산: word-level 신뢰도를 우선 사용
                    segment_confidence = calculate_segment_confidence_from_words(
                        words, 
                        segment.avg_logprob if hasattr(segment, 'avg_logprob') else None
                    )
                
                # SegmentInfo를 딕셔너리로 생성 (JSON 직렬화 문제 방지)
                segment_dict = {
                    "id": i,
                    "text": segment_text,
                    "start": segment_start,
                    "end": segment_end,
                    "confidence": segment_confidence,
                    "words": words
                }
                
                # 환각 감지
                duration = segment_dict['end'] - segment_dict['start']
                hallucination_info = hallucination_filter.detect_hallucination(segment_dict['text'], duration)
                
                if hallucination_info['is_hallucination']:
                    total_hallucination_count += 1
                    repeated_word = hallucination_info['repeated_word']
                    repeat_count = hallucination_info['repeat_count']
                    repetition_ratio = hallucination_info['repetition_ratio']
                    logger.warning(f"환각 감지: {segment_dict['start']:.1f}s-{segment_dict['end']:.1f}s, 단어: '{repeated_word}' ({repeat_count}회, {repetition_ratio:.1%})")
                
                # 환각 필터 적용
                filtered_segment = hallucination_filter.filter_segment(segment_dict)
                
                # 필터링된 텍스트만 전체 텍스트에 포함
                if filtered_segment['text'].strip():
                    filtered_text_parts.append(filtered_segment['text'])
                
                # confidence 값이 None이면 기본값 사용
                segment_confidence_value = filtered_segment.get('confidence')
                if segment_confidence_value is None:
                    segment_confidence_value = 0.8  # 기본 신뢰도
                
                avg_confidence += segment_confidence_value
                segment_infos.append(filtered_segment)
            
            if segments_list:
                avg_confidence /= len(segments_list)
            
            # 필터링된 전체 텍스트 생성 (NeMo에서 이미 설정되지 않은 경우만)
            if 'full_text' not in locals():
                full_text = " ".join(filtered_text_parts)
            
            # 키워드 부스팅 후처리 시도 (배치 요청에서 활성화된 경우)
            if batch_request and batch_request.enable_keyword_boosting and post_processing_corrector:
                logger.info(f"🎯 배치 키워드 교정 시도 중... call_id: {batch_request.call_id}")
                
                try:
                    # 키워드 준비
                    keywords_list = []
                    
                    # 1. call_id가 있는 경우 등록된 키워드 사용
                    if batch_request.call_id:
                        try:
                            active_keywords = await post_processing_corrector.get_keywords(batch_request.call_id)
                            if active_keywords:
                                keywords_list = list(active_keywords.keys())
                                logger.info(f"📋 call_id {batch_request.call_id}에서 {len(keywords_list)}개 키워드 로드됨")
                        except Exception as e:
                            logger.warning(f"⚠️ call_id {batch_request.call_id} 키워드 로드 실패: {e}")
                    
                    # 2. 직접 키워드 리스트가 제공된 경우
                    elif batch_request.keywords:
                        keywords_list = [kw.strip() for kw in batch_request.keywords if kw.strip()]
                        logger.info(f"📝 직접 제공된 {len(keywords_list)}개 키워드 사용")
                        
                        # 임시 call_id로 키워드 등록
                        temp_call_id = f"batch_temp_{uuid.uuid4().hex[:8]}"
                        await post_processing_corrector.register_keywords(temp_call_id, keywords_list)
                        logger.info(f"🔄 임시 call_id {temp_call_id}에 키워드 등록 완료")
                    
                    # 키워드 교정 적용
                    if keywords_list and full_text:
                        call_id_to_use = batch_request.call_id or temp_call_id
                        correction_result = await post_processing_corrector.apply_correction(
                            call_id=call_id_to_use,
                            text=full_text,
                            enable_fuzzy_matching=True,
                            min_similarity=0.8
                        )
                        
                        if correction_result and correction_result.corrected_text != full_text:
                            full_text = correction_result.corrected_text
                            boosting_applied = True
                            keyword_stats = {
                                'registered_keywords': len(keywords_list),
                                'keyword_list': keywords_list,
                                'boosting_applied': True,
                                'boost_factor': batch_request.keyword_boost_factor,
                                'original_text': full_text,
                                'corrected_text': correction_result.corrected_text,
                                'corrections': correction_result.corrections,
                                'detected_keywords': correction_result.keywords_detected
                            }
                            logger.info(f"✅ 배치 키워드 교정 적용 완료")
                            
                            # 키워드 매칭 확인
                            if correction_result.keywords_detected:
                                logger.info(f"🎯 감지된 키워드: {correction_result.keywords_detected}")
                        else:
                            logger.info(f"ℹ️ 키워드 교정 불필요")
                            
                except Exception as e:
                    logger.warning(f"⚠️ 배치 키워드 교정 실패: {e}")
                
                # 임시 키워드 정리
                if temp_call_id:
                    try:
                        await post_processing_corrector.delete_keywords(temp_call_id)
                        logger.info(f"🧹 임시 키워드 정리 완료: {temp_call_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ 임시 키워드 정리 실패: {e}")
            
            if total_hallucination_count > 0:
                logger.info(f"🔍 총 {total_hallucination_count}개 환각 구간 감지 및 처리됨")
            
            # 처리 시간과 오디오 길이 (NeMo에서 이미 설정되지 않은 경우만)
            if 'processing_time' not in locals():
                processing_time = time.time() - start_time
            if 'audio_duration' not in locals():
                audio_duration = len(audio_np) / 16000
            
            response_data = {
                "text": full_text,
                "language": info.language,
                "confidence": avg_confidence,
                "processing_time": processing_time,
                "audio_duration": audio_duration,
                "segments": segment_infos,
                "gpu_optimized": True,
                "boosting_applied": boosting_applied
            }
            
            if boosting_applied and keyword_stats:
                response_data['keyword_stats'] = keyword_stats
                
            return response_data
            
        except Exception as e:
            logger.error(f"❌ 배치 전사 실패: {e}")
            raise HTTPException(status_code=500, detail=f"STT 처리 중 오류가 발생했습니다: {str(e)}")
    
    def _create_timestamped_text(self, file_info: BatchFileInfo) -> str:
        """타임스탬프가 포함된 텍스트 생성"""
        try:
            logger.info(f"타임스탬프 텍스트 생성 시작 - 파일: {file_info.filename}")
            logger.info(f"segments 정보: {len(file_info.segments) if file_info.segments else 0}개")
            logger.info(f"segments 타입: {type(file_info.segments)}")
            
            if not file_info.segments:
                logger.warning(f"segments 정보가 없습니다 - 파일: {file_info.filename}")
                # segments가 없으면 기본 텍스트만 반환
                return f"파일명: {file_info.filename}\n전체 길이: {file_info.duration_seconds:.2f}초\n처리 시간: {file_info.processing_time_seconds:.2f}초\n\n전체 텍스트:\n{file_info.text}"
            
            timestamped_lines = []
            timestamped_lines.append(f"파일명: {file_info.filename}")
            timestamped_lines.append(f"전체 길이: {file_info.duration_seconds:.2f}초")
            timestamped_lines.append(f"처리 시간: {file_info.processing_time_seconds:.2f}초")
            timestamped_lines.append("=" * 60)
            timestamped_lines.append("")
            
            logger.info(f"처리할 segments 수: {len(file_info.segments)}")
            
            for i, segment in enumerate(file_info.segments):
                # segment가 dict인지 확인
                if isinstance(segment, dict):
                    start_time = segment.get('start', 0.0)
                    end_time = segment.get('end', 0.0)
                    text = segment.get('text', '').strip()
                    confidence = segment.get('confidence', 0.0)
                else:
                    # Pydantic 모델인 경우
                    start_time = getattr(segment, 'start', 0.0)
                    end_time = getattr(segment, 'end', 0.0)
                    text = getattr(segment, 'text', '').strip()
                    confidence = getattr(segment, 'confidence', 0.0)
                
                # 디버깅용 로그 (처음 몇 개만)
                if i < 3:
                    logger.info(f"Segment {i}: start={start_time}, end={end_time}, confidence={confidence}, text='{text[:50]}...'")
                
                # 시간을 MM:SS.sss 형식으로 변환
                start_minutes = int(start_time // 60)
                start_seconds = start_time % 60
                end_minutes = int(end_time // 60)
                end_seconds = end_time % 60
                
                timestamp_str = f"[{start_minutes:02d}:{start_seconds:06.3f} → {end_minutes:02d}:{end_seconds:06.3f}]"
                
                # 신뢰도 표시 (실제 값으로 표시, None이거나 0이면 표시하지 않음)
                if confidence is not None and confidence > 0:
                    confidence_str = f"(신뢰도: {confidence:.3f})"
                else:
                    confidence_str = ""
                
                timestamped_lines.append(f"{timestamp_str} {text} {confidence_str}")
            
            timestamped_lines.append("")
            timestamped_lines.append("=" * 60)
            timestamped_lines.append("전체 텍스트:")
            timestamped_lines.append(file_info.text)
            
            result = "\n".join(timestamped_lines)
            logger.info(f"타임스탬프 텍스트 생성 완료 - 파일: {file_info.filename}")
            return result
            
        except Exception as e:
            logger.error(f"타임스탬프 텍스트 생성 실패: {e}", exc_info=True)
            # 오류 발생 시 기본 텍스트 반환
            return f"타임스탬프 생성 실패 (오류: {str(e)})\n\n전체 텍스트:\n{file_info.text}"

    async def _create_result_package(self, batch_job: BatchJob) -> str:
        """결과 패키지 생성 (ZIP 파일)"""
        zip_filename = f"batch_stt_results_{batch_job.batch_id}.zip"
        zip_path = os.path.join(batch_job.temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 결과 JSON 파일 생성
            results_json = {
                "batch_id": batch_job.batch_id,
                "total_files": batch_job.total_files,
                "processed_files": batch_job.processed_files,
                "failed_files": batch_job.failed_files,
                "total_duration": batch_job.total_duration,
                "total_processing_time": batch_job.total_processing_time,
                "created_at": batch_job.created_at.isoformat(),
                "completed_at": batch_job.completed_at.isoformat() if batch_job.completed_at else None,
                "files": [self._file_info_to_dict(file_info) for file_info in batch_job.result_files]
            }
            
            # JSON 파일을 ZIP에 추가
            json_content = json.dumps(results_json, ensure_ascii=False, indent=2)
            zipf.writestr("batch_results.json", json_content)
            
            # 각 파일별 텍스트 결과를 별도 파일로 저장 (타임스탬프 포함)
            for file_info in batch_job.result_files:
                base_filename = os.path.splitext(file_info.filename)[0]
                
                # 타임스탬프가 포함된 상세 텍스트 파일
                timestamped_text = self._create_timestamped_text(file_info)
                zipf.writestr(f"transcripts/{base_filename}.txt", timestamped_text)
                
                # 기본 텍스트만 포함된 파일 (기존 호환성 유지)
                zipf.writestr(f"transcripts/{base_filename}_plain.txt", file_info.text)
        
        return zip_path
    
    def _file_info_to_dict(self, file_info: BatchFileInfo) -> dict:
        """BatchFileInfo를 딕셔너리로 변환"""
        try:
            # 안전한 딕셔너리 변환
            result = {
                "filename": getattr(file_info, 'filename', 'unknown'),
                "size_bytes": getattr(file_info, 'size_bytes', 0),
                "duration_seconds": getattr(file_info, 'duration_seconds', 0.0),
                "processing_time_seconds": getattr(file_info, 'processing_time_seconds', 0.0),
                "text": getattr(file_info, 'text', ''),
                "language": getattr(file_info, 'language', 'ko'),
                "confidence": getattr(file_info, 'confidence', 0.0),
                "segments": []
            }
            
            # segments 안전한 변환
            segments = getattr(file_info, 'segments', None)
            if segments:
                safe_segments = []
                for segment in segments:
                    if isinstance(segment, dict):
                        # 이미 딕셔너리인 경우
                        safe_segments.append(segment)
                    else:
                        # 객체인 경우 딕셔너리로 변환
                        segment_dict = {
                            "id": getattr(segment, 'id', 0),
                            "text": getattr(segment, 'text', ''),
                            "start": getattr(segment, 'start', 0.0),
                            "end": getattr(segment, 'end', 0.0),
                            "confidence": getattr(segment, 'confidence', 0.0),
                            "words": []
                        }
                        
                        # words 안전한 변환
                        words = getattr(segment, 'words', None)
                        if words:
                            safe_words = []
                            for word in words:
                                if isinstance(word, dict):
                                    safe_words.append(word)
                                else:
                                    word_dict = {
                                        "word": getattr(word, 'word', ''),
                                        "start": getattr(word, 'start', 0.0),
                                        "end": getattr(word, 'end', 0.0),
                                        "confidence": getattr(word, 'confidence', 0.0)
                                    }
                                    safe_words.append(word_dict)
                            segment_dict["words"] = safe_words
                        
                        safe_segments.append(segment_dict)
                
                result["segments"] = safe_segments
            
            return result
            
        except Exception as e:
            logger.error(f"파일 정보 변환 실패: {e}")
            # 최소한의 기본값 반환
            return {
                "filename": str(file_info) if file_info else 'unknown',
                "size_bytes": 0,
                "duration_seconds": 0.0,
                "processing_time_seconds": 0.0,
                "text": '',
                "language": 'ko',
                "confidence": 0.0,
                "segments": []
            }
    
    def get_batch_status(self, batch_id: str) -> Optional[BatchJob]:
        """배치 상태 조회"""
        return self.batch_jobs.get(batch_id)
    
    def get_batch_progress(self, batch_id: str) -> float:
        """배치 진행률 조회"""
        batch_job = self.batch_jobs.get(batch_id)
        if not batch_job or batch_job.total_files == 0:
            return 0.0
        return batch_job.processed_files / batch_job.total_files
    
    async def cancel_batch(self, batch_id: str) -> bool:
        """배치 취소"""
        batch_job = self.batch_jobs.get(batch_id)
        if not batch_job:
            return False
        
        if batch_job.status == "processing":
            batch_job.status = "cancelled"
            batch_job.completed_at = datetime.now()
            return True
        
        return False
    
    async def cleanup_old_batches(self, max_age_hours: int = 24):
        """오래된 배치 정리"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for batch_id, batch_job in self.batch_jobs.items():
            if batch_job.completed_at and batch_job.completed_at < cutoff_time:
                # 임시 디렉토리 삭제
                if batch_job.temp_dir and os.path.exists(batch_job.temp_dir):
                    shutil.rmtree(batch_job.temp_dir, ignore_errors=True)
                to_remove.append(batch_id)
        
        for batch_id in to_remove:
            del self.batch_jobs[batch_id]
        
        logger.info(f"정리된 배치 수: {len(to_remove)}")

# 전역 서비스 인스턴스
stt_service: Optional[BaseSTTService] = None
post_processing_corrector = None
batch_processor: Optional[BatchProcessor] = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 선택된 모델로 STT 서비스 초기화"""
    global stt_service, post_processing_corrector, stt_queue, batch_processor
    try:
        logger.info(f"🚀 {SERVER_MODEL_TYPE.upper()} {SERVER_MODEL_NAME} 모델로 STT Server 시작 중...")
        logger.info(f"cuDNN 활성화 상태: {torch.backends.cudnn.enabled}")
        logger.info(f"cuDNN 벤치마크 모드: {torch.backends.cudnn.benchmark}")
        logger.info(f"TF32 활성화 상태: {torch.backends.cuda.matmul.allow_tf32}")
        
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            logger.info(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"🎯 GPU 메모리: {gpu_props.total_memory / 1024**3:.1f}GB")
            logger.info(f"🎯 CUDA 버전: {torch.version.cuda}")
            logger.info(f"🎯 PyTorch 버전: {torch.__version__}")
            logger.info(f"🎯 GPU 아키텍처: {gpu_props.major}.{gpu_props.minor}")
            logger.info(f"🎯 멀티프로세서 수: {gpu_props.multi_processor_count}")
        
        # 선택된 모델로 STT 서비스 생성
        logger.info(f"📦 {SERVER_MODEL_TYPE.upper()} {SERVER_MODEL_NAME} 모델 로딩 중...")
        stt_service = STTServiceFactory.create_service(
            model_type=SERVER_MODEL_TYPE,
            model_name=SERVER_MODEL_NAME,
            device="cuda",
            compute_type="float16"
        )
        
        # 모델을 미리 로드하여 첫 번째 요청 지연 제거
        start_time = time.time()
        success = await stt_service.initialize()
        load_time = time.time() - start_time
        
        if success:
            logger.info(f"✅ {SERVER_MODEL_TYPE.upper()} STT 서비스 초기화 완료 - 모델 로딩 시간: {load_time:.2f}초")
            
            # Whisper 모델의 경우 웜업 수행
            if SERVER_MODEL_TYPE == "whisper" and hasattr(stt_service, 'whisper_service'):
                await warmup_large_model(stt_service.whisper_service)
        else:
            logger.error(f"❌ {SERVER_MODEL_TYPE.upper()} STT 서비스 초기화 실패")
            raise RuntimeError(f"STT 서비스 초기화 실패: {stt_service.initialization_error}")
        
        # GPU 메모리 상태 출력
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"🎯 GPU 메모리 사용량 - 할당: {gpu_memory_allocated:.2f}GB, 예약: {gpu_memory_reserved:.2f}GB")
        
        # 후처리 키워드 교정 시스템 초기화
        logger.info("🔧 후처리 키워드 교정 시스템 초기화 중...")
        try:
            post_processing_corrector = get_post_processing_corrector()
            logger.info("✅ 후처리 키워드 교정 시스템 초기화 완료")
            logger.info("   - 정확 매칭 교정 활성화")
            logger.info("   - 퍼지 매칭 교정 활성화")
            logger.info("   - 한국어 특화 교정 활성화")
        except Exception as e:
            logger.error(f"❌ 후처리 교정 시스템 초기화 실패: {e}")
            post_processing_corrector = None
        
        # 🔄 지능형 큐잉 시스템 초기화
        logger.info("🔄 지능형 STT 큐잉 시스템 초기화 중...")
        try:
            stt_queue = IntelligentSTTQueue(
                max_concurrent=8,  # 동시 처리 최대 8개 (테스트 결과 기반)
                max_queue_size=50,  # 최대 큐 크기 50개
                default_timeout=60,  # 기본 타임아웃 60초
                priority_timeout=30  # 우선순위 요청 타임아웃 30초
            )
            
            await stt_queue.start()
            
            # 백그라운드 요청 처리 시작
            asyncio.create_task(stt_queue.process_requests(stt_service))
            
            logger.info("✅ 지능형 큐잉 시스템 초기화 완료")
            logger.info(f"   최대 동시 처리: {stt_queue.max_concurrent}개")
            logger.info(f"   최대 큐 크기: {stt_queue.max_queue_size}개")
            
        except Exception as e:
            logger.error(f"❌ 큐잉 시스템 초기화 실패: {e}")
            stt_queue = None
        
        # 🗂️ 배치 프로세서 초기화
        logger.info("🗂️ 배치 프로세서 초기화 중...")
        try:
            batch_processor = BatchProcessor()
            logger.info("✅ 배치 프로세서 초기화 완료")
            logger.info(f"   배치 임시 디렉토리: {batch_processor.batch_temp_dir}")
            logger.info(f"   최대 동시 배치 처리: {batch_processor.max_concurrent_batches}개")
        except Exception as e:
            logger.error(f"❌ 배치 프로세서 초기화 실패: {e}")
            batch_processor = None
        
    except Exception as e:
        logger.error(f"❌ 서버 초기화 실패: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 큐잉 시스템 정리"""
    global stt_queue
    try:
        if stt_queue:
            logger.info("🛑 큐잉 시스템 종료 중...")
            await stt_queue.stop()
            logger.info("✅ 큐잉 시스템 종료 완료")
    except Exception as e:
        logger.error(f"❌ 큐잉 시스템 종료 중 오류: {e}")

@app.get("/")
async def root():
    """루트 엔드포인트 - 현재 로드된 모델 정보 포함"""
    gpu_features = {}
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_features = {
            "gpu_name": torch.cuda.get_device_name(0),
            "total_memory_gb": f"{gpu_props.total_memory / 1024**3:.1f}",
            "architecture": f"{gpu_props.major}.{gpu_props.minor}"
        }
    
    return {
        "message": f"{SERVER_MODEL_TYPE.upper()} {SERVER_MODEL_NAME} STT API Server", 
        "model_type": SERVER_MODEL_TYPE,
        "model_name": SERVER_MODEL_NAME,
        "optimization": "extreme" if SERVER_MODEL_TYPE == "whisper" else "optimized",
        "status": "running",
        "supported_models": STTServiceFactory.get_supported_models(),
        "features": {
            "model_type": SERVER_MODEL_TYPE,
            "model_name": SERVER_MODEL_NAME,
            "compute_type": "float16" if SERVER_MODEL_TYPE == "whisper" else "auto",
            "memory_fraction": 0.95 if SERVER_MODEL_TYPE == "whisper" else 0.8,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
            "gpu_available": torch.cuda.is_available(),
            "nemo_available": STTServiceFactory.is_nemo_available(),
            **gpu_features
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크 엔드포인트 (Large-v3 최적화 정보 포함)"""
    model_loaded = False
    gpu_info = {}
    optimization_status = {}
    
    if stt_service is not None:
        model_loaded = hasattr(stt_service, 'model') and stt_service.model is not None
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
        
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_gb": f"{gpu_props.total_memory / 1024**3:.1f}",
            "allocated_memory_gb": f"{gpu_memory_allocated:.2f}",
            "reserved_memory_gb": f"{gpu_memory_reserved:.2f}",
            "memory_utilization": f"{gpu_memory_allocated / (gpu_props.total_memory / 1024**3) * 100:.1f}%",
            "architecture": f"{gpu_props.major}.{gpu_props.minor}",
            "multiprocessor_count": gpu_props.multi_processor_count
        }
        
        optimization_status = {
            "model_type": "large-v3",
            "compute_type": "float16",
            "memory_fraction": "0.95",
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
            "flash_attention": "enabled"
        }
        
    return HealthResponse(
        status="healthy" if model_loaded else "loading",
        gpu_available=torch.cuda.is_available(),
        model_loaded=model_loaded,
        cudnn_enabled=torch.backends.cudnn.enabled,
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        gpu_info=gpu_info,
        optimization_status=optimization_status
    )

@app.get("/models/info")
async def get_model_info():
    """현재 로드된 모델의 상세 정보"""
    if stt_service is None:
        raise HTTPException(status_code=500, detail="STT 서비스가 초기화되지 않았습니다")
    
    return {
        "current_model": stt_service.get_model_info(),
        "supported_models": STTServiceFactory.get_supported_models(),
        "server_config": {
            "model_type": SERVER_MODEL_TYPE,
            "model_name": SERVER_MODEL_NAME,
            "nemo_available": STTServiceFactory.is_nemo_available(),
            "available_model_types": STTServiceFactory.get_available_model_types()
        },
        "stats": stt_service.get_stats()
    }

@app.post("/infer/utterance", response_model=TranscriptionResponse) 
async def transcribe_utterance(request: TranscriptionRequest):
    """
    신뢰도 정보 포함한 상세 전사 엔드포인트
    
    Large-v3 전용 극한 최적화와 함께 신뢰도 점수, 세그먼트, 단어 타임스탬프를 제공합니다.
    자동 키워드 부스팅이 적용됩니다.
    """
    if stt_service is None:
        raise HTTPException(status_code=500, detail="STT 서비스가 초기화되지 않았습니다")
    
    try:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        logger.info(f"🎯 Large-v3 신뢰도 전사 요청 {request_id} 시작 (언어: {request.language})")
        
        # 키워드 부스팅 체크 및 적용
        boosting_applied = False
        boosted_text = None
        keyword_stats = {}
        
        try:
            # 전체 키워드 통계 확인
            logger.info("🔍 키워드 부스팅 상태 확인 중...")
            global_stats = await keyword_service.get_global_statistics()
            total_keywords = global_stats.get('total_keywords', 0)
            active_keywords = global_stats.get('active_keywords', 0)
            
            logger.info(f"🔑 키워드 상태: 총 {total_keywords}개, 활성 {active_keywords}개")
            
            if active_keywords > 0:
                logger.info("🚀 활성 키워드 발견! 키워드 부스팅 전사 시도...")
                
                # 모든 활성 키워드 가져오기
                all_keywords = await keyword_service.get_all_active_keywords()
                if all_keywords:
                    keywords_list = list(all_keywords.keys())
                    logger.info(f"🎯 발견된 키워드: {keywords_list}")
                    
                    # 임시 call_id 생성
                    temp_call_id = f"auto_boost_{request_id}"
                    
                    # 키워드를 임시로 등록 (기존 키워드가 자동으로 적용되지 않을 수 있음)
                    try:
                        from src.api.keyword_models import KeywordEntry, KeywordRegistrationRequest
                        
                        temp_keywords = []
                        for keyword, details in all_keywords.items():
                            temp_keywords.append(KeywordEntry(
                                keyword=keyword,
                                boost_factor=details.get('boost_factor', 2.0),
                                category=details.get('category', 'custom'),
                                confidence_threshold=details.get('confidence_threshold', 0.3),
                                aliases=details.get('aliases', []),
                                enabled=True
                            ))
                        
                        temp_request = KeywordRegistrationRequest(
                            call_id=temp_call_id,
                            keywords=temp_keywords,
                            global_boost_factor=2.0,
                            replace_existing=True
                        )
                        
                        # 임시 키워드 등록
                        await keyword_service.register_keywords(temp_request)
                        logger.info(f"✅ 임시 키워드 등록 완료: {temp_call_id}")
                        
                        # 키워드 부스팅 전사 시도 (initial_prompt 사용)
                        logger.info("🎙️ 키워드 부스팅 상황에서 전사 실행...")
                        
                        # 키워드를 initial_prompt로 변환
                        keywords_prompt = ", ".join(keywords_list)
                        initial_prompt = f"다음 키워드들이 포함될 수 있습니다: {keywords_prompt}"
                        logger.info(f"🎯 Initial prompt: '{initial_prompt}'")
                        
                        # Base64 디코딩하여 NumPy 배열로 변환
                        try:
                            audio_bytes = base64.b64decode(request.audio_data)
                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            
                            # 안전한 모델 접근으로 initial_prompt 사용
                            model = None
                            if hasattr(stt_service, 'model') and hasattr(stt_service.model, 'transcribe'):
                                model = stt_service.model
                            elif hasattr(stt_service, 'whisper_service') and hasattr(stt_service.whisper_service, 'model'):
                                model = stt_service.whisper_service.model
                            elif hasattr(stt_service, 'service') and hasattr(stt_service.service, 'model'):
                                model = stt_service.service.model
                            
                            if model and hasattr(model, 'transcribe'):
                                segments, info = model.transcribe(
                                    audio_array,
                                    beam_size=5,
                                    best_of=5,
                                    temperature=0.0,
                                    vad_filter=False,
                                    language=request.language,
                                    word_timestamps=True,
                                    initial_prompt=initial_prompt  # 키워드 힌트 제공
                                )
                            else:
                                # 모델 직접 접근이 불가능한 경우
                                raise Exception("키워드 부스팅을 위한 모델 직접 접근이 불가능합니다")
                            
                            segments_list = list(segments)
                            if segments_list:
                                boosted_text = " ".join([segment.text.strip() for segment in segments_list if segment.text.strip()])
                                
                                # STTResult 형태로 변환
                                class MockResult:
                                    def __init__(self, text, language, segments, audio_duration):
                                        self.text = text
                                        self.language = language
                                        self.segments = segments
                                        self.audio_duration = len(audio_array) / 16000.0
                                        self.rtf = 0.05  # 임시값
                                
                                result = MockResult(boosted_text, request.language, segments_list, len(audio_array) / 16000.0)
                                logger.info(f"✅ 키워드 부스팅 전사 결과: '{boosted_text}'")
                            
                        except Exception as e:
                            logger.error(f"❌ 키워드 부스팅 전사 중 오류: {e}")
                            # fallback to normal transcription
                            result = await stt_service.transcribe_audio(
                                audio_data=request.audio_data,
                                audio_format=request.audio_format,
                                language=request.language
                            )
                            boosting_applied = False
                        
                        if result and result.text:
                            boosted_text = result.text
                            boosting_applied = True
                            keyword_stats = {
                                'registered_keywords': len(all_keywords),
                                'keyword_list': keywords_list,
                                'boosting_applied': True
                            }
                            logger.info(f"✅ 키워드 부스팅 전사 성공: '{boosted_text}'")
                            
                            # 키워드 매칭 확인
                            matches = []
                            for keyword in keywords_list:
                                if keyword in boosted_text:
                                    matches.append(keyword)
                            
                            if matches:
                                keyword_stats['detected_keywords'] = matches
                                logger.info(f"🎯 감지된 키워드: {matches}")
                            
                        # 임시 키워드 정리
                        await keyword_service.delete_keywords(temp_call_id)
                        logger.info(f"🧹 임시 키워드 정리 완료: {temp_call_id}")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ 키워드 부스팅 전사 실패: {e}")
                        # 임시 키워드 정리 시도
                        try:
                            await keyword_service.delete_keywords(temp_call_id)
                        except:
                            pass
                else:
                    logger.info("ℹ️ 활성 키워드가 실제로는 비어있음")
            else:
                logger.info("ℹ️ 활성 키워드 없음, 일반 전사 진행")
                
        except Exception as e:
            logger.warning(f"⚠️ 키워드 부스팅 확인 실패: {e}")
            boosting_applied = False
        
        # 키워드 부스팅이 성공하지 않은 경우 일반 전사 수행
        if not boosting_applied:
            logger.info("🎙️ 일반 FasterWhisper 전사 실행")
            # 오디오 전사 실행 (Large-v3 최적화 파라미터)
            result = await stt_service.transcribe_audio(
                audio_data=request.audio_data,
                audio_format=request.audio_format,
                language=request.language
            )
        else:
            # 키워드 부스팅된 결과 사용
            pass
        
        processing_time = time.time() - start_time
        
        # 응답 텍스트 결정
        final_text = boosted_text if boosting_applied else result.text
        
        # 기본 응답 생성 (세그먼트 정보 포함)
        response = TranscriptionResponse(
            text=final_text,
            language=result.language,
            rtf=result.rtf,
            processing_time=processing_time,
            audio_duration=result.audio_duration,
            gpu_optimized=True,
            model_load_time=None
        )
        
        # 세그먼트와 신뢰도 정보 추가 (가능한 경우)
        segments_list = []
        if hasattr(result, 'segments') and result.segments:
            for i, segment in enumerate(result.segments):
                segment_confidence = None
                words_list = []
                
                # 신뢰도 계산 (avg_logprob이 있는 경우)
                if hasattr(segment, 'avg_logprob') or 'avg_logprob' in segment:
                    avg_logprob = getattr(segment, 'avg_logprob', segment.get('avg_logprob', -1.0))
                    segment_confidence = logprob_to_confidence(avg_logprob)
                
                # 단어 레벨 정보 (있는 경우)
                if hasattr(segment, 'words') or 'words' in segment:
                    segment_words = getattr(segment, 'words', segment.get('words', []))
                    if segment_words:
                        for word in segment_words:
                            word_confidence = None
                            if hasattr(word, 'probability') or 'probability' in word:
                                prob = getattr(word, 'probability', word.get('probability'))
                                if prob is not None:
                                    word_confidence = normalize_word_probability(prob)
                            
                            words_list.append(WordSegment(
                                word=getattr(word, 'word', word.get('word', '')),
                                start=getattr(word, 'start', word.get('start', 0.0)),
                                end=getattr(word, 'end', word.get('end', 0.0)),
                                confidence=word_confidence
                            ))
                
                # 세그먼트 정보 추가
                segment_info = SegmentInfo(
                    id=i,
                    text=getattr(segment, 'text', segment.get('text', '')).strip(),
                    start=getattr(segment, 'start', segment.get('start', 0.0)),
                    end=getattr(segment, 'end', segment.get('end', result.audio_duration)),
                    confidence=segment_confidence,
                    words=words_list if words_list else None
                )
                
                segments_list.append(segment_info)
        
        response.segments = segments_list if segments_list else None
        
        # 로그 출력
        boost_status = "키워드부스팅적용" if boosting_applied else "일반전사"
        logger.info(f"✅ {boost_status} 완료 {request_id}: RTF={result.rtf:.3f}x, "
                   f"처리시간={processing_time:.3f}초, 텍스트='{final_text}'")
        
        if boosting_applied and keyword_stats:
            logger.info(f"🎯 키워드 부스팅 통계: {keyword_stats}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ 신뢰도 전사 실패: {e}")
        raise HTTPException(status_code=500, detail=f"전사 실패: {str(e)}")

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

# ============================================================================
# 큐잉 시스템 API 엔드포인트
# ============================================================================

@app.post("/queue/transcribe", response_model=QueuedTranscriptionResponse)
async def queue_transcribe_audio(request: QueuedTranscriptionRequest):
    """음성 인식 요청을 큐에 제출"""
    if stt_queue is None:
        raise HTTPException(status_code=503, detail="Queue system not initialized")
    
    try:
        # 우선순위 변환
        priority_map = {
            "high": RequestPriority.HIGH,
            "medium": RequestPriority.MEDIUM,
            "low": RequestPriority.LOW
        }
        priority = priority_map.get(request.priority.lower(), RequestPriority.MEDIUM)
        
        # 큐에 요청 제출
        request_id = await stt_queue.submit_request(
            audio_data=request.audio_data,
            language=request.language,
            audio_format=request.audio_format,
            client_id=request.client_id,
            priority=priority,
            timeout=request.timeout
        )
        
        # 큐 통계 가져오기
        stats = await stt_queue.get_queue_stats()
        estimated_wait_time = (stats.queued_requests / max(1, stats.current_throughput)) * 60 if stats.current_throughput > 0 else stats.queued_requests * 30
        
        return QueuedTranscriptionResponse(
            request_id=request_id,
            status="queued",
            message="Request successfully queued for processing",
            estimated_wait_time=estimated_wait_time,
            queue_position=stats.queued_requests
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 큐 요청 제출 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue request: {str(e)}")

@app.get("/queue/result/{request_id}")
async def get_queue_result(request_id: str):
    """큐 처리 결과 가져오기 (대기/완료까지)"""
    if stt_queue is None:
        raise HTTPException(status_code=503, detail="Queue system not initialized")
    
    try:
        result = await stt_queue.get_request_result(request_id)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 큐 결과 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get result: {str(e)}")

@app.get("/queue/status/{request_id}", response_model=QueueStatusResponse)
async def get_queue_status(request_id: str):
    """요청 상태 조회"""
    if stt_queue is None:
        raise HTTPException(status_code=503, detail="Queue system not initialized")
    
    try:
        status_info = await stt_queue.get_request_status(request_id)
        return QueueStatusResponse(**status_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 큐 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.delete("/queue/cancel/{request_id}")
async def cancel_queue_request(request_id: str):
    """큐 요청 취소"""
    if stt_queue is None:
        raise HTTPException(status_code=503, detail="Queue system not initialized")
    
    try:
        cancelled = await stt_queue.cancel_request(request_id)
        
        if cancelled:
            return {"message": f"Request {request_id} cancelled successfully", "cancelled": True}
        else:
            return {"message": f"Request {request_id} could not be cancelled (may be processing or completed)", "cancelled": False}
        
    except Exception as e:
        logger.error(f"❌ 큐 요청 취소 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel request: {str(e)}")

@app.get("/queue/stats", response_model=QueueStatsResponse)
async def get_queue_stats():
    """큐 통계 조회"""
    if stt_queue is None:
        raise HTTPException(status_code=503, detail="Queue system not initialized")
    
    try:
        stats = await stt_queue.get_queue_stats()
        
        return QueueStatsResponse(
            total_requests=stats.total_requests,
            queued_requests=stats.queued_requests,
            processing_requests=stats.processing_requests,
            completed_requests=stats.completed_requests,
            failed_requests=stats.failed_requests,
            timeout_requests=stats.timeout_requests,
            average_wait_time=stats.average_wait_time,
            average_processing_time=stats.average_processing_time,
            current_throughput=stats.current_throughput,
            queue_capacity=stt_queue.max_queue_size,
            max_concurrent=stt_queue.max_concurrent
        )
        
    except Exception as e:
        logger.error(f"❌ 큐 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue stats: {str(e)}")

@app.post("/queue/priority/transcribe", response_model=QueuedTranscriptionResponse)
async def priority_transcribe_audio(request: QueuedTranscriptionRequest):
    """우선순위 음성 인식 요청 (높은 우선순위로 큐에 제출)"""
    if stt_queue is None:
        raise HTTPException(status_code=503, detail="Queue system not initialized")
    
    # 요청을 높은 우선순위로 설정
    request.priority = "high"
    
    return await queue_transcribe_audio(request)


# 키워드 부스팅 API 엔드포인트들
@app.post("/keywords/register")
async def register_keywords(request: KeywordRegistrationRequest):
    """키워드 등록 API (후처리 교정용)"""
    if post_processing_corrector is None:
        raise HTTPException(status_code=500, detail="후처리 교정 시스템이 초기화되지 않았습니다")
    
    try:
        success = await post_processing_corrector.register_keywords(
            call_id=request.call_id,
            keywords=request.keywords
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="키워드 등록에 실패했습니다")
        
        return {
            "message": "키워드가 성공적으로 등록되었습니다",
            "call_id": request.call_id,
            "keyword_count": len(request.keywords),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"키워드 등록 실패: {e}")
        raise HTTPException(status_code=500, detail=f"키워드 등록 실패: {str(e)}")


@app.post("/keywords/correct", response_model=KeywordCorrectionResponse)
async def correct_keywords(request: KeywordCorrectionRequest):
    """키워드 교정 API"""
    if post_processing_corrector is None:
        raise HTTPException(status_code=500, detail="후처리 교정 시스템이 초기화되지 않았습니다")
    
    try:
        result = await post_processing_corrector.apply_correction(
            call_id=request.call_id,
            text=request.text,
            enable_fuzzy_matching=request.enable_fuzzy_matching,
            min_similarity=request.min_similarity
        )
        
        return KeywordCorrectionResponse(
            original_text=result.original_text,
            corrected_text=result.corrected_text,
            corrections=result.corrections,
            keywords_detected=result.keywords_detected,
            confidence_score=result.confidence_score,
            processing_time=result.processing_time
        )
        
    except Exception as e:
        logger.error(f"키워드 교정 실패: {e}")
        raise HTTPException(status_code=500, detail=f"키워드 교정 실패: {str(e)}")


@app.get("/keywords/{call_id}")
async def get_keywords(call_id: str):
    """키워드 조회 API"""
    if post_processing_corrector is None:
        raise HTTPException(status_code=500, detail="후처리 교정 시스템이 초기화되지 않았습니다")
    
    try:
        keywords = await post_processing_corrector.get_keywords(call_id)
        if keywords is None:
            raise HTTPException(status_code=404, detail=f"키워드를 찾을 수 없습니다: {call_id}")
        
        # KeywordEntry 객체를 딕셔너리로 변환
        keywords_dict = {}
        for keyword, entry in keywords.items():
            keywords_dict[keyword] = {
                "keyword": entry.keyword,
                "aliases": entry.aliases,
                "confidence_threshold": entry.confidence_threshold,
                "category": entry.category,
                "enabled": entry.enabled,
                "created_at": entry.created_at.isoformat()
            }
        
        return {
            "call_id": call_id,
            "keywords": keywords_dict,
            "keyword_count": len(keywords_dict)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"키워드 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"키워드 조회 실패: {str(e)}")


@app.delete("/keywords/{call_id}")
async def delete_keywords(call_id: str):
    """키워드 삭제 API"""
    if post_processing_corrector is None:
        raise HTTPException(status_code=500, detail="후처리 교정 시스템이 초기화되지 않았습니다")
    
    try:
        success = await post_processing_corrector.delete_keywords(call_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"키워드를 찾을 수 없습니다: {call_id}")
        
        return {"message": f"키워드가 삭제되었습니다: {call_id}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"키워드 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"키워드 삭제 실패: {str(e)}")


@app.get("/keywords/stats", response_model=KeywordStatsResponse)
async def get_keyword_statistics():
    """키워드 교정 통계 API"""
    if post_processing_corrector is None:
        raise HTTPException(status_code=500, detail="후처리 교정 시스템이 초기화되지 않았습니다")
    
    try:
        stats = post_processing_corrector.get_stats()
        
        return KeywordStatsResponse(
            total_keywords=len(post_processing_corrector.keyword_cache),
            total_corrections=stats.get('total_corrections', 0),
            successful_corrections=stats.get('successful_corrections', 0),
            success_rate=stats.get('success_rate', 0.0),
            avg_processing_time=stats.get('avg_processing_time', 0.0),
            categories={}  # 카테고리별 통계는 필요시 추가
        )
        
    except Exception as e:
        logger.error(f"키워드 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"키워드 통계 조회 실패: {str(e)}")


@app.post("/transcribe/with-correction/{call_id}", response_model=TranscriptionWithCorrection)
async def transcribe_with_keyword_correction(
    call_id: str,
    audio: UploadFile = File(...),
    language: str = Form("ko"),
    vad_filter: bool = Form(False),
    enable_fuzzy_matching: bool = Form(True),
    min_similarity: float = Form(0.8)
):
    """키워드 교정이 적용된 파일 전사 API"""
    if stt_service is None or post_processing_corrector is None:
        raise HTTPException(status_code=500, detail="STT 서비스 또는 교정 시스템이 초기화되지 않았습니다")
    
    try:
        start_time = time.time()
        
        # 파일 읽기
        audio_bytes = await audio.read()
        
        # 기본 전사 실행
        stt_result = await stt_service.transcribe_file_bytes(
            audio_bytes=audio_bytes,
            language=language,
            vad_filter=vad_filter
        )
        
        # 키워드 교정 적용
        correction_result = await post_processing_corrector.apply_correction(
            call_id=call_id,
            text=stt_result.text,
            enable_fuzzy_matching=enable_fuzzy_matching,
            min_similarity=min_similarity
        )
        
        processing_time = time.time() - start_time
        
        # 세그먼트 변환
        segments = []
        for i, segment in enumerate(stt_result.segments):
            segment_dict = segment if isinstance(segment, dict) else asdict(segment)
            segments.append(SegmentInfo(
                id=segment_dict.get('id', i),
                text=segment_dict.get('text', ''),
                start=segment_dict.get('start', 0.0),
                end=segment_dict.get('end', 0.0),
                confidence=segment_dict.get('confidence')
            ))
        
        return TranscriptionWithCorrection(
            text=stt_result.text,
            corrected_text=correction_result.corrected_text,
            language=stt_result.language,
            segments=segments,
            keyword_correction=KeywordCorrectionResponse(
                original_text=correction_result.original_text,
                corrected_text=correction_result.corrected_text,
                corrections=correction_result.corrections,
                keywords_detected=correction_result.keywords_detected,
                confidence_score=correction_result.confidence_score,
                processing_time=correction_result.processing_time
            ),
            metrics=ProcessingMetrics(
                total_duration=processing_time,
                audio_duration=stt_result.audio_duration,
                rtf=stt_result.rtf,
                inference_time=stt_result.processing_time
            )
        )
        
    except Exception as e:
        logger.error(f"키워드 교정 전사 실패: {e}")
        raise HTTPException(status_code=500, detail=f"키워드 교정 전사 실패: {str(e)}")


# ============================================================================
# 배치 처리 API 엔드포인트들
# ============================================================================

@app.post("/batch/transcribe")
async def batch_transcribe(
    language: str = Form("ko"),
    vad_filter: bool = Form(False),
    enable_word_timestamps: bool = Form(True),
    enable_confidence: bool = Form(True),
    client_id: Optional[str] = Form(None),
    priority: str = Form("medium"),
    # 키워드 부스팅 관련 매개변수
    call_id: Optional[str] = Form(None),  # 키워드가 등록된 call_id
    enable_keyword_boosting: bool = Form(False),  # 키워드 부스팅 활성화
    keywords: Optional[str] = Form(None),  # 쉼표로 구분된 키워드 리스트
    keyword_boost_factor: float = Form(2.0),  # 키워드 부스팅 강도
    files: List[UploadFile] = File(...)
):
    """
    배치 STT 처리 (키워드 부스팅 및 환각 필터 포함)
    
    - files: 업로드할 오디오 파일들
    - language: 언어 코드 (기본값: ko)
    - vad_filter: Voice Activity Detection 필터 사용 여부
    - enable_word_timestamps: 단어별 타임스탬프 생성 여부
    - enable_confidence: 신뢰도 정보 포함 여부
    - call_id: 등록된 키워드를 사용할 call_id
    - enable_keyword_boosting: 키워드 부스팅 활성화 여부
    - keywords: 직접 키워드 지정 (쉼표로 구분, 예: "안녕하세요,감사합니다")
    - keyword_boost_factor: 키워드 부스팅 강도 (1.0-5.0)
    """
    
    if not files:
        raise HTTPException(status_code=400, detail="최소 1개의 파일이 필요합니다")
    
    # 키워드 리스트 파싱
    keywords_list = None
    if keywords:
        keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
    
    # 배치 요청 객체 생성
    batch_request = BatchTranscriptionRequest(
        language=language,
        vad_filter=vad_filter,
        enable_word_timestamps=enable_word_timestamps,
        enable_confidence=enable_confidence,
        client_id=client_id,
        priority=priority,
        call_id=call_id,
        enable_keyword_boosting=enable_keyword_boosting,
        keywords=keywords_list,
        keyword_boost_factor=keyword_boost_factor
    )
    
    logger.info(f"🎯 배치 STT 요청 - 파일 수: {len(files)}, 키워드 부스팅: {enable_keyword_boosting}, call_id: {call_id}")
    if keywords_list:
        logger.info(f"📝 제공된 키워드: {keywords_list}")
    
    # 배치 작업 제출 (비동기적으로 처리됨)
    batch_id = await batch_processor.submit_batch(files, batch_request)
    
    # 즉시 200 응답과 함께 batch_id 반환
    return {
        "batch_id": batch_id,
        "status": "processing",
        "message": f"배치 처리가 시작되었습니다. {len(files)}개 파일 처리 중",
        "total_files": len(files),
        "processed_files": 0,
        "failed_files": 0,
        "created_at": datetime.now().isoformat(),
        "progress_url": f"/batch/progress/{batch_id}",
        "status_url": f"/batch/status/{batch_id}",
        "result_url": f"/batch/result/{batch_id}"
    }

@app.get("/batch/status/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str):
    """배치 처리 상태 조회"""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="배치 프로세서가 초기화되지 않았습니다")
    
    try:
        batch_job = batch_processor.get_batch_status(batch_id)
        if not batch_job:
            raise HTTPException(status_code=404, detail=f"배치를 찾을 수 없습니다: {batch_id}")
        
        progress = batch_processor.get_batch_progress(batch_id)
        
        # 남은 시간 추정
        estimated_remaining_time = None
        if batch_job.status == "processing" and progress > 0:
            elapsed_time = (datetime.now() - batch_job.created_at).total_seconds()
            total_estimated_time = elapsed_time / progress
            estimated_remaining_time = total_estimated_time - elapsed_time
        
        return BatchStatusResponse(
            batch_id=batch_id,
            status=batch_job.status,
            progress=progress,
            total_files=batch_job.total_files,
            processed_files=batch_job.processed_files,
            failed_files=batch_job.failed_files,
            estimated_remaining_time=estimated_remaining_time,
            created_at=batch_job.created_at.isoformat(),
            completed_at=batch_job.completed_at.isoformat() if batch_job.completed_at else None,
            error_message=batch_job.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"배치 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"배치 상태 조회 실패: {str(e)}")

@app.get("/batch/result/{batch_id}", response_model=BatchTranscriptionResponse)
async def get_batch_result(batch_id: str):
    """배치 처리 결과 조회"""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="배치 프로세서가 초기화되지 않았습니다")
    
    try:
        batch_job = batch_processor.get_batch_status(batch_id)
        if not batch_job:
            raise HTTPException(status_code=404, detail=f"배치를 찾을 수 없습니다: {batch_id}")
        
        # 다운로드 URL 생성 (배치가 완료된 경우)
        download_url = None
        if batch_job.status == "completed" and batch_job.download_path:
            download_url = f"/batch/download/{batch_id}"
        
        return BatchTranscriptionResponse(
            batch_id=batch_id,
            status=batch_job.status,
            message="배치 처리 결과" if batch_job.status == "completed" else f"배치 상태: {batch_job.status}",
            total_files=batch_job.total_files,
            processed_files=batch_job.processed_files,
            failed_files=batch_job.failed_files,
            total_duration=batch_job.total_duration,
            total_processing_time=batch_job.total_processing_time,
            created_at=batch_job.created_at.isoformat(),
            download_url=download_url,
            files=batch_job.result_files if batch_job.status == "completed" else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"배치 결과 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"배치 결과 조회 실패: {str(e)}")

@app.get("/batch/download/{batch_id}")
async def download_batch_result(batch_id: str):
    """배치 처리 결과 파일 다운로드"""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="배치 프로세서가 초기화되지 않았습니다")
    
    try:
        batch_job = batch_processor.get_batch_status(batch_id)
        if not batch_job:
            raise HTTPException(status_code=404, detail=f"배치를 찾을 수 없습니다: {batch_id}")
        
        if batch_job.status != "completed":
            raise HTTPException(status_code=400, detail=f"배치가 아직 완료되지 않았습니다. 현재 상태: {batch_job.status}")
        
        if not batch_job.download_path or not os.path.exists(batch_job.download_path):
            raise HTTPException(status_code=404, detail="결과 파일을 찾을 수 없습니다")
        
        # 파일 다운로드 응답
        filename = f"batch_stt_results_{batch_id}.zip"
        return FileResponse(
            path=batch_job.download_path,
            filename=filename,
            media_type="application/zip"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"배치 다운로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"배치 다운로드 실패: {str(e)}")

@app.delete("/batch/cancel/{batch_id}")
async def cancel_batch_processing(batch_id: str):
    """배치 처리 취소"""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="배치 프로세서가 초기화되지 않았습니다")
    
    try:
        success = await batch_processor.cancel_batch(batch_id)
        
        if success:
            return {"message": f"배치 {batch_id}가 취소되었습니다", "cancelled": True}
        else:
            return {"message": f"배치 {batch_id}를 취소할 수 없습니다 (이미 완료되었거나 존재하지 않음)", "cancelled": False}
        
    except Exception as e:
        logger.error(f"배치 취소 실패: {e}")
        raise HTTPException(status_code=500, detail=f"배치 취소 실패: {str(e)}")

@app.get("/batch/list")
async def list_batch_jobs():
    """모든 배치 작업 목록 조회"""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="배치 프로세서가 초기화되지 않았습니다")
    
    try:
        batch_list = []
        for batch_id, batch_job in batch_processor.batch_jobs.items():
            progress = batch_processor.get_batch_progress(batch_id)
            
            batch_info = {
                "batch_id": batch_id,
                "status": batch_job.status,
                "progress": progress,
                "total_files": batch_job.total_files,
                "processed_files": batch_job.processed_files,
                "failed_files": batch_job.failed_files,
                "created_at": batch_job.created_at.isoformat(),
                "completed_at": batch_job.completed_at.isoformat() if batch_job.completed_at else None
            }
            batch_list.append(batch_info)
        
        # 최신 순으로 정렬
        batch_list.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "total_batches": len(batch_list),
            "batches": batch_list
        }
        
    except Exception as e:
        logger.error(f"배치 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"배치 목록 조회 실패: {str(e)}")

@app.post("/batch/cleanup")
async def cleanup_old_batches(max_age_hours: int = 24):
    """오래된 배치 작업 정리"""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="배치 프로세서가 초기화되지 않았습니다")
    
    try:
        await batch_processor.cleanup_old_batches(max_age_hours)
        return {"message": f"{max_age_hours}시간 이상 된 배치 작업들이 정리되었습니다"}
        
    except Exception as e:
        logger.error(f"배치 정리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"배치 정리 실패: {str(e)}")


# ============================================================================
# 실시간 진행 상황 전달 시스템
# ============================================================================

class ProgressNotifier:
    """배치 처리 진행 상황을 실시간으로 클라이언트에 전달"""
    
    def __init__(self):
        # WebSocket 연결 관리
        self.websocket_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        # SSE 연결 관리  
        self.sse_connections: Dict[str, Set] = defaultdict(set)
        # 진행 상황 저장
        self.progress_data: Dict[str, Dict] = {}
        # 업데이트 간격 (초)
        self.update_interval = 5
        
    async def add_websocket_connection(self, batch_id: str, websocket: WebSocket):
        """WebSocket 연결 추가"""
        await websocket.accept()
        self.websocket_connections[batch_id].add(websocket)
        logger.info(f"📡 WebSocket 연결 추가 - 배치: {batch_id}")
        
        # 현재 진행 상황 즉시 전송
        if batch_id in self.progress_data:
            try:
                await websocket.send_json(self.progress_data[batch_id])
            except Exception as e:
                logger.error(f"❌ WebSocket 초기 데이터 전송 실패: {e}")
                await self.remove_websocket_connection(batch_id, websocket)
    
    async def remove_websocket_connection(self, batch_id: str, websocket: WebSocket):
        """WebSocket 연결 제거"""
        self.websocket_connections[batch_id].discard(websocket)
        if not self.websocket_connections[batch_id]:
            del self.websocket_connections[batch_id]
        logger.info(f"📡 WebSocket 연결 제거 - 배치: {batch_id}")
    
    async def add_sse_connection(self, batch_id: str, response_queue: asyncio.Queue):
        """SSE 연결 추가"""
        self.sse_connections[batch_id].add(response_queue)
        logger.info(f"📡 SSE 연결 추가 - 배치: {batch_id}")
        
        # 현재 진행 상황 즉시 전송
        if batch_id in self.progress_data:
            try:
                await response_queue.put(self.progress_data[batch_id])
            except Exception as e:
                logger.error(f"❌ SSE 초기 데이터 전송 실패: {e}")
    
    async def remove_sse_connection(self, batch_id: str, response_queue: asyncio.Queue):
        """SSE 연결 제거"""
        self.sse_connections[batch_id].discard(response_queue)
        if not self.sse_connections[batch_id]:
            del self.sse_connections[batch_id]
        logger.info(f"📡 SSE 연결 제거 - 배치: {batch_id}")
    
    async def update_progress(self, batch_id: str, progress_data: Dict):
        """진행 상황 업데이트 및 전송"""
        # 타임스탬프 추가
        progress_data["timestamp"] = datetime.now().isoformat()
        progress_data["batch_id"] = batch_id
        
        # 진행 상황 저장
        self.progress_data[batch_id] = progress_data
        
        # WebSocket으로 전송
        await self._send_to_websockets(batch_id, progress_data)
        
        # SSE로 전송
        await self._send_to_sse(batch_id, progress_data)
        
        logger.debug(f"📊 진행 상황 업데이트 - 배치: {batch_id}, 진행률: {progress_data.get('progress', 0):.1%}")
    
    async def _send_to_websockets(self, batch_id: str, data: Dict):
        """WebSocket 연결들에 데이터 전송"""
        if batch_id not in self.websocket_connections:
            return
            
        dead_connections = set()
        for websocket in self.websocket_connections[batch_id].copy():
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.error(f"❌ WebSocket 전송 실패: {e}")
                dead_connections.add(websocket)
        
        # 죽은 연결 제거
        for websocket in dead_connections:
            await self.remove_websocket_connection(batch_id, websocket)
    
    async def _send_to_sse(self, batch_id: str, data: Dict):
        """SSE 연결들에 데이터 전송"""
        if batch_id not in self.sse_connections:
            return
            
        dead_connections = set()
        for response_queue in self.sse_connections[batch_id].copy():
            try:
                await response_queue.put(data)
            except Exception as e:
                logger.error(f"❌ SSE 전송 실패: {e}")
                dead_connections.add(response_queue)
        
        # 죽은 연결 제거
        for response_queue in dead_connections:
            await self.remove_sse_connection(batch_id, response_queue)
    
    async def cleanup_batch(self, batch_id: str):
        """배치 완료 시 연결 정리"""
        # 최종 상태 전송
        if batch_id in self.progress_data:
            final_data = self.progress_data[batch_id].copy()
            final_data["status"] = "completed"
            final_data["completed_at"] = datetime.now().isoformat()
            
            await self._send_to_websockets(batch_id, final_data)
            await self._send_to_sse(batch_id, final_data)
        
        # 연결 정리
        if batch_id in self.websocket_connections:
            for websocket in self.websocket_connections[batch_id].copy():
                try:
                    await websocket.close()
                except:
                    pass
            del self.websocket_connections[batch_id]
        
        if batch_id in self.sse_connections:
            del self.sse_connections[batch_id]
        
        # 진행 상황 데이터 정리 (1시간 후)
        await asyncio.sleep(3600)
        if batch_id in self.progress_data:
            del self.progress_data[batch_id]
        
        logger.info(f"🧹 배치 {batch_id} 진행 상황 데이터 정리 완료")

# 전역 진행 상황 알리미
progress_notifier = ProgressNotifier()

# ============================================================================
# 실시간 진행 상황 전달 엔드포인트 (WebSocket & SSE)
# ============================================================================

@app.websocket("/batch/progress/{batch_id}")
async def websocket_batch_progress(websocket: WebSocket, batch_id: str):
    """WebSocket으로 배치 처리 진행 상황 실시간 전달"""
    try:
        await progress_notifier.add_websocket_connection(batch_id, websocket)
        
        # 연결 유지 및 메시지 처리
        while True:
            try:
                # 클라이언트로부터 ping 메시지 수신 (연결 유지용)
                data = await websocket.receive_text()
                
                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "status":
                    # 현재 상태 요청 시 즉시 전송
                    if batch_id in progress_notifier.progress_data:
                        await websocket.send_json(progress_notifier.progress_data[batch_id])
                        
            except WebSocketDisconnect:
                logger.info(f"📡 WebSocket 연결 종료 - 배치: {batch_id}")
                break
                
    except Exception as e:
        logger.error(f"❌ WebSocket 오류: {e}")
    finally:
        await progress_notifier.remove_websocket_connection(batch_id, websocket)

@app.get("/batch/progress/{batch_id}")
async def sse_batch_progress(batch_id: str):
    """Server-Sent Events로 배치 처리 진행 상황 실시간 전달"""
    
    async def event_stream():
        response_queue = asyncio.Queue()
        
        try:
            # SSE 연결 등록
            await progress_notifier.add_sse_connection(batch_id, response_queue)
            
            while True:
                try:
                    # 진행 상황 데이터 대기 (타임아웃 30초)
                    data = await asyncio.wait_for(response_queue.get(), timeout=30.0)
                    
                    # SSE 형식으로 데이터 전송
                    json_data = json.dumps(data, ensure_ascii=False)
                    yield f"data: {json_data}\n\n"
                    
                    # 완료되면 연결 종료
                    if data.get("status") in ["completed", "failed", "cancelled"]:
                        break
                        
                except asyncio.TimeoutError:
                    # 연결 유지를 위한 heartbeat
                    yield f"data: {json.dumps({'heartbeat': True, 'timestamp': datetime.now().isoformat()})}\n\n"
                    
                except Exception as e:
                    logger.error(f"❌ SSE 이벤트 스트림 오류: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"❌ SSE 연결 오류: {e}")
        finally:
            await progress_notifier.remove_sse_connection(batch_id, response_queue)
    
    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

# ============================================================================
# 기존 배치 처리 엔드포인트들 (이미 구현됨)
# ============================================================================

class HallucinationFilter:
    """Whisper 환각 현상 감지 및 필터링 클래스"""
    
    def __init__(self):
        self.repetition_threshold = 3  # 3회 이상 반복 시 환각으로 판단
        self.min_segment_duration = 5.0  # 5초 이상 긴 세그먼트 주의
        self.max_repetition_ratio = 0.7  # 반복률 70% 이상 시 환각 의심
        
    def detect_hallucination(self, text: str, duration: float) -> Dict[str, any]:
        """환각 현상 감지"""
        words = text.strip().split()
        
        if len(words) < 3:
            return {'is_hallucination': False, 'confidence': 1.0}
        
        # 단어 반복 분석
        word_counts = Counter(words)
        most_common = word_counts.most_common(1)[0] if word_counts else ('', 0)
        repeated_word, repeat_count = most_common
        
        repetition_ratio = repeat_count / len(words) if len(words) > 0 else 0
        
        # 환각 판단 기준
        is_repetitive = repeat_count >= self.repetition_threshold
        is_high_ratio = repetition_ratio >= self.max_repetition_ratio
        is_long_duration = duration >= self.min_segment_duration
        
        # 특별한 반복 패턴 감지 (아이콘이, 그런데, 그래서 등)
        suspicious_patterns = ['아이콘이', '그런데', '그래서', '그리고', '그냥', '근데']
        is_suspicious_word = repeated_word in suspicious_patterns
        
        is_hallucination = (is_repetitive and is_high_ratio) or (is_suspicious_word and is_repetitive)
        
        confidence_penalty = 0.0
        if is_hallucination:
            confidence_penalty = min(0.8, repetition_ratio * 0.9)  # 최대 80% 신뢰도 감소
        
        return {
            'is_hallucination': is_hallucination,
            'repeated_word': repeated_word,
            'repeat_count': repeat_count,
            'repetition_ratio': repetition_ratio,
            'confidence_penalty': confidence_penalty,
            'original_confidence': 1.0,
            'adjusted_confidence': max(0.1, 1.0 - confidence_penalty)
        }
    
    def filter_segment(self, segment: Dict) -> Dict:
        """세그먼트 필터링 및 신뢰도 조정"""
        text = segment.get('text', '').strip()
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        duration = end - start
        
        hallucination_info = self.detect_hallucination(text, duration)
        
        if hallucination_info['is_hallucination']:
            logger.warning(f"환각 감지: {start:.1f}s-{end:.1f}s, 단어: '{hallucination_info['repeated_word']}' ({hallucination_info['repeat_count']}회, {hallucination_info['repetition_ratio']:.1%})")
            
            # 환각 구간 처리 옵션
            # 1. 빈 텍스트로 변경 (완전 제거)
            # 2. 단일 인스턴스만 남기기
            # 3. 신뢰도만 크게 낮추기
            
            # 옵션 2: 단일 인스턴스만 남기기
            words = text.split()
            unique_words = []
            word_counts = {}
            
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                if word_counts[word] <= 2:  # 최대 2번까지만 허용
                    unique_words.append(word)
            
            filtered_text = ' '.join(unique_words)
            
            # 너무 짧아진 경우 아예 제거
            if len(filtered_text.strip()) < 3:
                filtered_text = ""
            
            segment['text'] = filtered_text
            segment['confidence'] = hallucination_info['adjusted_confidence']
            segment['hallucination_detected'] = True
            segment['original_text'] = text
        else:
            segment['hallucination_detected'] = False
            
        return segment

# GPU 최적화된 STT 서버 클래스에 환각 필터 추가
class GPUOptimizedSTTServer:
    def __init__(self):
        # ... existing initialization ...
        self.hallucination_filter = HallucinationFilter()  # 환각 필터 추가

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Optimized STT Server with Model Selection")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8004, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    # 모델 선택 옵션
    parser.add_argument("--model", 
                       choices=STTServiceFactory.get_available_model_types(),
                       default="whisper",
                       help="STT 모델 타입 선택 (whisper 또는 nemo)")
    parser.add_argument("--list-models", 
                       action="store_true",
                       help="지원되는 모델 목록 출력")
    
    args = parser.parse_args()
    
    # 지원 모델 목록 출력
    if args.list_models:
        print("🤖 지원되는 STT 모델:")
        supported_models = STTServiceFactory.get_supported_models()
        for model_type, info in supported_models.items():
            if info.get("available", True):
                print(f"\n📦 {model_type.upper()} ({info['description']}):")
                for model in info['models']:
                    marker = " (기본)" if model == info['default'] else ""
                    print(f"  - {model}{marker}")
            else:
                print(f"\n❌ {model_type.upper()} - 사용 불가능")
        
        if not STTServiceFactory.is_nemo_available():
            print(f"\n⚠️  NeMo 모델을 사용하려면 다음 명령을 실행하세요:")
            print(f"   pip install nemo-toolkit[asr] omegaconf hydra-core")
        
        sys.exit(0)
    
    # 모델 설정
    SERVER_MODEL_TYPE = args.model
    SERVER_MODEL_NAME = STTServiceFactory.get_default_model(SERVER_MODEL_TYPE)
    
    # 모델 유효성 검증
    if not STTServiceFactory.validate_model(SERVER_MODEL_TYPE, SERVER_MODEL_NAME):
        print(f"❌ 모델 설정 오류:")
        if SERVER_MODEL_TYPE == "nemo" and not STTServiceFactory.is_nemo_available():
            print("NeMo 패키지가 설치되지 않았습니다.")
            print("다음 명령을 실행하여 설치하세요:")
            print("pip install nemo-toolkit[asr] omegaconf hydra-core")
        else:
            print(f"모델 타입 '{SERVER_MODEL_TYPE}'이 지원되지 않습니다.")
        sys.exit(1)
    
    # 서버 설정
    SERVER_HOST = args.host
    SERVER_PORT = args.port
    
    print(f"🚀 STT 서버 시작:")
    print(f"   모델: {SERVER_MODEL_TYPE} ({SERVER_MODEL_NAME})")
    print(f"   주소: {SERVER_HOST}:{SERVER_PORT}")
    print(f"   워커 수: {args.workers}")
    
    # NeMo 경고 메시지
    if SERVER_MODEL_TYPE == "nemo" and not STTServiceFactory.is_nemo_available():
        print("❌ NeMo 패키지가 설치되지 않았습니다.")
        print("서버 시작에 실패할 수 있습니다.")
    
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, workers=args.workers)


