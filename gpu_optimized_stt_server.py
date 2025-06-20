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
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

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
        # STT 서비스의 내부 모델에 직접 접근 (Large-v3 최적화 파라미터)
        if hasattr(stt_service, 'model') and hasattr(stt_service.model, 'transcribe'):
            # FasterWhisper 모델의 transcribe 메서드 직접 호출 (Large-v3 최적화)
            segments, info = stt_service.model.transcribe(
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

from src.api.stt_service import FasterWhisperSTTService
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

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 극한 GPU 최적화 실행 (logger 정의 후)
setup_extreme_gpu_optimizations()

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
    Whisper의 avg_logprob을 신뢰도 점수(0.0-1.0)로 변환
    
    Args:
        avg_logprob: Whisper 세그먼트의 평균 로그 확률
        
    Returns:
        0.0-1.0 범위의 신뢰도 점수
    """
    if avg_logprob is None:
        return 0.0
    
    # avg_logprob은 일반적으로 -inf ~ 0 범위
    # -1.0 이상이면 높은 신뢰도, -3.0 이하면 낮은 신뢰도로 간주
    if avg_logprob >= -0.5:
        return 0.95
    elif avg_logprob >= -1.0:
        return 0.8 + (avg_logprob + 1.0) * 0.3  # -1.0~-0.5 -> 0.8~0.95
    elif avg_logprob >= -2.0:
        return 0.5 + (avg_logprob + 2.0) * 0.3  # -2.0~-1.0 -> 0.5~0.8
    elif avg_logprob >= -3.0:
        return 0.2 + (avg_logprob + 3.0) * 0.3  # -3.0~-2.0 -> 0.2~0.5
    else:
        return max(0.1, 0.2 + (avg_logprob + 3.0) * 0.1)  # -3.0 이하 -> 0.1~0.2


def normalize_word_probability(probability: float) -> float:
    """
    Whisper 단어 확률을 정규화된 신뢰도로 변환
    
    Args:
        probability: Whisper word probability (보통 0.0-1.0 범위)
        
    Returns:
        정규화된 신뢰도 점수 (0.0-1.0)
    """
    if probability is None:
        return 0.0
    
    # Whisper의 word probability는 이미 0-1 범위이지만,
    # 실제로는 0.3-1.0 범위에서 더 의미있는 값들이 나옴
    if probability >= 0.8:
        return probability  # 높은 신뢰도는 그대로 유지
    elif probability >= 0.5:
        return 0.6 + (probability - 0.5) * 0.6  # 0.5-0.8 -> 0.6-0.8
    elif probability >= 0.3:
        return 0.3 + (probability - 0.3) * 1.5  # 0.3-0.5 -> 0.3-0.6
    else:
        return max(0.1, probability * 1.0)  # 0.3 이하 -> 최소 0.1


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

# 전역 서비스 인스턴스
stt_service: Optional[FasterWhisperSTTService] = None
post_processing_corrector = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 STT 서비스, 후처리 교정 시스템 및 큐잉 시스템 초기화"""
    global stt_service, post_processing_corrector, stt_queue
    try:
        logger.info("🚀 Large-v3 전용 극한 최적화 STT Server 시작 중...")
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
        
        # STT 서비스 생성 및 즉시 초기화 (Large-v3 전용 최적화)
        logger.info("📦 Large-v3 모델 로딩 중 (float16 극한 최적화)...")
        stt_service = FasterWhisperSTTService(
            model_size="large-v3",
            device="cuda",
            compute_type="float16"
        )
        
        # 모델을 미리 로드하여 첫 번째 요청 지연 제거
        start_time = time.time()
        await stt_service.initialize()
        load_time = time.time() - start_time
        
        logger.info(f"✅ Large-v3 STT 서비스 초기화 완료 - 모델 로딩 시간: {load_time:.2f}초")
        
        # 모델 웜업 수행 (첫 요청 지연 최소화)
        await warmup_large_model(stt_service)
        
        # GPU 메모리 상태 출력
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
    """루트 엔드포인트 - Large-v3 전용 극한 최적화 서버"""
    gpu_features = {}
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_features = {
            "gpu_name": torch.cuda.get_device_name(0),
            "total_memory_gb": f"{gpu_props.total_memory / 1024**3:.1f}",
            "architecture": f"{gpu_props.major}.{gpu_props.minor}"
        }
    
    return {
        "message": "Large-v3 전용 극한 최적화 STT API Server", 
        "model": "large-v3",
        "optimization": "extreme",
        "status": "running",
        "features": {
            "model_type": "large-v3",
            "compute_type": "float16", 
            "memory_fraction": 0.95,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
            "gpu_available": torch.cuda.is_available(),
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
                            
                            # FasterWhisper 모델에 직접 접근하여 initial_prompt 사용
                            segments, info = stt_service.model.transcribe(
                                audio_array,
                                beam_size=5,
                                best_of=5,
                                temperature=0.0,
                                vad_filter=False,
                                language=request.language,
                                word_timestamps=True,
                                initial_prompt=initial_prompt  # 키워드 힌트 제공
                            )
                            
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


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Optimized STT Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


