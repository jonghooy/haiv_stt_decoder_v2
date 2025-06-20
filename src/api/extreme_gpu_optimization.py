#!/usr/bin/env python3
"""
극한 GPU 최적화 모듈
RTF < 0.05x 달성을 위한 고급 최적화 기법들
"""

import torch
import torch.nn as nn
import torch.cuda.profiler as profiler
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
import gc
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """배치 처리용 요청 데이터"""
    request_id: str
    audio_data: np.ndarray
    language: str
    priority: str = "medium"

class ExtremeGPUOptimizer:
    """극한 GPU 최적화 클래스"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.batch_size = 8  # 동적 조정 가능
        self.max_batch_wait_ms = 10  # 배치 대기 시간
        self.optimization_level = "extreme"
        
        # 고급 GPU 설정
        self._apply_extreme_gpu_settings()
        
    def _apply_extreme_gpu_settings(self):
        """극한 GPU 설정 적용"""
        if not torch.cuda.is_available():
            return
            
        logger.info("🔥 극한 GPU 최적화 설정 적용 중...")
        
        # 1. 메모리 최적화
        torch.cuda.empty_cache()
        gc.collect()
        
        # 2. 컴퓨팅 모드 최적화
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        
        # 3. 메모리 풀 최적화
        torch.cuda.set_per_process_memory_fraction(0.98)  # 최대 사용
        
        # 4. 스트림 최적화
        torch.cuda.current_stream().synchronize()
        
        # 5. JIT 컴파일 활성화
        torch.jit.set_fusion_strategy([("STATIC", 2), ("DYNAMIC", 2)])
        
        logger.info("✅ 극한 GPU 설정 완료")

class BatchProcessor:
    """배치 처리를 위한 클래스"""
    
    def __init__(self, stt_model, batch_size: int = 8, max_wait_ms: int = 10):
        self.stt_model = stt_model
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_requests: List[BatchRequest] = []
        self.processing = False
        
    async def add_request(self, request: BatchRequest) -> Dict[str, Any]:
        """요청을 배치에 추가하고 결과 반환"""
        self.pending_requests.append(request)
        
        # 배치가 가득 차거나 대기 시간 초과시 처리
        if len(self.pending_requests) >= self.batch_size or self._should_process_batch():
            return await self._process_batch()
        
        # 짧은 대기 후 다시 확인
        await asyncio.sleep(self.max_wait_ms / 1000.0)
        if self.pending_requests and not self.processing:
            return await self._process_batch()
    
    def _should_process_batch(self) -> bool:
        """배치 처리 여부 결정"""
        if not self.pending_requests:
            return False
        
        # 높은 우선순위 요청이 있으면 즉시 처리
        high_priority = any(req.priority == "high" for req in self.pending_requests)
        return high_priority or len(self.pending_requests) >= self.batch_size
    
    async def _process_batch(self) -> List[Dict[str, Any]]:
        """배치 처리 실행"""
        if self.processing or not self.pending_requests:
            return []
        
        self.processing = True
        batch = self.pending_requests[:self.batch_size]
        self.pending_requests = self.pending_requests[self.batch_size:]
        
        try:
            # 배치 처리
            results = await self._process_audio_batch(batch)
            return results
        finally:
            self.processing = False
    
    async def _process_audio_batch(self, batch: List[BatchRequest]) -> List[Dict[str, Any]]:
        """실제 배치 오디오 처리"""
        start_time = time.time()
        
        # GPU에서 병렬 처리
        audio_arrays = [req.audio_data for req in batch]
        
        # 배치 처리를 위한 텐서 준비
        max_length = max(len(audio) for audio in audio_arrays)
        
        # 패딩 및 스택
        padded_audios = []
        for audio in audio_arrays:
            if len(audio) < max_length:
                padded = np.pad(audio, (0, max_length - len(audio)))
            else:
                padded = audio[:max_length]
            padded_audios.append(padded)
        
        # GPU 텐서로 변환
        batch_tensor = torch.from_numpy(np.stack(padded_audios)).to(self.stt_model.device)
        
        # 병렬 추론
        with torch.no_grad():
            batch_results = await self._parallel_inference(batch_tensor, batch)
        
        processing_time = time.time() - start_time
        
        results = []
        for i, (req, result) in enumerate(zip(batch, batch_results)):
            audio_duration = len(req.audio_data) / 16000.0
            rtf = processing_time / audio_duration if audio_duration > 0 else 0
            
            results.append({
                "request_id": req.request_id,
                "text": result.get("text", ""),
                "language": result.get("language", req.language),
                "rtf": rtf,
                "processing_time": processing_time / len(batch),
                "audio_duration": audio_duration,
                "batch_processed": True,
                "batch_size": len(batch)
            })
        
        return results
    
    async def _parallel_inference(self, batch_tensor: torch.Tensor, batch: List[BatchRequest]) -> List[Dict[str, Any]]:
        """병렬 추론 실행"""
        # 여기서 실제 모델 추론을 배치로 처리
        # 현재는 개별 처리를 시뮬레이션
        results = []
        
        for i, req in enumerate(batch):
            audio_slice = batch_tensor[i].cpu().numpy()
            
            # 개별 추론 (실제로는 배치 추론으로 대체해야 함)
            result = await self._single_inference(audio_slice, req)
            results.append(result)
        
        return results
    
    async def _single_inference(self, audio: np.ndarray, req: BatchRequest) -> Dict[str, Any]:
        """단일 추론 (배치 추론 구현까지의 임시 처리)"""
        # 실제 STT 모델 호출
        try:
            # 오디오를 바이트로 변환
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            result = await self.stt_model.transcribe_file_bytes(audio_bytes, language=req.language)
            
            return {
                "text": result.text,
                "language": result.language,
                "confidence": getattr(result, 'confidence', 0.0)
            }
        except Exception as e:
            logger.error(f"배치 추론 오류: {e}")
            return {"text": "", "language": req.language, "confidence": 0.0}

class MemoryOptimizer:
    """메모리 최적화 클래스"""
    
    @staticmethod
    def optimize_memory():
        """메모리 최적화 실행"""
        if not torch.cuda.is_available():
            return
        
        # 1. 가비지 컬렉션
        gc.collect()
        torch.cuda.empty_cache()
        
        # 2. 메모리 풀 최적화
        try:
            torch.cuda.memory.set_per_process_memory_fraction(0.98)
        except Exception as e:
            logger.warning(f"메모리 풀 설정 실패: {e}")
        
        # 3. 메모리 할당 전략 최적화 (PyTorch 버전 확인)
        try:
            if hasattr(torch.cuda.memory, 'set_allocator_settings'):
                # PyTorch 2.5+ 호환성을 위해 주석 처리
                # torch.cuda.memory.set_allocator_settings("backend:cudaMallocAsync")
                logger.info("⚠️ set_allocator_settings는 PyTorch 2.5+에서 사용할 수 없습니다")
            else:
                # 대안: 기본 메모리 관리 최적화
                torch.cuda.empty_cache()
                logger.info("✅ CUDA 메모리 캐시 정리 완료")
        except Exception as e:
            logger.warning(f"⚠️ 메모리 할당자 설정 실패: {e}")
        
        logger.info("✅ 메모리 최적화 완료")

class ModelOptimizer:
    """모델 최적화 클래스"""
    
    @staticmethod
    def optimize_model_for_speed(model):
        """속도를 위한 모델 최적화"""
        if not hasattr(model, 'model'):
            return model
        
        try:
            # 1. JIT 컴파일
            if hasattr(model.model, 'encoder'):
                model.model.encoder = torch.jit.script(model.model.encoder)
            
            # 2. 혼합 정밀도 활성화
            if hasattr(model.model, 'half'):
                model.model = model.model.half()
            
            # 3. 평가 모드로 전환
            model.model.eval()
            
            logger.info("✅ 모델 속도 최적화 완료")
            
        except Exception as e:
            logger.warning(f"모델 최적화 부분 실패: {e}")
        
        return model

class StreamProcessor:
    """스트림 처리 최적화"""
    
    def __init__(self, num_streams: int = 4):
        self.num_streams = num_streams
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream = 0
        
    def get_next_stream(self) -> torch.cuda.Stream:
        """다음 스트림 반환"""
        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % self.num_streams
        return stream
    
    async def process_with_stream(self, func, *args, **kwargs):
        """스트림을 사용한 비동기 처리"""
        stream = self.get_next_stream()
        
        with torch.cuda.stream(stream):
            result = await func(*args, **kwargs)
            
        # 스트림 동기화
        stream.synchronize()
        return result

class ExtremeOptimizedSTTService:
    """극한 최적화된 STT 서비스"""
    
    def __init__(self, base_stt_service):
        self.base_service = base_stt_service
        self.gpu_optimizer = ExtremeGPUOptimizer()
        self.batch_processor = BatchProcessor(base_stt_service, batch_size=8)
        self.memory_optimizer = MemoryOptimizer()
        self.stream_processor = StreamProcessor(num_streams=4)
        
        # 모델 최적화
        self.base_service = ModelOptimizer.optimize_model_for_speed(self.base_service)
        
        logger.info("🚀 극한 최적화 STT 서비스 초기화 완료")
    
    async def transcribe_file_bytes(self, audio_bytes: bytes, language: str = "ko") -> Dict[str, Any]:
        """극한 최적화된 음성 인식"""
        start_time = time.time()
        
        # 메모리 최적화
        self.memory_optimizer.optimize_memory()
        
        # 오디오 전처리
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 배치 요청 생성
        request = BatchRequest(
            request_id=f"extreme_{int(time.time() * 1000000)}",
            audio_data=audio_array,
            language=language,
            priority="high"  # 최우선 처리
        )
        
        # 스트림 처리
        result = await self.stream_processor.process_with_stream(
            self._process_single_request, request
        )
        
        processing_time = time.time() - start_time
        audio_duration = len(audio_array) / 16000.0
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        
        # 결과 포맷팅
        return type('STTResult', (), {
            'text': result.get('text', ''),
            'language': result.get('language', language),
            'rtf': rtf,
            'processing_time': processing_time,
            'audio_duration': audio_duration,
            'optimization_applied': 'extreme'
        })()
    
    async def _process_single_request(self, request: BatchRequest) -> Dict[str, Any]:
        """단일 요청 처리 (극한 최적화)"""
        # 기존 서비스 호출 (추후 배치로 대체)
        audio_bytes = (request.audio_data * 32767).astype(np.int16).tobytes()
        
        # GPU 최적화된 처리
        with torch.cuda.amp.autocast():
            result = await self.base_service.transcribe_file_bytes(audio_bytes, request.language)
        
        return {
            'text': result.text,
            'language': result.language,
            'confidence': getattr(result, 'confidence', 0.0)
        }

def create_extreme_optimized_service(base_stt_service):
    """극한 최적화 STT 서비스 생성"""
    return ExtremeOptimizedSTTService(base_stt_service) 