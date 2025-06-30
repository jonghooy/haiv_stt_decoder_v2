#!/usr/bin/env python3
"""
Base STT Service
추상 기본 클래스 - Whisper와 NeMo STT 서비스의 공통 인터페이스
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class BaseSTTService(ABC):
    """STT 서비스의 추상 기본 클래스"""
    
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        self.model_name = model_name
        self.device = device
        self.is_initialized = False
        self.initialization_error: Optional[str] = None
        
        # 공통 설정
        self.sample_rate = 16000
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_audio_duration': 0.0,
            'total_processing_time': 0.0,
            'avg_rtf': 0.0,
            'avg_latency_ms': 0.0
        }
    
    @abstractmethod
    async def initialize(self) -> bool:
        """서비스 초기화"""
        pass
    
    @abstractmethod
    async def transcribe(self, audio_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """오디오 전사"""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """서비스 상태 확인"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """서비스 통계 반환"""
        return self.stats.copy()
    
    def update_stats(self, processing_time: float, audio_duration: float, success: bool = True):
        """통계 업데이트"""
        self.stats['total_requests'] += 1
        self.stats['total_processing_time'] += processing_time
        self.stats['total_audio_duration'] += audio_duration
        
        if success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
        
        # RTF 계산
        if self.stats['total_audio_duration'] > 0:
            self.stats['avg_rtf'] = self.stats['total_processing_time'] / self.stats['total_audio_duration']
        
        # 평균 지연 시간 계산 (ms)
        if self.stats['total_requests'] > 0:
            self.stats['avg_latency_ms'] = (self.stats['total_processing_time'] / self.stats['total_requests']) * 1000 