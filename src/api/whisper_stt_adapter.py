#!/usr/bin/env python3
"""
Whisper STT Service Adapter
기존 FasterWhisperSTTService를 BaseSTTService 인터페이스에 맞춰 래핑
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import numpy as np

from .base_stt_service import BaseSTTService
from .stt_service import FasterWhisperSTTService

logger = logging.getLogger(__name__)


class WhisperSTTServiceAdapter(BaseSTTService):
    """기존 FasterWhisperSTTService를 BaseSTTService 인터페이스에 맞춰 래핑"""
    
    def __init__(self, model_name: str = "large-v3", device: str = "cuda", compute_type: str = "float16", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.whisper_service = FasterWhisperSTTService(
            model_size=model_name,
            device=device,
            compute_type=compute_type,
            **kwargs
        )
        logger.info(f"🎯 Whisper STT 서비스 어댑터 생성: {model_name}")
    
    async def initialize(self) -> bool:
        """Whisper 모델 초기화"""
        try:
            logger.info(f"🤖 Whisper STT 모델 초기화 중: {self.model_name}")
            success = await self.whisper_service.initialize()
            
            if success:
                self.is_initialized = True
                self.initialization_error = None
                logger.info(f"✅ Whisper STT 모델 초기화 완료: {self.model_name}")
            else:
                self.is_initialized = False
                self.initialization_error = self.whisper_service.initialization_error
                logger.error(f"❌ Whisper STT 모델 초기화 실패: {self.initialization_error}")
            
            return success
            
        except Exception as e:
            error_msg = f"Whisper 모델 초기화 실패: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.initialization_error = error_msg
            self.is_initialized = False
            return False
    
    async def transcribe(self, audio_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Whisper를 사용한 오디오 전사"""
        if not self.is_initialized:
            raise RuntimeError("Whisper 모델이 초기화되지 않았습니다")
        
        try:
            # 기존 Whisper 서비스 사용
            result = await self.whisper_service.transcribe(audio_data, **kwargs)
            
            # 결과에 모델 정보 추가
            result["model_type"] = "whisper"
            result["model_name"] = self.model_name
            
            # 통계 업데이트
            processing_time = result.get("processing_time", 0)
            audio_duration = result.get("audio_duration", 0)
            self.update_stats(processing_time, audio_duration, success=True)
            
            logger.info(f"✅ Whisper 전사 완료: {len(result.get('text', ''))}자, RTF: {result.get('rtf', 0):.3f}")
            return result
            
        except Exception as e:
            error_msg = f"Whisper 전사 중 오류: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            # 통계 업데이트 (실패)
            audio_duration = len(audio_data.flatten()) / self.sample_rate if len(audio_data.shape) > 0 else 0
            self.update_stats(0, audio_duration, success=False)
            
            raise RuntimeError(error_msg)
    
    async def transcribe_audio(self, audio_data: str, audio_format: str = "pcm_16khz", language: str = "ko", **kwargs) -> Dict[str, Any]:
        """서버 호환성을 위한 transcribe_audio 메서드"""
        # 기존 서비스의 transcribe_audio 메서드 호출
        return await self.whisper_service.transcribe_audio(audio_data, audio_format, language, **kwargs)
    
    def is_healthy(self) -> bool:
        """Whisper 서비스 상태 확인"""
        return self.is_initialized and self.whisper_service.is_healthy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Whisper 모델 정보 반환"""
        # 기존 서비스의 모델 정보 가져오기
        info = self.whisper_service.get_model_info()
        
        # 모델 타입 정보 추가
        info["model_type"] = "whisper"
        
        return info
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환 (기존 서비스와 어댑터 통계 결합)"""
        # 기존 서비스 통계
        whisper_stats = self.whisper_service.stats
        
        # 어댑터 통계와 결합
        combined_stats = self.stats.copy()
        combined_stats.update({
            "whisper_stats": whisper_stats,
            "adapter_stats": super().get_stats()
        })
        
        return combined_stats 