#!/usr/bin/env python3
"""
STT Service Factory
모델 타입에 따라 적절한 STT 서비스를 생성하는 팩토리 클래스
"""

import logging
from typing import Dict, Any, Optional

from .base_stt_service import BaseSTTService
from .whisper_stt_adapter import WhisperSTTServiceAdapter

logger = logging.getLogger(__name__)

# NeMo 지원 확인
try:
    from .nemo_stt_service import NeMoSTTService, NEMO_AVAILABLE
except ImportError:
    NEMO_AVAILABLE = False
    logger.warning("⚠️ NeMo STT 서비스를 가져올 수 없습니다")

# 지원되는 모델 타입과 이름들
SUPPORTED_MODELS = {
    "whisper": {
        "models": ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"],
        "default": "large-v3",
        "description": "OpenAI Whisper 모델 (Faster Whisper 구현)"
    },
    "nemo": {
        "models": [
            "./FastConformer-Transducer-BPE_9.75.nemo",
            "./FastConformer-Transducer-BPE_6.00.nemo",
            "eesungkim/stt_kr_conformer_transducer_large",
            "eesungkim/korean-stt-model",
            "nvidia/stt_ko_conformer_ctc_large",
            "nvidia/stt_ko_conformer_transducer_large",
            "nvidia/stt_en_conformer_ctc_large",
            "nvidia/stt_en_conformer_transducer_large"
        ],
        "default": "./FastConformer-Transducer-BPE_9.75.nemo",
        "description": "NVIDIA NeMo ASR 모델 (로컬 커스텀 모델 포함)",
        "available": NEMO_AVAILABLE
    }
}


class STTServiceFactory:
    """STT 서비스 생성 팩토리"""
    
    @staticmethod
    def create_service(model_type: str, model_name: str, device: str = "cuda", **kwargs) -> BaseSTTService:
        """모델 타입에 따라 적절한 STT 서비스 생성"""
        model_type = model_type.lower()
        
        logger.info(f"🏭 STT 서비스 생성: {model_type} - {model_name}")
        
        if model_type == "whisper":
            return WhisperSTTServiceAdapter(
                model_name=model_name,
                device=device,
                compute_type=kwargs.get("compute_type", "float16"),
                **{k: v for k, v in kwargs.items() if k != "compute_type"}
            )
        elif model_type == "nemo":
            if not NEMO_AVAILABLE:
                raise ValueError(
                    "NeMo 패키지가 설치되지 않았습니다. "
                    "'pip install nemo-toolkit[asr] omegaconf hydra-core'를 실행하세요"
                )
            return NeMoSTTService(
                model_name=model_name,
                device=device,
                **kwargs
            )
        else:
            supported_types = [k for k, v in SUPPORTED_MODELS.items() if v.get("available", True)]
            raise ValueError(
                f"지원하지 않는 모델 타입: {model_type}. "
                f"지원 모델: {supported_types}"
            )
    
    @staticmethod
    def get_supported_models() -> Dict[str, Any]:
        """지원되는 모델 목록 반환"""
        return SUPPORTED_MODELS
    
    @staticmethod
    def validate_model(model_type: str, model_name: Optional[str] = None) -> bool:
        """모델 타입과 이름 유효성 검증"""
        model_type = model_type.lower()
        
        # 모델 타입 확인
        if model_type not in SUPPORTED_MODELS:
            return False
        
        model_info = SUPPORTED_MODELS[model_type]
        
        # 모델 사용 가능성 확인
        if not model_info.get("available", True):
            return False
        
        # 모델 이름 확인 (제공된 경우)
        if model_name and model_name not in model_info["models"]:
            # 엄격한 검증을 하지 않음 (사용자 정의 모델 허용)
            logger.warning(f"⚠️ 모델 {model_name}은 공식 지원 목록에 없습니다")
        
        return True
    
    @staticmethod
    def get_default_model(model_type: str) -> str:
        """모델 타입의 기본 모델 이름 반환"""
        model_type = model_type.lower()
        
        if model_type in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[model_type]["default"]
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    @staticmethod
    def get_model_description(model_type: str) -> str:
        """모델 타입 설명 반환"""
        model_type = model_type.lower()
        
        if model_type in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[model_type]["description"]
        else:
            return f"알 수 없는 모델 타입: {model_type}"
    
    @staticmethod
    def is_nemo_available() -> bool:
        """NeMo 지원 여부 확인"""
        return NEMO_AVAILABLE
    
    @staticmethod
    def get_available_model_types() -> list:
        """사용 가능한 모델 타입 목록 반환 (설치 여부와 관계없이 모든 타입 반환)"""
        return list(SUPPORTED_MODELS.keys())
    
    @staticmethod
    def get_usable_model_types() -> list:
        """실제 사용 가능한 모델 타입 목록 반환 (설치 여부 확인)"""
        return [
            model_type for model_type, info in SUPPORTED_MODELS.items()
            if info.get("available", True)
        ] 