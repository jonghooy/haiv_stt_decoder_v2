#!/usr/bin/env python3
"""
NeMo STT Service
NVIDIA NeMo ASR 모델을 사용한 STT 서비스
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
import numpy as np
import torch

from .base_stt_service import BaseSTTService
from .models import STTResult

logger = logging.getLogger(__name__)

# NeMo 패키지 가져오기 시도
try:
    import nemo
    import nemo.collections.asr as nemo_asr
    from omegaconf import DictConfig, OmegaConf
    NEMO_AVAILABLE = True
    logger.info("✅ NeMo 패키지 가져오기 성공")
except ImportError as e:
    NEMO_AVAILABLE = False
    logger.warning(f"⚠️ NeMo 패키지를 가져올 수 없습니다: {e}")
    logger.warning("NeMo 모델을 사용하려면 'pip install nemo-toolkit[asr]'를 실행하세요")


class NeMoSTTService(BaseSTTService):
    """NeMo ASR 서비스"""
    
    def __init__(self, model_name: str = "./FastConformer-Transducer-BPE_9.75.nemo", device: str = "cuda", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.model: Optional[Any] = None
        
        if not NEMO_AVAILABLE:
            raise ImportError("NeMo 패키지가 설치되지 않았습니다. 'pip install nemo-toolkit[asr] omegaconf hydra-core'를 실행하세요")
    
    def _configure_nemo_compatibility(self):
        """NeMo 호환성 설정"""
        try:
            # use_pytorch_sdpa 호환성 문제 해결을 위한 환경 변수 설정
            import os
            os.environ['NEMO_DISABLE_PYTORCH_SDPA'] = '1'
            os.environ['PYTORCH_DISABLE_SDPA'] = '1'
            
            # OmegaConf 설정
            from omegaconf import OmegaConf
            OmegaConf.set_struct(OmegaConf.create({}), False)
            
            logger.info("🔧 NeMo 호환성 설정 완료")
        except Exception as e:
            logger.warning(f"⚠️ NeMo 호환성 설정 실패 (비중요): {e}")
    
    def _optimize_decoding_config(self):
        """NeMo 모델의 디코딩 설정 최적화"""
        try:
            logger.info("🔧 NeMo 디코딩 설정 최적화 시작...")
            
            if not hasattr(self.model, 'cfg'):
                logger.warning("⚠️ 모델에 cfg 속성이 없어 디코딩 최적화 건너뜀")
                return
            
            cfg = self.model.cfg
            
            # 언어 설정
            if hasattr(cfg, 'language'):
                logger.info(f"🇰🇷 기존 언어 설정: {cfg.language}")
                cfg.language = 'ko'
                logger.info("🇰🇷 한국어로 언어 설정 변경")
            
            # 디코딩 설정 최적화
            if hasattr(cfg, 'decoding'):
                decoding = cfg.decoding
                logger.info(f"🔧 기존 디코딩 전략: {getattr(decoding, 'strategy', 'Unknown')}")
                
                # 빔 서치 설정 최적화
                if hasattr(decoding, 'beam'):
                    beam = decoding.beam
                    original_beam_size = getattr(beam, 'beam_size', 1)
                    
                    # 빔 크기 증가 (더 정확한 결과, 단어 누락 방지)
                    beam.beam_size = max(4, original_beam_size)  # 6 → 4로 감소
                    
                    # 길이 정규화 개선 (더 긴 문장 선호)
                    if hasattr(beam, 'len_pen'):
                        beam.len_pen = 0.5  # 길이 패널티 조정 (0.3 → 0.5로 증가)
                    
                    logger.info(f"🔧 빔 크기: {original_beam_size} -> {beam.beam_size}")
                    logger.info(f"🔧 길이 정규화: {getattr(beam, 'len_pen', 'Unknown')}")
                
                # 전략별 설정
                strategy = getattr(decoding, 'strategy', 'greedy')
                if strategy == 'greedy':
                    # 그리디에서 빔 서치로 변경 시도
                    try:
                        from omegaconf import DictConfig, OmegaConf
                        
                        # 빔 서치 설정 생성 (단어 누락 방지 설정)
                        beam_config = DictConfig({
                            'beam_size': 4,  # 더 보수적인 빔 크기 (6 → 4)
                            'len_pen': 0.5,  # 적절한 길이 패널티
                            'max_generation_delta': -1,
                            'score_norm': True,  # 점수 정규화
                            'return_best_hypothesis': True  # 최고 가설만 반환
                        })
                        
                        # 디코딩 설정 업데이트
                        decoding.strategy = 'beam'
                        decoding.beam = beam_config
                        
                        logger.info("🔧 그리디에서 빔 서치로 변경")
                    except Exception as e:
                        logger.warning(f"⚠️ 빔 서치 변경 실패: {e}")
            
            # 모델에 변경된 설정 적용
            if hasattr(self.model, 'change_decoding_strategy'):
                try:
                    self.model.change_decoding_strategy(cfg.decoding)
                    logger.info("✅ 디코딩 전략 변경 적용 완료")
                except Exception as e:
                    logger.warning(f"⚠️ 디코딩 전략 변경 적용 실패: {e}")
            
            logger.info("✅ NeMo 디코딩 설정 최적화 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 디코딩 설정 최적화 실패: {e}")
            import traceback

    
    def _patch_model_config(self, model_path):
        """모델 설정에서 호환성 문제가 있는 파라미터 제거"""
        try:
            import tempfile
            import tarfile
            import yaml
            import os
            from omegaconf import OmegaConf
            
            # .nemo 파일을 임시로 압축 해제
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(model_path, 'r') as tar:
                    tar.extractall(temp_dir)
                
                # model_config.yaml 수정
                config_path = os.path.join(temp_dir, 'model_config.yaml')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # SDPA 관련 모든 파라미터들을 제거
                    sdpa_keywords = [
                        'use_pytorch_sdpa',
                        'use_pytorch_sdpa_backends',
                        'pytorch_sdpa',
                        'sdpa_backends',
                        'enable_flash_attention',
                        'flash_attention'
                    ]
                    
                    def remove_problematic_params(cfg, path="root"):
                        if isinstance(cfg, dict):
                            # SDPA 관련 키 제거
                            keys_to_remove = []
                            for key in cfg.keys():
                                for keyword in sdpa_keywords:
                                    if keyword in key.lower():
                                        keys_to_remove.append(key)
                                        break
                            
                            for key in keys_to_remove:
                                logger.info(f"🔧 {path}.{key} 파라미터 제거")
                                del cfg[key]
                            
                            # 모든 하위 딕셔너리에 대해서도 재귀적으로 처리
                            for key, value in cfg.items():
                                if isinstance(value, dict):
                                    remove_problematic_params(value, f"{path}.{key}")
                                elif isinstance(value, list):
                                    for i, item in enumerate(value):
                                        if isinstance(item, dict):
                                            remove_problematic_params(item, f"{path}.{key}[{i}]")
                    
                    remove_problematic_params(config)
                    
                    # 추가로 encoder 설정에서 특정 문제 파라미터들 제거
                    if 'encoder' in config:
                        encoder_config = config['encoder']
                        if isinstance(encoder_config, dict):
                            # Conformer encoder의 문제가 되는 파라미터들 제거
                            problematic_keys = [
                                'use_pytorch_sdpa_backends',
                                'use_pytorch_sdpa',
                                'self_attention_model',
                                'rel_pos_enc_type'
                            ]
                            
                            for key in problematic_keys:
                                if key in encoder_config:
                                    logger.info(f"🔧 encoder.{key} 파라미터 제거")
                                    del encoder_config[key]
                    
                    # 수정된 설정 저장
                    with open(config_path, 'w') as f:
                        yaml.safe_dump(config, f)
                    
                    # 새로운 .nemo 파일 생성
                    patched_path = model_path.replace('.nemo', '_patched.nemo')
                    with tarfile.open(patched_path, 'w') as tar:
                        for item in os.listdir(temp_dir):
                            tar.add(os.path.join(temp_dir, item), arcname=item)
                    
                    logger.info(f"✅ 패치된 모델 생성: {patched_path}")
                    return patched_path
                else:
                    logger.warning("⚠️ model_config.yaml을 찾을 수 없음")
                    return model_path
                    
        except Exception as e:
            logger.warning(f"⚠️ 모델 설정 패치 실패: {e}")
            return model_path

    async def initialize(self) -> bool:
        """NeMo 모델 초기화"""
        try:
            logger.info(f"🤖 NeMo STT 모델 초기화 중: {self.model_name}")
            start_time = time.time()
            
            # NeMo 호환성 설정
            self._configure_nemo_compatibility()
            
            # 모델 경로 처리
            import os
            if self.model_name.startswith('./'):
                # 상대 경로를 절대 경로로 변환
                model_path = os.path.abspath(self.model_name)
                logger.info(f"📂 상대 경로를 절대 경로로 변환: {self.model_name} -> {model_path}")
            else:
                model_path = self.model_name
            
            # 모델 로드 방식 개선
            if model_path.endswith('.nemo') or os.path.exists(model_path):
                # 로컬 .nemo 파일 로드
                logger.info(f"📦 로컬 .nemo 파일 로드 중: {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"로컬 모델 파일을 찾을 수 없습니다: {model_path}")
                
                # 호환성 문제가 있는 파라미터를 제거한 패치된 모델 생성
                logger.info("🔧 모델 호환성 패치 적용 중...")
                patched_model_path = self._patch_model_config(model_path)
                
                # restore_from을 사용하여 로컬 .nemo 파일 로드
                try:
                    # 먼저 RNN-T BPE 모델로 시도 (FastConformer-Transducer에 적합)
                    from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
                    logger.info("🔄 EncDecRNNTBPEModel로 로드 시도 중...")
                    
                    # 추가 로드 옵션
                    load_options = {
                        'strict': False,
                        'map_location': 'cpu' if self.device == 'cpu' else None
                    }
                    
                    self.model = EncDecRNNTBPEModel.restore_from(patched_model_path, **load_options)
                    logger.info("✅ EncDecRNNTBPEModel로 로드 성공")
                    
                except Exception as e1:
                    logger.warning(f"⚠️ EncDecRNNTBPEModel로 로드 실패: {e1}")
                    try:
                        # CTC BPE 모델로 시도 (올바른 클래스명 사용)
                        from nemo.collections.asr.models.ctc_models import EncDecCTCModel
                        logger.info("🔄 EncDecCTCModel로 로드 시도 중...")
                        self.model = EncDecCTCModel.restore_from(patched_model_path, strict=False)
                        logger.info("✅ EncDecCTCModel로 로드 성공")
                    except Exception as e2:
                        logger.warning(f"⚠️ EncDecCTCModel로 로드 실패: {e2}")
                        try:
                            # 마지막으로 일반 ASRModel로 시도
                            logger.info("🔄 일반 ASRModel로 로드 시도 중...")
                            self.model = nemo_asr.models.ASRModel.restore_from(patched_model_path, strict=False)
                            logger.info("✅ 일반 ASRModel로 로드 성공")
                        except Exception as e3:
                            logger.error(f"❌ 모든 로컬 로드 방법 실패: {e3}")
                            # 패치된 파일 정리
                            if patched_model_path != model_path and os.path.exists(patched_model_path):
                                os.remove(patched_model_path)
                                logger.info("🧹 패치된 임시 파일 정리 완료")
                            raise Exception(f"모든 NeMo 모델 로드 방법이 실패했습니다. 마지막 에러: {e3}")
            else:
                # Hugging Face 모델 로드
                logger.info(f"📦 Hugging Face에서 모델 로드 중: {model_path}")
                try:
                    # 먼저 EncDecRNNTBPEModel로 시도
                    from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
                    self.model = EncDecRNNTBPEModel.from_pretrained(model_path)
                except Exception as e1:
                    logger.warning(f"⚠️ EncDecRNNTBPEModel로 로드 실패: {e1}")
                    try:
                        # EncDecCTCModel로 시도
                        from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCBPEModel
                        self.model = EncDecCTCBPEModel.from_pretrained(model_path)
                    except Exception as e2:
                        logger.warning(f"⚠️ EncDecCTCBPEModel로 로드 실패: {e2}")
                        # 마지막으로 일반 ASRModel로 시도
                        self.model = nemo_asr.models.ASRModel.from_pretrained(model_path)
            
            # GPU로 모델 이동
            if torch.cuda.is_available() and self.device == "cuda":
                self.model = self.model.to('cuda')
                logger.info(f"✅ 모델을 GPU로 이동: {torch.cuda.get_device_name(0)}")
            
            # 평가 모드로 설정
            self.model.eval()
            
            # NeMo 모델 디코딩 설정 최적화
            self._optimize_decoding_config()
            
            # 모델 최적화 설정
            if torch.cuda.is_available() and self.device == "cuda":
                # 컴파일 최적화 (PyTorch 2.0+)
                try:
                    if hasattr(torch, 'compile'):
                        logger.info("🔧 PyTorch 2.0 컴파일 최적화 적용 중...")
                        self.model = torch.compile(self.model, mode='reduce-overhead')
                        logger.info("✅ PyTorch 컴파일 최적화 완료")
                except Exception as e:
                    logger.warning(f"⚠️ PyTorch 컴파일 최적화 실패: {e}")
            
            load_time = time.time() - start_time
            logger.info(f"✅ NeMo STT 모델 초기화 완료: {load_time:.2f}초")
            
            # 웜업 수행
            await self._warmup_model()
            
            # 패치된 임시 파일 정리 (모델이 메모리에 로드된 후)
            if 'patched_model_path' in locals() and patched_model_path != model_path and os.path.exists(patched_model_path):
                try:
                    os.remove(patched_model_path)
                    logger.info("🧹 패치된 임시 파일 정리 완료")
                except Exception as e:
                    logger.warning(f"⚠️ 임시 파일 정리 실패: {e}")
            
            self.is_initialized = True
            self.initialization_error = None
            return True
            
        except Exception as e:
            error_msg = f"NeMo 모델 초기화 실패: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.initialization_error = error_msg
            self.is_initialized = False
            return False
    
    async def _warmup_model(self):
        """모델 웜업 (첫 요청 지연 최소화)"""
        try:
            logger.info("🔥 NeMo 모델 웜업 시작...")
            
            # 더미 오디오 데이터 생성 (1초, 16kHz)
            dummy_audio = np.random.randn(16000).astype(np.float32) * 0.1
            
            # 웜업 전사 수행
            start_time = time.time()
            await self.transcribe(dummy_audio, language="ko")
            warmup_time = time.time() - start_time
            
            logger.info(f"✅ NeMo 모델 웜업 완료 ({warmup_time:.3f}초)")
            
        except Exception as e:
            logger.warning(f"⚠️ 모델 웜업 실패 (비중요): {e}")
    
    async def transcribe(self, audio_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """NeMo 모델을 사용한 음성 인식 (다중 후보 처리)"""
        try:
            logger.info("🎤 NeMo STT 전사 시작")
            
            # 오디오 전처리
            processed_audio = self._preprocess_audio(audio_data)
            
            # 기본 전사 수행
            logger.info("🔄 기본 전사 수행 중...")
            results = self.model.transcribe([processed_audio])
            
            # 여러 후보를 생성하는 추가 전사 (빔 서치 활용)
            try:
                logger.info("🔄 다중 후보 생성을 위한 추가 전사...")
                
                # 현재 빔 크기 확인
                current_beam_size = 1
                if hasattr(self.model, 'cfg') and hasattr(self.model.cfg, 'decoding'):
                    decoding = self.model.cfg.decoding
                    if hasattr(decoding, 'beam') and hasattr(decoding.beam, 'beam_size'):
                        current_beam_size = decoding.beam.beam_size
                
                # 더 많은 후보를 위해 빔 크기 임시 증가
                if current_beam_size < 6:  # 8 → 6으로 감소
                    logger.info(f"🔧 빔 크기 임시 증가: {current_beam_size} → 6")  # 8 → 6
                    
                    # 원본 설정 백업
                    original_beam_size = current_beam_size
                    
                    # 빔 크기 증가
                    if hasattr(decoding, 'beam'):
                        decoding.beam.beam_size = 6  # 8 → 6으로 감소
                        decoding.beam.return_best_hypothesis = True
                    
                    # 다시 전사
                    additional_results = self.model.transcribe([processed_audio])
                    
                    # 원본 설정 복원
                    decoding.beam.beam_size = original_beam_size
                    
                    # 결과 병합
                    if additional_results and len(additional_results) > 0:
                        logger.info(f"📝 추가 후보 {len(additional_results)}개 생성")
                        # 기본 결과와 추가 결과를 함께 고려
                        all_candidates = []
                        
                        # 기본 결과 추가
                        base_text = self._extract_text_from_result(results)
                        if base_text.strip():
                            all_candidates.append({
                                'text': base_text.strip(),
                                'confidence': self._calculate_text_confidence(base_text),
                                'source': 'base'
                            })
                        
                        # 추가 결과들 추가
                        for i, add_result in enumerate(additional_results):
                            add_text = self._extract_text_from_result(add_result)
                            if add_text.strip() and add_text.strip() != base_text.strip():
                                all_candidates.append({
                                    'text': add_text.strip(),
                                    'confidence': self._calculate_text_confidence(add_text),
                                    'source': f'beam_{i}'
                                })
                        
                        # 최고 신뢰도 후보 선택
                        if all_candidates:
                            best_candidate = max(all_candidates, key=lambda x: x['confidence'])
                            logger.info(f"🏆 최적 후보 선택: '{best_candidate['text']}' (신뢰도: {best_candidate['confidence']:.3f}, 출처: {best_candidate['source']})")
                            
                            # 다른 후보들도 로깅
                            for i, candidate in enumerate(all_candidates):
                                if candidate != best_candidate:
                                    logger.info(f"   후보 {i+1}: '{candidate['text']}' (신뢰도: {candidate['confidence']:.3f})")
                            
                            # 최적 결과로 대체
                            results = [best_candidate['text']]
                
            except Exception as e:
                logger.warning(f"⚠️ 다중 후보 처리 실패, 기본 결과 사용: {e}")
            
            # 최종 텍스트 추출
            text = self._extract_text_from_result(results)
            
            if not text.strip():
                logger.warning("⚠️ 전사 결과가 비어있음")
                return {
                    'text': '',
                    'confidence': 0.0,
                    'segments': [],
                    'model': self.model_name
                }
            
            # 결과 준비
            audio_duration = len(audio_data) / 16000.0
            confidence = self._calculate_confidence(text, audio_duration)
            segments = self._create_segments(text, audio_duration)
            
            logger.info(f"✅ NeMo 전사 완료: '{text[:100]}{'...' if len(text) > 100 else ''}' (신뢰도: {confidence:.3f})")
            
            return {
                'text': text,
                'confidence': confidence,
                'segments': segments,
                'model': self.model_name,
                'duration': audio_duration
            }
            
        except Exception as e:
            logger.error(f"❌ NeMo STT 전사 실패: {e}")
            import traceback
            logger.error(f"상세 에러: {traceback.format_exc()}")
            
            return {
                'text': '',
                'confidence': 0.0,
                'segments': [],
                'model': self.model_name,
                'error': str(e)
            }
    
    async def transcribe_audio(self, audio_data: str, audio_format: str = "pcm_16khz", language: str = "ko", **kwargs) -> STTResult:
        """오디오 전사 (기본 진입점) - 길이에 따라 단일/청크 처리 자동 선택"""
        print("🚨🚨🚨 NeMo transcribe_audio 함수 시작! 🚨🚨🚨")
        logger.info("🚨🚨🚨 NeMo transcribe_audio 함수 시작! 🚨🚨🚨")
        try:
            import base64
            import numpy as np
            import tempfile
            import wave
            import os
            
            logger.info("=" * 100)
            logger.info("🎯 NeMo transcribe_audio 함수 호출됨")
            logger.info("=" * 100)
            logger.info(f"📥 입력 파라미터:")
            logger.info(f"   • audio_format: {audio_format}")
            logger.info(f"   • language: {language}")
            logger.info(f"   • audio_data 길이: {len(audio_data)} chars")
            
            # base64 디코딩 및 오디오 데이터 변환
            if audio_format == "pcm_16khz":
                audio_bytes = base64.b64decode(audio_data)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                logger.info(f"🔄 PCM 16kHz 디코딩 완료: {len(audio_array)} 샘플")
            elif audio_format == "wav":
                # WAV 파일 처리
                import tempfile
                import wave
                import os
                
                print(f"🚨 WAV 처리 시작! audio_data 길이: {len(audio_data)}")
                logger.info(f"🔄 WAV 처리 시작 - audio_data 길이: {len(audio_data)} chars")
                
                try:
                    audio_bytes = base64.b64decode(audio_data)
                    print(f"🚨 base64 디코딩 완료! audio_bytes 길이: {len(audio_bytes)}")
                    logger.info(f"🔄 base64 디코딩 완료 - audio_bytes 길이: {len(audio_bytes)} bytes")
                except Exception as e:
                    print(f"🚨 base64 디코딩 실패: {e}")
                    logger.error(f"❌ base64 디코딩 실패: {e}")
                    raise
                
                # 임시 WAV 파일 생성
                try:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                        temp_wav.write(audio_bytes)
                        temp_wav_path = temp_wav.name
                    print(f"🚨 임시 WAV 파일 생성: {temp_wav_path}")
                    logger.info(f"🔄 임시 WAV 파일 생성: {temp_wav_path}")
                except Exception as e:
                    print(f"🚨 임시 파일 생성 실패: {e}")
                    logger.error(f"❌ 임시 파일 생성 실패: {e}")
                    raise
                
                try:
                    # WAV 파일을 numpy 배열로 읽기
                    print("🚨 WAV 파일 읽기 시작...")
                    with wave.open(temp_wav_path, 'rb') as wav_file:
                        frames = wav_file.readframes(wav_file.getnframes())
                        audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        sample_rate = wav_file.getframerate()
                        
                        print(f"🚨 WAV 파일 읽기 완료! sample_rate: {sample_rate}, 샘플 수: {len(audio_array)}")
                        logger.info(f"🔄 WAV 파일 읽기 완료 - sample_rate: {sample_rate}, 샘플 수: {len(audio_array)}")
                        
                        # 16kHz로 리샘플링이 필요한 경우
                        if sample_rate != 16000:
                            print(f"🚨 리샘플링 시작: {sample_rate} -> 16000")
                            import librosa
                            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                            print(f"🚨 리샘플링 완료! 새 샘플 수: {len(audio_array)}")
                            logger.info(f"🔄 {sample_rate}Hz -> 16kHz 리샘플링 완료")
                        else:
                            print("🚨 리샘플링 불필요 - 이미 16kHz")
                            
                except Exception as e:
                    print(f"🚨 WAV 파일 읽기 실패: {e}")
                    logger.error(f"❌ WAV 파일 읽기 실패: {e}")
                    raise
                finally:
                    # 임시 파일 정리
                    try:
                        if os.path.exists(temp_wav_path):
                            os.unlink(temp_wav_path)
                            print(f"🚨 임시 파일 정리 완료: {temp_wav_path}")
                    except Exception as e:
                        print(f"🚨 임시 파일 정리 실패: {e}")
                        logger.warning(f"⚠️ 임시 파일 정리 실패: {e}")
            else:
                raise ValueError(f"지원하지 않는 오디오 형식: {audio_format}")
            
            # 오디오 길이 계산 (초 단위)
            audio_duration = len(audio_array) / 16000.0
            print(f"🚨 오디오 길이 계산 완료: {audio_duration:.2f}초 (샘플 수: {len(audio_array)})")
            logger.info(f"🎧 오디오 길이: {audio_duration:.2f}초")
            
            # 20초 기준 분기점 로깅
            chunk_threshold = 20.0  # 20초 기준으로 변경
            print(f"🚨 청크 분할 임계값: {chunk_threshold}초")
            print(f"🚨 조건 체크: {audio_duration} >= {chunk_threshold} = {audio_duration >= chunk_threshold}")
            logger.info(f"🔍 청크 분할 임계값: {chunk_threshold}초")
            logger.info(f"🔍 오디오 길이 >= 임계값? {audio_duration} >= {chunk_threshold} = {audio_duration >= chunk_threshold}")
            
            if audio_duration >= chunk_threshold:
                print(f"🚨 청크 분할 처리로 진입! ({audio_duration:.1f}초)")
                logger.info(f"📦 20초 이상 오디오 감지 ({audio_duration:.1f}초) - VAD 기반 청크 분할 처리 (10초 제한)")
                return await self._transcribe_with_chunks(audio_array, audio_duration, 10.0)
            else:
                # 20초 미만 오디오는 VAD 없이 전체 처리
                print(f"🚨 단일 처리로 진입! ({audio_duration:.1f}초)")
                logger.info(f"🚀 20초 미만 오디오 ({audio_duration:.1f}초) - VAD 없이 전체 처리")
                return await self._transcribe_single(audio_array, audio_duration)
                
        except Exception as e:
            print(f"🚨🚨🚨 NeMo transcribe_audio 에러: {e}")
            logger.error(f"❌ NeMo transcribe_audio 실패: {e}")
            import traceback
            logger.error(f"❌ 상세 에러: {traceback.format_exc()}")
            raise ValueError(f"NeMo 전사 중 오류: {str(e)}")
    
    async def _transcribe_single(self, audio_array: np.ndarray, audio_duration: float) -> STTResult:
        """단일 오디오 처리 (VAD 없이 전체 오디오를 한 번에 처리)"""
        try:
            start_time = time.time()
            
            logger.info("=" * 60)
            logger.info("🚀 VAD 없이 전체 오디오 단일 처리 시작")
            logger.info("=" * 60)
            logger.info(f"📏 전체 오디오 길이: {audio_duration:.2f}초")
            logger.info(f"🎯 처리 방식: VAD 분할 없이 전체 오디오 한 번에 전사")
            logger.info(f"💡 이 방식으로 처리하면 앞부분 손실 문제가 없어야 합니다")
            
            # NeMo 모델로 전사 (간단한 접근법)
            logger.info("🤖 NeMo 모델 전사 중...")
            
            # torch dynamo 에러 억제
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            
            try:
                # 임시 WAV 파일로 저장해서 transcribe
                import tempfile
                import soundfile as sf
                import os
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # WAV 파일로 저장
                sf.write(temp_path, audio_array, 16000, subtype='PCM_16')
                
                try:
                    # 한국어 설정 시도
                    try:
                        if hasattr(self.model, 'cfg') and hasattr(self.model.cfg, 'language'):
                            logger.info("🇰🇷 단일 처리 - 모델에 한국어 설정 적용 중...")
                            original_language = getattr(self.model.cfg, 'language', None)
                            self.model.cfg.language = 'ko'
                    except Exception as e:
                        logger.warning(f"⚠️ 단일 처리 - 언어 설정 적용 실패: {e}")
                    
                    # 파일 경로를 사용해서 transcribe
                    result = self.model.transcribe([temp_path])
                finally:
                    # 임시 파일 삭제
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            except Exception as file_error:
                logger.warning(f"⚠️ 파일 기반 전사 실패: {file_error}")
                # 빈 결과 반환
                result = [""]
            
            # 디버깅: 원시 결과 로깅
            logger.info(f"🔍 NeMo 원시 결과 타입: {type(result)}")
            if hasattr(result, '__len__'):
                logger.info(f"🔍 결과 길이: {len(result)}")
            if isinstance(result, (list, tuple)) and len(result) > 0:
                logger.info(f"🔍 첫 번째 요소 타입: {type(result[0])}")
                if hasattr(result[0], '__len__') and len(str(result[0])) < 200:
                    logger.info(f"🔍 첫 번째 요소 내용: {result[0]}")
            
            # 결과 추출
            text = self._extract_text_from_result(result)
            processing_time = time.time() - start_time
            rtf = processing_time / audio_duration if audio_duration > 0 else 0.0
            
            logger.info("-" * 60)
            logger.info("🔍 단일 처리 결과 분석:")
            logger.info(f"   📝 전체 전사 텍스트: '{text}'")
            logger.info(f"   📏 텍스트 길이: {len(text)}자")
            logger.info(f"   ⏱️ 처리 시간: {processing_time:.2f}초")
            logger.info(f"   🚀 RTF: {rtf:.3f}")
            
            # "코로나" 키워드 체크
            if "코로나" in text:
                logger.info("   ✅ '코로나' 키워드 발견! - VAD 없이 정상 처리됨")
            else:
                logger.warning("   ⚠️ '코로나' 키워드 누락 - 모델 자체 문제일 가능성")
            
            # 시작 단어들 분석
            if text.strip():
                words = text.strip().split()
                first_words = words[:10]  # 처음 10단어
                logger.info(f"   📝 시작 10단어: {' '.join(first_words)}")
                
                # 첫 번째 문장 추출
                first_sentence = text.split('.')[0] if '.' in text else text[:50]
                logger.info(f"   📄 첫 문장: '{first_sentence}{'...' if len(text) > 50 and '.' not in text[:50] else ''}'")
            
            logger.info("=" * 60)
            
            logger.info(f"✅ NeMo 단일 전사 완료: {len(text)}자, 처리 시간: {processing_time:.2f}초")
            
            return STTResult(
                text=text,
                language="ko",  # 기본 언어
                confidence=0.95,  # NeMo는 일반적으로 높은 신뢰도
                rtf=rtf,  # Real-time factor
                audio_duration=audio_duration,
                segments=self._create_segments(text, audio_duration)
            )
            
        except Exception as e:
            logger.error(f"❌ NeMo 단일 전사 실패: {e}")
            raise
    
    async def _transcribe_with_chunks(self, audio_array: np.ndarray, audio_duration: float, chunk_duration: float) -> STTResult:
        """VAD 기반 청크 단위로 오디오 분할하여 처리"""
        print("🚨🚨🚨 _transcribe_with_chunks 함수 시작!")
        print(f"🚨 입력 파라미터 - audio_duration: {audio_duration:.2f}초, chunk_duration: {chunk_duration:.2f}초")
        logger.info("🚨🚨🚨 _transcribe_with_chunks 함수 시작!")
        
        try:
            start_time = time.time()
            print(f"🚨 VAD 기반 청크 분할 시작 (총 {audio_duration:.1f}초)")
            logger.info(f"📦 VAD 기반 청크 분할 시작 (총 {audio_duration:.1f}초)")
            
            # VAD 기반 음성 구간 감지
            print("🚨 VAD 음성 구간 감지 시작...")
            voice_segments = await self._detect_voice_segments(audio_array)
            print(f"🚨 VAD 음성 구간 감지 완료! 감지된 구간 수: {len(voice_segments) if voice_segments else 0}")
            
            if not voice_segments:
                print("🚨 음성 구간이 감지되지 않음!")
                logger.warning("⚠️ 음성 구간이 감지되지 않았습니다")
                return STTResult(
                    text="",
                    language="ko",
                    confidence=0.0,
                    rtf=0.0,
                    audio_duration=audio_duration,
                    segments=[]
                )
            
            logger.info(f"🎙️ {len(voice_segments)}개 음성 구간 감지됨")
            
            # 음성 구간을 적절한 크기의 청크로 그룹화
            print(f"🚨 청크 그룹화 시작! 음성 구간 수: {len(voice_segments)}개")
            chunks = []
            current_chunk_segments = []
            current_chunk_duration = 0.0
            is_first_chunk = True
            
            print(f"🚨 chunk_duration 값: {chunk_duration}")
            print(f"🚨 voice_segments 첫 번째 구간: {voice_segments[0] if voice_segments else 'None'}")
            
            for i, segment in enumerate(voice_segments):
                print(f"🚨 구간 {i+1} 처리 시작: {segment}")
                
                segment_duration = segment['duration']
                segment_start = segment['start_time']
                segment_end = segment['end_time']
                
                print(f"🚨 구간 {i+1} - start: {segment_start:.2f}s, end: {segment_end:.2f}s, duration: {segment_duration:.2f}s")
                logger.info(f"🔄 구간 {i+1} 처리 중: {segment_start:.2f}s~{segment_end:.2f}s (길이: {segment_duration:.2f}s)")
                
                # 단일 세그먼트가 10초를 넘는 경우 강제로 분할
                if segment_duration > chunk_duration:
                    logger.warning(f"⚠️ 구간 {i+1}이 {chunk_duration}초를 초과 ({segment_duration:.2f}s)! 강제 분할 필요")
                    
                    # 현재 청크 먼저 처리
                    if current_chunk_segments:
                        logger.info(f"   📦 현재 청크 {len(chunks)+1} 완료 (구간 {len(current_chunk_segments)}개, 총 {current_chunk_duration:.2f}s)")
                        chunk_info = self._create_chunk_from_segments(current_chunk_segments, audio_array, is_first_chunk)
                        chunks.append(chunk_info)
                        is_first_chunk = False
                        current_chunk_segments = []
                        current_chunk_duration = 0.0
                    
                    # 긴 세그먼트를 10초 단위로 분할
                    segment_chunks = self._split_long_segment(segment, chunk_duration, audio_array, is_first_chunk)
                    for seg_chunk in segment_chunks:
                        chunks.append(seg_chunk)
                        is_first_chunk = False
                        logger.info(f"   📦 분할된 청크 {len(chunks)} 추가: {seg_chunk['start_time']:.2f}s~{seg_chunk['end_time']:.2f}s (길이: {seg_chunk['duration']:.2f}s)")
                    
                    continue
                
                # 패딩을 고려한 예상 청크 길이 계산
                estimated_padding = 0.7 if not current_chunk_segments else 0.3  # 첫 번째 청크 여부에 따라
                estimated_chunk_duration = current_chunk_duration + segment_duration + estimated_padding
                
                # 현재 청크에 추가할 수 있는지 확인 (패딩 고려)
                if estimated_chunk_duration <= chunk_duration:
                    current_chunk_segments.append(segment)
                    current_chunk_duration += segment_duration
                    logger.info(f"   ✅ 현재 청크에 추가 (누적 길이: {current_chunk_duration:.2f}s, 패딩 포함 예상: {estimated_chunk_duration:.2f}s)")
                else:
                    # 현재 청크 완료
                    if current_chunk_segments:
                        logger.info(f"   📦 청크 {len(chunks)+1} 완료 (구간 {len(current_chunk_segments)}개, 총 {current_chunk_duration:.2f}s)")
                        chunk_info = self._create_chunk_from_segments(current_chunk_segments, audio_array, is_first_chunk)
                        chunks.append(chunk_info)
                        is_first_chunk = False
                    
                    # 새 청크 시작
                    current_chunk_segments = [segment]
                    current_chunk_duration = segment_duration
                    logger.info(f"   🆕 새 청크 {len(chunks)+1} 시작 (길이: {current_chunk_duration:.2f}s)")
            
            # 마지막 청크 처리
            if current_chunk_segments:
                print(f"🚨 마지막 청크 처리 중...")
                logger.info(f"   📦 마지막 청크 {len(chunks)+1} 완료 (구간 {len(current_chunk_segments)}개, 총 {current_chunk_duration:.2f}s)")
                chunk_info = self._create_chunk_from_segments(current_chunk_segments, audio_array, is_first_chunk)
                chunks.append(chunk_info)
            
            print(f"🚨 총 생성된 청크 수: {len(chunks)}개")
            print(f"🚨 청크 전사 처리 시작...")
            
            all_texts = []
            total_confidence = 0.0
            
            for i, chunk_info in enumerate(chunks):
                print(f"🚨 청크 {i+1}/{len(chunks)} 전사 시작!")
                
                chunk_audio = chunk_info['audio']
                chunk_start = chunk_info['start_time']
                chunk_end = chunk_info['end_time']
                chunk_duration = chunk_info['duration']
                
                print(f"🚨 청크 {i+1} 정보 - start: {chunk_start:.1f}s, end: {chunk_end:.1f}s, duration: {chunk_duration:.2f}s")
                logger.info(f"🔄 청크 {i+1}/{len(chunks)} 처리 중 ({chunk_start:.1f}s-{chunk_end:.1f}s, 길이: {chunk_duration:.2f}s)")

                
                try:
                    # GPU 메모리 정리
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 청크 전사 (파일 기반)
                    try:
                        import tempfile
                        import soundfile as sf
                        import os
                        from pathlib import Path
                        
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                            temp_path = temp_file.name
                        
                        # WAV 파일로 저장
                        sf.write(temp_path, chunk_audio, 16000, subtype='PCM_16')
                        
                        try:
                            # 파일 경로를 사용해서 transcribe
                            chunk_result = self.model.transcribe([temp_path])
                        finally:
                            # 임시 파일 삭제
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                                
                    except Exception as chunk_error:
                        logger.warning(f"⚠️ 청크 {i+1} 전사 실패: {chunk_error}")
                        chunk_result = [""]
                    
                    chunk_text = self._extract_text_from_result(chunk_result)
                    
                    # 🎯 청크별 텍스트 즉시 출력 (콘솔)
                    print("=" * 80)
                    print(f"🎯 청크 {i+1}/{len(chunks)} 전사 완료!")
                    print(f"📍 시간: {chunk_start:.2f}s ~ {chunk_end:.2f}s ({chunk_duration:.2f}초)")
                    if chunk_text.strip():
                        print(f"📝 텍스트: '{chunk_text}'")
                        print(f"📊 길이: {len(chunk_text)}자, 단어 수: {len(chunk_text.split())}개")
                        if "코로나" in chunk_text:
                            print(f"   ✅ '코로나' 키워드 발견!")
                    else:
                        print(f"⚪ 무음 구간 - 텍스트 없음")
                    print("=" * 80)
                    
                    # 🔍 청크별 텍스트 결과 상세 디버깅
                    logger.info("=" * 80)
                    logger.info(f"🔍 청크 {i+1}/{len(chunks)} 전사 결과 상세 분석")
                    logger.info("=" * 80)
                    logger.info(f"📍 청크 정보:")
                    logger.info(f"   • 시간 범위: {chunk_start:.2f}s ~ {chunk_end:.2f}s")
                    logger.info(f"   • 청크 길이: {chunk_duration:.2f}초")
                    logger.info(f"   • 오디오 샘플: {len(chunk_audio):,}개")
                    
                    # 원시 결과 분석
                    logger.info(f"🤖 NeMo 원시 결과:")
                    logger.info(f"   • 결과 타입: {type(chunk_result)}")
                    if hasattr(chunk_result, '__len__'):
                        logger.info(f"   • 결과 길이: {len(chunk_result)}")
                    if isinstance(chunk_result, (list, tuple)) and len(chunk_result) > 0:
                        logger.info(f"   • 첫 번째 요소 타입: {type(chunk_result[0])}")
                        logger.info(f"   • 첫 번째 요소: {chunk_result[0]}")
                    
                    # 추출된 텍스트 분석
                    logger.info(f"📝 추출된 텍스트:")
                    logger.info(f"   • 텍스트 길이: {len(chunk_text)}자")
                    logger.info(f"   • 빈 텍스트 여부: {'예' if not chunk_text.strip() else '아니오'}")
                    
                    if chunk_text.strip():
                        logger.info(f"   • 전체 텍스트: '{chunk_text}'")
                        
                        # 단어 분석
                        words = chunk_text.strip().split()
                        logger.info(f"   • 단어 수: {len(words)}개")
                        if len(words) > 0:
                            logger.info(f"   • 첫 번째 단어: '{words[0]}'")
                            logger.info(f"   • 마지막 단어: '{words[-1]}'")
                        
                        # 특정 키워드 체크
                        if "코로나" in chunk_text:
                            logger.info(f"   ✅ '코로나' 키워드 발견!")
                        
                        # 문장 분석
                        sentences = [s.strip() for s in chunk_text.split('.') if s.strip()]
                        logger.info(f"   • 문장 수: {len(sentences)}개")
                        for j, sentence in enumerate(sentences[:3]):  # 처음 3문장만
                            logger.info(f"     문장 {j+1}: '{sentence}'")
                    else:
                        logger.info(f"   ⚪ 무음 구간 - 텍스트 없음")
                    
                    logger.info("=" * 80)
                    
                    if chunk_text.strip():
                        all_texts.append(chunk_text.strip())
                        total_confidence += 0.95  # 각 청크의 기본 신뢰도
                        logger.info(f"✅ 청크 {i+1} 처리 완료: {len(chunk_text)}자 추가됨")
                    else:
                        logger.info(f"⚪ 청크 {i+1}: 무음 구간으로 건너뜀")
                
                except Exception as chunk_error:
                    import traceback
                    logger.warning(f"⚠️ 청크 {i+1} 처리 실패: {chunk_error}")
                    logger.warning(f"⚠️ 청크 {i+1} 에러 상세: {traceback.format_exc()}")
                    # 청크 실패 시 건너뛰고 계속 진행
                    continue
            
            # 최종 결과 조합
            final_text = " ".join(all_texts)
            avg_confidence = total_confidence / max(len(all_texts), 1) if all_texts else 0.0
            processing_time = time.time() - start_time
            rtf = processing_time / audio_duration if audio_duration > 0 else 0.0
            
            # 🎯 최종 결과 즉시 출력 (콘솔)
            print("=" * 100)
            print("🎯 청크 전사 완료! 최종 결과 요약")
            print("=" * 100)
            print(f"📊 처리 통계: {len(chunks)}개 청크, {len(all_texts)}개 텍스트, {processing_time:.2f}초")
            print(f"📝 개별 청크 결과:")
            for i, text in enumerate(all_texts):
                print(f"   청크 {i+1}: '{text}'")
            print(f"🔗 최종 병합 텍스트: '{final_text}'")
            print(f"📊 총 {len(final_text)}자, {len(final_text.split())}개 단어")
            if "코로나" in final_text:
                print(f"✅ '코로나' 키워드 최종 확인됨!")
            else:
                print(f"⚠️ '코로나' 키워드 최종 결과에서 누락")
            print("=" * 100)
            
            # 🔍 최종 결과 상세 분석
            logger.info("=" * 100)
            logger.info("🎯 최종 전사 결과 종합 분석")
            logger.info("=" * 100)
            logger.info(f"📊 처리 통계:")
            logger.info(f"   • 처리된 청크 수: {len(chunks)}개")
            logger.info(f"   • 텍스트가 있는 청크: {len(all_texts)}개")
            logger.info(f"   • 무음 청크: {len(chunks) - len(all_texts)}개")
            logger.info(f"   • 총 처리 시간: {processing_time:.2f}초")
            logger.info(f"   • RTF (Real-time Factor): {rtf:.3f}")
            logger.info(f"   • 평균 신뢰도: {avg_confidence:.3f}")
            
            logger.info(f"📝 개별 청크 텍스트:")
            for i, text in enumerate(all_texts):
                logger.info(f"   청크 {i+1}: '{text}'")
            
            logger.info(f"🔗 최종 병합 텍스트:")
            logger.info(f"   • 총 길이: {len(final_text)}자")
            logger.info(f"   • 단어 수: {len(final_text.split())}개")
            logger.info(f"   • 전체 텍스트: '{final_text}'")
            
            # 특정 키워드 최종 체크
            if "코로나" in final_text:
                logger.info(f"   ✅ 최종 결과에서 '코로나' 키워드 확인됨!")
            else:
                logger.info(f"   ⚠️ 최종 결과에서 '코로나' 키워드 누락")
            
            logger.info("=" * 100)
            
            logger.info(f"✅ VAD 기반 청크 처리 완료: 총 {len(final_text)}자, 처리 시간: {processing_time:.2f}초")
            
            return STTResult(
                text=final_text,
                language="ko",  # 기본 언어
                confidence=avg_confidence,
                rtf=rtf,  # Real-time factor
                audio_duration=audio_duration,
                segments=self._create_segments(final_text, audio_duration)
            )
            
        except Exception as e:
            print(f"🚨🚨🚨 _transcribe_with_chunks 에러: {e}")
            logger.error(f"❌ NeMo VAD 청크 전사 실패: {e}")
            import traceback
            print(f"🚨 상세 에러: {traceback.format_exc()}")
            logger.error(f"❌ 상세 에러: {traceback.format_exc()}")
            raise
    
    async def _detect_voice_segments(self, audio_array: np.ndarray) -> List[Dict]:
        """VAD를 사용하여 음성 구간 감지"""
        try:
            # Silero VAD 사용
            logger.info("🎙️ Silero VAD로 음성 구간 감지 중...")
            
            try:
                import torch
                
                # Silero VAD 모델 로드 (한 번만 로드)
                if not hasattr(self, '_vad_model'):
                    self._vad_model, utils = torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=False,
                        onnx=False
                    )
                    self._vad_get_speech_timestamps = utils[0]
                    logger.info("📦 Silero VAD 모델 로드 완료")
                
                # 오디오를 torch tensor로 변환
                audio_tensor = torch.tensor(audio_array).float()
                
                # 음성 구간 감지 (16kHz 가정) - 보수적 설정으로 조정
                speech_timestamps = self._vad_get_speech_timestamps(
                    audio_tensor, 
                    self._vad_model,
                    sampling_rate=16000,
                    threshold=0.4,  # 음성 감지 임계값 (0.2 → 0.4로 증가, 덜 민감)
                    min_speech_duration_ms=200,  # 최소 음성 길이 유지
                    min_silence_duration_ms=100,  # 최소 무음 길이 유지
                    window_size_samples=512,  # 윈도우 크기 유지
                    speech_pad_ms=100  # 음성 구간 패딩 (300ms → 100ms로 감소)
                )
                
                # 결과를 초 단위로 변환
                voice_segments = []
                for segment in speech_timestamps:
                    start_sample = segment['start']
                    end_sample = segment['end']
                    start_time = start_sample / 16000.0
                    end_time = end_sample / 16000.0
                    
                    voice_segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_sample': start_sample,
                        'end_sample': end_sample,
                        'duration': end_time - start_time
                    })
                
                logger.info(f"🎙️ Silero VAD: {len(voice_segments)}개 음성 구간 감지")
                
                # 🔍 VAD 구간 상세 디버깅
                logger.info("=" * 60)
                logger.info("🔍 VAD 음성 구간 상세 분석")
                logger.info("=" * 60)
                total_speech_duration = 0.0
                for i, segment in enumerate(voice_segments):
                    start_time = segment['start_time']
                    end_time = segment['end_time']
                    duration = segment['duration']
                    total_speech_duration += duration
                    
                    logger.info(f"📍 구간 {i+1:2d}: {start_time:6.2f}s ~ {end_time:6.2f}s (길이: {duration:5.2f}s)")
                    
                    # 구간 내 오디오 샘플 미리보기 (처음 몇 글자만)
                    start_sample = int(start_time * 16000)
                    end_sample = int(end_time * 16000)
                    segment_audio = audio_array[start_sample:end_sample]
                    
                    # 에너지 레벨 계산
                    energy = np.mean(segment_audio ** 2) if len(segment_audio) > 0 else 0.0
                    max_amplitude = np.max(np.abs(segment_audio)) if len(segment_audio) > 0 else 0.0
                    
                    logger.info(f"   🔊 에너지: {energy:.6f}, 최대 진폭: {max_amplitude:.3f}")
                
                logger.info("-" * 60)
                logger.info(f"📊 VAD 요약:")
                logger.info(f"   • 전체 오디오 길이: {len(audio_array) / 16000.0:.2f}초")
                logger.info(f"   • 총 음성 구간: {len(voice_segments)}개")
                logger.info(f"   • 총 음성 길이: {total_speech_duration:.2f}초")
                logger.info(f"   • 음성 비율: {total_speech_duration / (len(audio_array) / 16000.0) * 100:.1f}%")
                
                # 무음 구간 분석
                if len(voice_segments) > 1:
                    logger.info(f"🔇 무음 구간 분석:")
                    for i in range(len(voice_segments) - 1):
                        silence_start = voice_segments[i]['end_time']
                        silence_end = voice_segments[i + 1]['start_time']
                        silence_duration = silence_end - silence_start
                        logger.info(f"   무음 {i+1}: {silence_start:.2f}s ~ {silence_end:.2f}s (길이: {silence_duration:.2f}s)")
                
                logger.info("=" * 60)
                
                return voice_segments
                
            except Exception as silero_error:
                logger.warning(f"⚠️ Silero VAD 실패, 간단한 에너지 기반 VAD 사용: {silero_error}")
                return self._simple_energy_vad(audio_array)
                
        except Exception as e:
            logger.error(f"❌ VAD 처리 실패: {e}")
            # VAD 실패 시 전체 오디오를 하나의 세그먼트로 처리
            return [{
                'start_time': 0.0,
                'end_time': len(audio_array) / 16000.0,
                'start_sample': 0,
                'end_sample': len(audio_array),
                'duration': len(audio_array) / 16000.0
            }]
    
    def _simple_energy_vad(self, audio_array: np.ndarray) -> List[Dict]:
        """간단한 에너지 기반 VAD (Silero VAD 실패 시 대안)"""
        try:
            # 프레임 단위로 에너지 계산
            frame_length = int(0.025 * 16000)  # 25ms 프레임
            frame_step = int(0.010 * 16000)    # 10ms 스텝
            
            frames = []
            for i in range(0, len(audio_array) - frame_length, frame_step):
                frame = audio_array[i:i + frame_length]
                energy = np.sum(frame ** 2)
                frames.append(energy)
            
            # 에너지 임계값 설정 (적절한 민감도로 조정)
            threshold = np.percentile(frames, 90) * 0.05  # 85% → 90%, 1% → 5%로 조정
            
            # 음성 구간 감지
            voice_segments = []
            in_speech = False
            start_frame = 0
            
            for i, energy in enumerate(frames):
                if energy > threshold and not in_speech:
                    # 음성 시작
                    in_speech = True
                    start_frame = i
                elif energy <= threshold and in_speech:
                    # 음성 종료
                    in_speech = False
                    
                    start_sample = start_frame * frame_step
                    end_sample = i * frame_step
                    start_time = start_sample / 16000.0
                    end_time = end_sample / 16000.0
                    
                    # 최소 길이 체크 (0.1초 이상으로 더 완화)
                    if end_time - start_time >= 0.1:
                        voice_segments.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'start_sample': start_sample,
                            'end_sample': end_sample,
                            'duration': end_time - start_time
                        })
            
            # 마지막 구간 처리
            if in_speech:
                start_sample = start_frame * frame_step
                end_sample = len(audio_array)
                start_time = start_sample / 16000.0
                end_time = end_sample / 16000.0
                
                if end_time - start_time >= 0.1:
                    voice_segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_sample': start_sample,
                        'end_sample': end_sample,
                        'duration': end_time - start_time
                    })
            
            logger.info(f"🔊 에너지 기반 VAD: {len(voice_segments)}개 음성 구간 감지")
            return voice_segments
            
        except Exception as e:
            logger.error(f"❌ 에너지 기반 VAD 실패: {e}")
            # 최후의 수단: 전체 오디오를 하나의 세그먼트로 처리
            return [{
                'start_time': 0.0,
                'end_time': len(audio_array) / 16000.0,
                'start_sample': 0,
                'end_sample': len(audio_array),
                'duration': len(audio_array) / 16000.0
            }]
    
    def _group_segments_into_chunks(self, voice_segments: List[Dict], audio_array: np.ndarray, max_chunk_duration: float = 10.0) -> List[Dict]:
        """음성 구간들을 적절한 크기의 청크로 그룹화 (NeMo 최적화: 10초 제한)"""
        logger.info("=" * 60)
        logger.info("🔍 청크 그룹화 과정 상세 분석 (NeMo 10초 제한)")
        logger.info("=" * 60)
        logger.info(f"📦 최대 청크 길이: {max_chunk_duration}초 (NeMo 권장: ≤10초)")
        logger.info(f"📝 처리할 음성 구간: {len(voice_segments)}개")
        
        chunks = []
        current_chunk_segments = []
        current_chunk_duration = 0.0
        is_first_chunk = True
        
        for i, segment in enumerate(voice_segments):
            segment_duration = segment['duration']
            segment_start = segment['start_time']
            segment_end = segment['end_time']
            
            logger.info(f"🔄 구간 {i+1} 처리 중: {segment_start:.2f}s~{segment_end:.2f}s (길이: {segment_duration:.2f}s)")
            
            # 단일 세그먼트가 15초를 넘는 경우 강제로 분할
            if segment_duration > max_chunk_duration:
                logger.warning(f"⚠️ 구간 {i+1}이 {max_chunk_duration}초를 초과 ({segment_duration:.2f}s)! 강제 분할 필요")
                
                # 현재 청크 먼저 처리
                if current_chunk_segments:
                    logger.info(f"   📦 현재 청크 {len(chunks)+1} 완료 (구간 {len(current_chunk_segments)}개, 총 {current_chunk_duration:.2f}s)")
                    chunk_info = self._create_chunk_from_segments(current_chunk_segments, audio_array, is_first_chunk)
                    chunks.append(chunk_info)
                    is_first_chunk = False
                    current_chunk_segments = []
                    current_chunk_duration = 0.0
                
                # 긴 세그먼트를 15초 단위로 분할
                segment_chunks = self._split_long_segment(segment, max_chunk_duration, audio_array, is_first_chunk)
                for seg_chunk in segment_chunks:
                    chunks.append(seg_chunk)
                    is_first_chunk = False
                    logger.info(f"   📦 분할된 청크 {len(chunks)} 추가: {seg_chunk['start_time']:.2f}s~{seg_chunk['end_time']:.2f}s (길이: {seg_chunk['duration']:.2f}s)")
                
                continue
            
            # 패딩을 고려한 예상 청크 길이 계산
            estimated_padding = 0.7 if not current_chunk_segments else 0.3  # 첫 번째 청크 여부에 따라
            estimated_chunk_duration = current_chunk_duration + segment_duration + estimated_padding
            
            # 현재 청크에 추가할 수 있는지 확인 (패딩 고려)
            if estimated_chunk_duration <= max_chunk_duration:
                current_chunk_segments.append(segment)
                current_chunk_duration += segment_duration
                logger.info(f"   ✅ 현재 청크에 추가 (누적 길이: {current_chunk_duration:.2f}s, 패딩 포함 예상: {estimated_chunk_duration:.2f}s)")
            else:
                # 현재 청크 완료
                if current_chunk_segments:
                    logger.info(f"   📦 청크 {len(chunks)+1} 완료 (구간 {len(current_chunk_segments)}개, 총 {current_chunk_duration:.2f}s)")
                    chunk_info = self._create_chunk_from_segments(current_chunk_segments, audio_array, is_first_chunk)
                    chunks.append(chunk_info)
                    is_first_chunk = False
                
                # 새 청크 시작
                current_chunk_segments = [segment]
                current_chunk_duration = segment_duration
                logger.info(f"   🆕 새 청크 {len(chunks)+1} 시작 (길이: {current_chunk_duration:.2f}s)")
        
        # 마지막 청크 처리
        if current_chunk_segments:
            logger.info(f"   📦 마지막 청크 {len(chunks)+1} 완료 (구간 {len(current_chunk_segments)}개, 총 {current_chunk_duration:.2f}s)")
            chunk_info = self._create_chunk_from_segments(current_chunk_segments, audio_array, is_first_chunk)
            chunks.append(chunk_info)
        
        logger.info("-" * 60)
        logger.info(f"📊 청크 그룹화 요약:")
        logger.info(f"   • 생성된 청크 수: {len(chunks)}개")
        
        # 10초 초과 청크 체크
        over_limit_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_duration = chunk['duration']
            logger.info(f"   • 청크 {i+1}: {chunk['start_time']:.2f}s~{chunk['end_time']:.2f}s (길이: {chunk_duration:.2f}s)")
            
            if chunk_duration > 10.0:
                over_limit_chunks.append((i+1, chunk_duration))
                logger.warning(f"     ⚠️ 청크 {i+1}이 10초 초과! ({chunk_duration:.2f}s)")
        
        if over_limit_chunks:
            logger.error(f"❌ {len(over_limit_chunks)}개 청크가 10초를 초과했습니다!")
            for chunk_num, duration in over_limit_chunks:
                logger.error(f"   청크 {chunk_num}: {duration:.2f}s")
        else:
            logger.info(f"✅ 모든 청크가 10초 이하로 제한됨")
        
        logger.info("=" * 60)
        
        return chunks
    
    def _split_long_segment(self, segment: Dict, max_duration: float, audio_array: np.ndarray, is_first_chunk: bool) -> List[Dict]:
        """10초를 넘는 긴 세그먼트를 더 작은 청크로 분할"""
        logger.info(f"🔧 긴 세그먼트 분할 시작: {segment['start_time']:.2f}s~{segment['end_time']:.2f}s (길이: {segment['duration']:.2f}s)")
        
        segment_start = segment['start_time']
        segment_end = segment['end_time']
        segment_duration = segment['duration']
        
        # 분할할 청크 수 계산
        num_chunks = int(np.ceil(segment_duration / max_duration))
        chunk_duration = segment_duration / num_chunks
        
        logger.info(f"   📊 분할 계획: {num_chunks}개 청크, 각 청크 약 {chunk_duration:.2f}초")
        
        chunks = []
        current_first_chunk = is_first_chunk
        
        for i in range(num_chunks):
            # 청크 시간 범위 계산
            chunk_start_time = segment_start + (i * chunk_duration)
            chunk_end_time = min(segment_end, segment_start + ((i + 1) * chunk_duration))
            
            # 마지막 청크는 남은 부분을 모두 포함
            if i == num_chunks - 1:
                chunk_end_time = segment_end
            
            logger.info(f"   🔸 분할 청크 {i+1}: {chunk_start_time:.2f}s~{chunk_end_time:.2f}s")
            
            # 패딩 적용
            if current_first_chunk:
                # 첫 번째 청크는 0초부터 시작
                padded_start_time = 0.0
                actual_start_padding = chunk_start_time - padded_start_time
            else:
                # 나머지 청크는 0.3초 패딩
                padded_start_time = max(0.0, chunk_start_time - 0.3)
                actual_start_padding = chunk_start_time - padded_start_time
            
            # 끝 패딩 0.7초
            max_end_time = len(audio_array) / 16000.0
            padded_end_time = min(max_end_time, chunk_end_time + 0.7)
            actual_end_padding = padded_end_time - chunk_end_time
            
            # 샘플 인덱스 계산
            chunk_start_sample = int(padded_start_time * 16000)
            chunk_end_sample = int(padded_end_time * 16000)
            chunk_end_sample = min(chunk_end_sample, len(audio_array))
            
            # 오디오 추출
            chunk_audio = audio_array[chunk_start_sample:chunk_end_sample]
            final_duration = len(chunk_audio) / 16000.0
            
            # 청크 정보 생성
            chunk_info = {
                'audio': chunk_audio,
                'start_time': padded_start_time,
                'end_time': padded_end_time,
                'original_start_time': chunk_start_time,
                'original_end_time': chunk_end_time,
                'segments': [{
                    'start_time': chunk_start_time,
                    'end_time': chunk_end_time,
                    'duration': chunk_end_time - chunk_start_time,
                    'start_sample': int(chunk_start_time * 16000),
                    'end_sample': int(chunk_end_time * 16000)
                }],
                'duration': final_duration,
                'start_padding': actual_start_padding,
                'end_padding': actual_end_padding,
                'overlap_start': actual_start_padding,
                'overlap_end': actual_end_padding,
                'split_from_long_segment': True  # 분할된 청크임을 표시
            }
            
            chunks.append(chunk_info)
            current_first_chunk = False
            
            logger.info(f"     ✅ 분할 청크 {i+1} 생성: {padded_start_time:.2f}s~{padded_end_time:.2f}s (실제 길이: {final_duration:.2f}s)")
        
        logger.info(f"🔧 긴 세그먼트 분할 완료: {len(chunks)}개 청크 생성")
        return chunks
    
    def _create_chunk_from_segments(self, segments: List[Dict], audio_array: np.ndarray, is_first_chunk: bool = False) -> Dict:
        """세그먼트들로부터 청크 생성 (무음 구간 포함, 패딩 추가, 오버랩 처리)"""
        if not segments:
            return None
        
        logger.info("🔍 청크 생성 상세 과정:")
        logger.info(f"   📝 포함된 음성 구간: {len(segments)}개")
        
        # 청크의 시작과 끝 시간
        chunk_start_time = segments[0]['start_time']
        chunk_end_time = segments[-1]['end_time']
        original_chunk_duration = chunk_end_time - chunk_start_time
        
        logger.info(f"   📏 원본 청크 범위: {chunk_start_time:.2f}s~{chunk_end_time:.2f}s (길이: {original_chunk_duration:.2f}s)")
        
        # 패딩 및 오버랩 추가
        if is_first_chunk:
            # 첫 번째 청크는 항상 0초부터 시작 (시작 부분 손실 방지)
            padded_start_time = 0.0
            actual_start_padding = chunk_start_time - padded_start_time
            logger.info(f"   🎯 첫 번째 청크: 0초부터 시작 (패딩: {actual_start_padding:.2f}s)")
        else:
            # 나머지 청크는 앞쪽에 0.3초 패딩
            start_padding_seconds = 0.3
            padded_start_time = max(0.0, chunk_start_time - start_padding_seconds)
            actual_start_padding = chunk_start_time - padded_start_time
            logger.info(f"   ⏪ 시작 패딩: {actual_start_padding:.2f}s")
        
        # 끝 부분은 0.7초 패딩 + 오버랩 (적절한 컨텍스트)
        end_padding_seconds = 0.7  # 1초 → 0.7초로 감소
        max_end_time = len(audio_array) / 16000.0
        padded_end_time = min(max_end_time, chunk_end_time + end_padding_seconds)
        actual_end_padding = padded_end_time - chunk_end_time
        
        logger.info(f"   ⏩ 끝 패딩+오버랩: {chunk_end_time:.2f}s → {padded_end_time:.2f}s (패딩: {actual_end_padding:.2f}s)")
        
        # 샘플 인덱스 (패딩 포함)
        chunk_start_sample = int(padded_start_time * 16000)
        chunk_end_sample = int(padded_end_time * 16000)
        
        # 범위 체크
        chunk_end_sample = min(chunk_end_sample, len(audio_array))
        actual_end_time = chunk_end_sample / 16000.0
        
        logger.info(f"   🔢 샘플 인덱스: {chunk_start_sample} ~ {chunk_end_sample}")
        logger.info(f"   ⏱️ 최종 청크 시간: {padded_start_time:.2f}s~{actual_end_time:.2f}s")
        
        # 오디오 추출 (패딩 포함)
        chunk_audio = audio_array[chunk_start_sample:chunk_end_sample]
        final_duration = len(chunk_audio) / 16000.0
        
        logger.info(f"   🎵 추출된 오디오: {len(chunk_audio)}샘플 ({final_duration:.2f}초)")
        
        # 청크 내 무음 구간 분석
        if len(segments) > 1:
            logger.info(f"   🔇 청크 내 무음 구간:")
            for i in range(len(segments) - 1):
                silence_start = segments[i]['end_time']
                silence_end = segments[i + 1]['start_time']
                silence_duration = silence_end - silence_start
                logger.info(f"      무음 {i+1}: {silence_start:.2f}s~{silence_end:.2f}s (길이: {silence_duration:.2f}s)")
        
        # 앞부분 손실 위험 체크
        if chunk_start_time > 0.5:  # 첫 번째 음성이 0.5초 이후에 시작되면 경고
            logger.warning(f"⚠️ 앞부분 손실 위험! 첫 번째 음성이 {chunk_start_time:.2f}초에 시작")
            if not is_first_chunk:
                logger.warning(f"   이 청크는 첫 번째가 아니므로 {padded_start_time:.2f}초부터 시작")
        
        return {
            'audio': chunk_audio,
            'start_time': padded_start_time,
            'end_time': actual_end_time,
            'original_start_time': chunk_start_time,
            'original_end_time': chunk_end_time,
            'segments': segments,
            'duration': final_duration,
            'start_padding': actual_start_padding,
            'end_padding': actual_end_padding,
            'overlap_start': actual_start_padding,  # 오버랩 정보 추가
            'overlap_end': actual_end_padding       # 오버랩 정보 추가
        }
    
    def _extract_text_from_result(self, result) -> str:
        """NeMo 결과에서 텍스트 추출 (다양한 형태 지원, JSON 직렬화 안전)"""
        try:
            # None 체크
            if result is None:
                return ""
            
            # numpy.ndarray 처리 - 더 안전한 방식
            if hasattr(result, '__array__') and hasattr(result, 'dtype'):
                try:
                    # 배열을 Python 객체로 안전하게 변환
                    if hasattr(result, 'tolist'):
                        converted = result.tolist()
                        return self._extract_text_from_result(converted)
                    elif hasattr(result, 'item') and result.ndim == 0:
                        # 스칼라 배열
                        converted = result.item()
                        return str(converted).strip()
                    else:
                        # 마지막 수단: 문자열 변환
                        converted = str(result)
                        return converted.strip()
                except Exception as e:
                    logger.warning(f"⚠️ numpy 배열 처리 실패: {e}")
                    return ""
            
            # 문자열인 경우
            if isinstance(result, str):
                return result.strip()
            
            # 리스트인 경우
            elif isinstance(result, list):
                if len(result) == 0:
                    return ""
                
                # 첫 번째 요소가 문자열인 경우
                first_item = result[0]
                
                if isinstance(first_item, str):
                    return first_item.strip()
                
                # 첫 번째 요소가 딕셔너리인 경우
                elif isinstance(first_item, dict):
                    if 'text' in first_item:
                        return str(first_item['text']).strip()
                    elif 'transcription' in first_item:
                        return str(first_item['transcription']).strip()
                
                # 첫 번째 요소가 객체인 경우
                elif hasattr(first_item, 'text'):
                    return str(first_item.text).strip()
                
                # numpy 배열이나 직렬화 불가능한 객체인 경우 재귀 처리
                elif hasattr(first_item, '__array__'):
                    return self._extract_text_from_result(first_item)
                elif not self._is_json_serializable(first_item):
                    try:
                        result_str = str(first_item).strip()
                        return result_str
                    except:
                        return ""
                
                # 기타 경우 문자열 변환 시도
                else:
                    try:
                        result_str = str(first_item).strip()
                        return result_str
                    except:
                        return ""
            
            # 딕셔너리인 경우
            elif isinstance(result, dict):
                if 'text' in result:
                    return str(result['text']).strip()
                elif 'transcription' in result:
                    return str(result['transcription']).strip()
                elif 'result' in result:
                    return self._extract_text_from_result(result['result'])
            
            # 튜플인 경우
            elif isinstance(result, tuple):
                if len(result) > 0:
                    return self._extract_text_from_result(result[0])
                return ""
            
            # 객체인 경우 (text 속성 확인)
            elif hasattr(result, 'text'):
                return str(result.text).strip()
            
            # 직렬화 가능성 체크
            elif not self._is_json_serializable(result):
                try:
                    result_str = str(result).strip()
                    return result_str
                except:
                    return ""
            
            # 기타 경우 문자열 변환 시도
            else:
                try:
                    result_str = str(result).strip()
                    return result_str
                except:
                    return ""
                    
        except Exception as e:
            logger.error(f"❌ 텍스트 추출 실패: {e}")
            return ""
    
    def _is_json_serializable(self, obj) -> bool:
        """객체가 JSON 직렬화 가능한지 확인"""
        try:
            import json
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False

    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """오디오 데이터 전처리 (정규화, 노이즈 감소, 필터링)"""
        try:
            # 1. 기본 정규화
            if np.max(np.abs(audio_data)) > 0:
                # RMS 기반 정규화 (적절한 볼륨 조정)
                rms = np.sqrt(np.mean(audio_data ** 2))
                target_rms = 0.15  # 목표 RMS 레벨 (0.2 → 0.15로 감소)
                if rms > 0:
                    normalization_factor = target_rms / rms
                    audio_data = audio_data * normalization_factor
                    logger.debug(f"🔧 RMS 정규화: {rms:.4f} → {target_rms:.4f} (factor: {normalization_factor:.4f})")
                
                # 클리핑 방지
                max_val = np.max(np.abs(audio_data))
                if max_val > 0.9:  # 0.95 → 0.9로 감소
                    audio_data = audio_data * (0.9 / max_val)
                    logger.debug(f"🔧 클리핑 방지: 최대값 {max_val:.4f} → 0.9")
            
            # 2. 고주파 및 저주파 필터링 (optional, 더 보수적으로 조정)
            try:
                from scipy import signal
                
                # 16kHz 샘플링 레이트 가정
                fs = 16000
                
                # 고주파 노이즈 제거 (7.5kHz 이상 차단 → 7kHz로 낮춤)
                nyquist = fs / 2
                high_cutoff = 7000  # 7kHz
                sos_high = signal.butter(3, high_cutoff / nyquist, btype='low', output='sos')
                audio_data = signal.sosfilt(sos_high, audio_data)
                
                # 저주파 노이즈 제거 (100Hz 이하 차단 → 80Hz로 낮춤)
                low_cutoff = 80  # 80Hz
                sos_low = signal.butter(2, low_cutoff / nyquist, btype='high', output='sos')
                audio_data = signal.sosfilt(sos_low, audio_data)
                
                logger.debug(f"🔧 주파수 필터링 적용: {low_cutoff}Hz~{high_cutoff}Hz")
                
            except ImportError:
                logger.debug("📝 scipy 없음, 주파수 필터링 스킵")
            except Exception as e:
                logger.warning(f"⚠️ 주파수 필터링 실패: {e}")
            
            # 3. 동적 범위 압축 (소리가 작은 부분을 적절히 증폭)
            try:
                # 컴프레서 효과 (보수적 적용)
                threshold = 0.15  # 임계값 (0.1 → 0.15로 증가)
                ratio = 1.8      # 압축비 (2.0 → 1.8로 감소, 더 자연스럽게)
                
                # 신호의 절댓값 계산
                abs_audio = np.abs(audio_data)
                
                # 임계값 이상에서만 압축 적용
                compressed_mask = abs_audio > threshold
                if np.any(compressed_mask):
                    # 압축 적용
                    compressed_gain = threshold + (abs_audio[compressed_mask] - threshold) / ratio
                    compression_factor = compressed_gain / abs_audio[compressed_mask]
                    
                    # 원본 신호에 압축 적용 (부호 유지)
                    audio_data[compressed_mask] = audio_data[compressed_mask] * compression_factor
                    
                    logger.debug(f"🔧 동적 범위 압축 적용: 임계값 {threshold}, 비율 1:{ratio}")
                
            except Exception as e:
                logger.warning(f"⚠️ 동적 범위 압축 실패: {e}")
            
            # 4. 최종 정규화 (부드럽게)
            max_val = np.max(np.abs(audio_data))
            if max_val > 0.9:
                audio_data = audio_data * (0.9 / max_val)
                logger.debug(f"🔧 최종 정규화: {max_val:.4f} → 0.9")
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"⚠️ 오디오 전처리 실패: {e}")
            return audio_data.astype(np.float32)
    
    def _calculate_text_confidence(self, text: str) -> float:
        """텍스트 기반 신뢰도 계산 (단어 누락 방지를 위한 휴리스틱)"""
        if not text or not text.strip():
            return 0.0
        
        try:
            # 기본 점수
            base_score = 0.5
            
            # 1. 텍스트 길이 점수 (적절한 길이일수록 높은 점수)
            text_length = len(text.strip())
            if text_length > 5:
                length_score = min(0.3, text_length / 100.0)  # 최대 0.3점
            else:
                length_score = text_length * 0.02  # 짧은 텍스트는 낮은 점수
            
            # 2. 한국어 비율 점수
            korean_chars = sum(1 for char in text if '\uac00' <= char <= '\ud7af')
            if len(text) > 0:
                korean_ratio = korean_chars / len(text)
                korean_score = korean_ratio * 0.2  # 최대 0.2점
            else:
                korean_score = 0.0
            
            # 3. 완성도 점수 (문장 끝 표시, 문법적 완성도)
            completion_score = 0.0
            if text.endswith(('.', '!', '?', '다', '요', '습니다')):
                completion_score += 0.1
            
            # 단어 수가 적절한지 확인
            words = text.split()
            if len(words) >= 2:
                completion_score += 0.05
            
            # 4. 반복 패턴 감지 (부정적 요소)
            repeated_chars = 0
            for char in set(text):
                count = text.count(char)
                if count > len(text) * 0.3:  # 한 글자가 30% 이상 반복
                    repeated_chars += 1
            
            repetition_penalty = min(0.2, repeated_chars * 0.05)
            
            # 5. 특수문자 비율 (너무 많으면 감점)
            special_chars = sum(1 for char in text if not char.isalnum() and char not in ' .,!?')
            if len(text) > 0:
                special_ratio = special_chars / len(text)
                special_penalty = min(0.1, special_ratio * 0.5)
            else:
                special_penalty = 0.0
            
            # 최종 신뢰도 계산
            confidence = base_score + length_score + korean_score + completion_score - repetition_penalty - special_penalty
            
            # 0.0 ~ 1.0 범위로 제한
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            logger.warning(f"⚠️ 텍스트 신뢰도 계산 실패: {e}")
            return 0.5  # 기본값
    
    def _calculate_confidence(self, text: str, duration: float) -> float:
        """간단한 신뢰도 계산"""
        if not text.strip():
            return 0.0
        
        # 기본 신뢰도
        base_confidence = 0.85
        
        # 텍스트 길이 기반 조정
        text_length = len(text.strip())
        if text_length < 3:
            base_confidence -= 0.2
        elif text_length > 50:
            base_confidence += 0.1
        
        # 지속 시간 기반 조정
        if duration < 1.0:
            base_confidence -= 0.1
        elif duration > 10.0:
            base_confidence += 0.05
        
        return min(0.99, max(0.1, base_confidence))
    
    def _create_segments(self, text: str, duration: float) -> List[Dict[str, Any]]:
        """기본적인 세그먼트 생성 (NeMo는 기본적으로 세그먼트 정보를 제공하지 않음)"""
        if not text.strip():
            return []
        
        # 간단한 단일 세그먼트 생성
        confidence = self._calculate_confidence(text, duration)
        
        return [{
            "id": 0,
            "text": text.strip(),
            "start": 0.0,
            "end": duration,
            "confidence": confidence,
            "words": []  # NeMo에서 단어 레벨 타임스탬프는 별도 처리 필요
        }]
    
    def is_healthy(self) -> bool:
        """NeMo 서비스 상태 확인"""
        return self.is_initialized and self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """NeMo 모델 정보 반환"""
        info = {
            "model_type": "nemo",
            "model_name": self.model_name,
            "device": self.device,
            "is_initialized": self.is_initialized,
            "is_healthy": self.is_healthy(),
            "sample_rate": self.sample_rate,
            "gpu_optimized": torch.cuda.is_available() and self.device == "cuda",
            "nemo_available": NEMO_AVAILABLE
        }
        
        # 모델이 로드된 경우 추가 정보
        if self.model is not None:
            try:
                # NeMo 모델의 설정 정보 추가
                if hasattr(self.model, 'cfg'):
                    info["model_config"] = {
                        "sample_rate": getattr(self.model.cfg, 'sample_rate', self.sample_rate),
                        "n_mels": getattr(self.model.cfg.preprocessor, 'n_mels', 'unknown'),
                        "vocab_size": len(getattr(self.model, 'decoder', {}).vocabulary) if hasattr(self.model, 'decoder') else 'unknown'
                    }
            except Exception as e:
                logger.warning(f"⚠️ 모델 설정 정보 가져오기 실패: {e}")
        
        return info 