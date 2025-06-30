#!/usr/bin/env python3
"""
NeMo STT 개선된 설정 테스트 스크립트
- 빔 서치 크기 증가 (8)
- 길이 패널티 감소 (0.3) 
- 다중 후보 처리
- 향상된 오디오 전처리
- 더 긴 오버랩 (3초)
"""

import asyncio
import numpy as np
import logging
import time
import soundfile as sf
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_improved_nemo():
    """개선된 NeMo 설정 테스트"""
    try:
        # NeMo 서비스 임포트 및 초기화
        from src.api.nemo_stt_service import NeMoSTTService
        
        logger.info("🤖 NeMo STT 서비스 초기화 중...")
        service = NeMoSTTService(model_name="./FastConformer-Transducer-BPE_9.75.nemo")
        
        # 모델 초기화
        await service.initialize()
        
        if not service.is_healthy():
            logger.error("❌ NeMo 서비스 초기화 실패")
            return
        
        logger.info("✅ NeMo 서비스 초기화 완료")
        
        # 테스트할 WAV 파일들
        test_files = [
            "test_samples/test_short.wav",     # 짧은 파일
            "test_samples/test_medium.wav",    # 중간 길이 파일
            "test_samples/test_long.wav"       # 긴 파일
        ]
        
        for wav_file in test_files:
            if not Path(wav_file).exists():
                logger.warning(f"⚠️ 테스트 파일 없음: {wav_file}")
                continue
                
            logger.info(f"\n{'='*60}")
            logger.info(f"🎵 테스트 파일: {wav_file}")
            logger.info(f"{'='*60}")
            
            try:
                # WAV 파일 로드
                audio_data, sample_rate = sf.read(wav_file)
                
                # 모노로 변환 (필요한 경우)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # 16kHz로 리샘플링 (필요한 경우)
                if sample_rate != 16000:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
                
                logger.info(f"📊 오디오 정보:")
                logger.info(f"   • 길이: {len(audio_data)/sample_rate:.2f}초")
                logger.info(f"   • 샘플링 레이트: {sample_rate}Hz")
                logger.info(f"   • 샘플 수: {len(audio_data):,}개")
                
                # 전사 수행
                logger.info("\n🎤 전사 시작...")
                start_time = time.time()
                
                result = await service.transcribe_audio(
                    audio_data.astype(np.float32).tobytes(),
                    audio_format="pcm_16khz",
                    language="ko"
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                rtf = processing_time / (len(audio_data) / sample_rate)
                
                logger.info("\n📝 전사 결과:")
                logger.info(f"   • 텍스트: '{result.text}'")
                logger.info(f"   • 신뢰도: {result.confidence:.3f}")
                logger.info(f"   • 처리 시간: {processing_time:.2f}초")
                logger.info(f"   • RTF: {rtf:.3f}")
                
                if hasattr(result, 'segments') and result.segments:
                    logger.info(f"   • 세그먼트 수: {len(result.segments)}")
                    for i, segment in enumerate(result.segments[:3]):  # 처음 3개만 표시
                        logger.info(f"     세그먼트 {i+1}: '{segment.get('text', '')[:50]}...'")
                
                # 텍스트 품질 분석
                text = result.text
                if text:
                    logger.info("\n📊 텍스트 품질 분석:")
                    logger.info(f"   • 텍스트 길이: {len(text)}자")
                    logger.info(f"   • 단어 수: {len(text.split())}개")
                    
                    # 한국어 비율
                    korean_chars = sum(1 for char in text if '\uac00' <= char <= '\ud7af')
                    korean_ratio = korean_chars / len(text) if text else 0
                    logger.info(f"   • 한국어 비율: {korean_ratio:.1%}")
                    
                    # 특수문자 비율
                    special_chars = sum(1 for char in text if not char.isalnum() and char not in ' .,!?')
                    special_ratio = special_chars / len(text) if text else 0
                    logger.info(f"   • 특수문자 비율: {special_ratio:.1%}")
                else:
                    logger.warning("⚠️ 전사 결과가 비어있음")
                
            except Exception as e:
                logger.error(f"❌ 파일 {wav_file} 처리 실패: {e}")
                import traceback
                logger.error(f"상세 에러: {traceback.format_exc()}")
        
        # 설정 확인
        logger.info(f"\n{'='*60}")
        logger.info("🔧 모델 설정 확인")
        logger.info(f"{'='*60}")
        
        model_info = service.get_model_info()
        logger.info(f"📋 모델 정보:")
        for key, value in model_info.items():
            logger.info(f"   • {key}: {value}")
        
        # 디코딩 설정 확인
        if hasattr(service.model, 'cfg') and hasattr(service.model.cfg, 'decoding'):
            decoding = service.model.cfg.decoding
            logger.info(f"\n🔧 디코딩 설정:")
            logger.info(f"   • 전략: {getattr(decoding, 'strategy', 'unknown')}")
            
            if hasattr(decoding, 'beam'):
                beam = decoding.beam
                logger.info(f"   • 빔 크기: {getattr(beam, 'beam_size', 'unknown')}")
                logger.info(f"   • 길이 패널티: {getattr(beam, 'len_pen', 'unknown')}")
                logger.info(f"   • 점수 정규화: {getattr(beam, 'score_norm', 'unknown')}")
                logger.info(f"   • 최고 가설만 반환: {getattr(beam, 'return_best_hypothesis', 'unknown')}")
        
        logger.info("\n✅ 모든 테스트 완료")
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(f"상세 에러: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(test_improved_nemo()) 