#!/usr/bin/env python3
"""
WAV 파일을 사용한 STT 테스트 클라이언트
실제 오디오 파일로 Whisper와 NeMo 모델을 테스트합니다.
"""

import argparse
import asyncio
import aiohttp
import base64
import json
import time
import os
import wave
import numpy as np
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WAVSTTClient:
    def __init__(self, server_url: str = "http://localhost:8004"):
        self.server_url = server_url
        
    async def test_server_health(self):
        """서버 상태 확인"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/health") as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ 서버 상태: {result}")
                        return True
                    else:
                        logger.error(f"❌ 서버 오류: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ 서버 연결 실패: {e}")
            return False

    async def get_model_info(self):
        """모델 정보 확인"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/models/info") as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"📊 모델 정보:")
                        current_model = result.get("current_model", {})
                        logger.info(f"   - 모델 타입: {current_model.get('model_type')}")
                        logger.info(f"   - 모델 이름: {current_model.get('model_name')}")
                        logger.info(f"   - 초기화 상태: {current_model.get('is_initialized')}")
                        logger.info(f"   - 헬스 상태: {current_model.get('is_healthy')}")
                        return result
                    else:
                        logger.error(f"❌ 모델 정보 가져오기 실패: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"❌ 모델 정보 오류: {e}")
            return None

    def load_wav_file(self, wav_path: str) -> tuple[np.ndarray, int]:
        """WAV 파일 로드 및 16kHz로 리샘플링"""
        try:
            with wave.open(wav_path, 'rb') as wav_file:
                # WAV 파일 정보
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frames = wav_file.getnframes()
                duration = frames / sample_rate
                
                logger.info(f"📁 WAV 파일 정보:")
                logger.info(f"   - 파일: {os.path.basename(wav_path)}")
                logger.info(f"   - 샘플레이트: {sample_rate}Hz")
                logger.info(f"   - 채널: {channels}")
                logger.info(f"   - 비트 깊이: {sample_width * 8}bit")
                logger.info(f"   - 길이: {duration:.2f}초")
                
                # 오디오 데이터 읽기
                audio_data = wav_file.readframes(frames)
                
                # numpy 배열로 변환
                if sample_width == 1:
                    audio_array = np.frombuffer(audio_data, dtype=np.uint8)
                    audio_array = (audio_array.astype(np.float32) - 128) / 128
                elif sample_width == 2:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_array = audio_array.astype(np.float32) / 32768
                elif sample_width == 4:
                    audio_array = np.frombuffer(audio_data, dtype=np.int32)
                    audio_array = audio_array.astype(np.float32) / 2147483648
                else:
                    raise ValueError(f"지원하지 않는 비트 깊이: {sample_width * 8}bit")
                
                # 스테레오를 모노로 변환
                if channels == 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1)
                elif channels > 2:
                    audio_array = audio_array.reshape(-1, channels).mean(axis=1)
                
                # 16kHz로 리샘플링 (간단한 방법)
                if sample_rate != 16000:
                    # 리샘플링 비율 계산
                    resample_ratio = 16000 / sample_rate
                    new_length = int(len(audio_array) * resample_ratio)
                    
                    # 선형 보간으로 리샘플링
                    indices = np.linspace(0, len(audio_array) - 1, new_length)
                    audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)
                    
                    logger.info(f"🔄 리샘플링: {sample_rate}Hz → 16000Hz")
                
                return audio_array, 16000
                
        except Exception as e:
            logger.error(f"❌ WAV 파일 로드 실패: {e}")
            raise

    def audio_to_base64(self, audio_array: np.ndarray) -> str:
        """오디오를 PCM 16kHz base64로 인코딩"""
        # int16으로 변환
        audio_int16 = (audio_array * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return base64.b64encode(audio_bytes).decode('utf-8')

    async def transcribe_with_confidence(self, audio_base64: str, language: str = "ko") -> dict:
        """신뢰도 분석 전사 요청"""
        payload = {
            "audio_data": audio_base64,
            "language": language,
            "audio_format": "pcm_16khz",
            "enable_confidence": True,
            "enable_timestamps": True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.post(
                    f"{self.server_url}/infer/utterance",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        request_time = time.time() - start_time
                        
                        logger.info(f"✅ 전사 완료:")
                        logger.info(f"   텍스트: '{result.get('text', 'N/A')}'")
                        logger.info(f"   RTF: {result.get('rtf', 'N/A')}")
                        logger.info(f"   처리시간: {result.get('processing_time', 'N/A')}초")
                        logger.info(f"   요청시간: {request_time:.3f}초")
                        logger.info(f"   오디오 길이: {result.get('audio_duration', 'N/A')}초")
                        logger.info(f"   모델 타입: {result.get('model_type', 'N/A')}")
                        
                        # 세그먼트별 신뢰도 출력
                        segments = result.get('segments', [])
                        if segments:
                            logger.info(f"📊 세그먼트별 신뢰도:")
                            for i, segment in enumerate(segments, 1):
                                confidence = segment.get('confidence')
                                start_time = segment.get('start')
                                end_time = segment.get('end')
                                text = segment.get('text', '')
                                
                                # None 값 처리
                                confidence_str = f"{confidence:.3f}" if confidence is not None else "N/A"
                                start_str = f"{start_time:.2f}" if start_time is not None else "N/A"
                                end_str = f"{end_time:.2f}" if end_time is not None else "N/A"
                                
                                logger.info(f"   [{i}] {start_str}s-{end_str}s: '{text}' (신뢰도: {confidence_str})")
                        
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ 전사 실패 ({response.status}): {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ 전사 요청 실패: {e}")
            return None

    async def basic_transcribe(self, audio_base64: str, language: str = "ko") -> dict:
        """기본 전사 요청"""
        payload = {
            "audio_data": audio_base64,
            "language": language,
            "audio_format": "pcm_16khz"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.post(
                    f"{self.server_url}/transcribe",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        request_time = time.time() - start_time
                        
                        logger.info(f"✅ 기본 전사 완료:")
                        logger.info(f"   텍스트: '{result.get('text', 'N/A')}'")
                        logger.info(f"   RTF: {result.get('rtf', 'N/A')}")
                        logger.info(f"   처리시간: {result.get('processing_time', 'N/A')}초")
                        logger.info(f"   요청시간: {request_time:.3f}초")
                        logger.info(f"   모델 타입: {result.get('model_type', 'N/A')}")
                        
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ 기본 전사 실패 ({response.status}): {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ 기본 전사 요청 실패: {e}")
            return None


async def main():
    parser = argparse.ArgumentParser(description="WAV 파일 STT 테스트 클라이언트")
    parser.add_argument("wav_file", help="테스트할 WAV 파일 경로")
    parser.add_argument("--port", type=int, default=8004, help="서버 포트 (기본값: 8004)")
    parser.add_argument("--host", default="localhost", help="서버 호스트 (기본값: localhost)")
    parser.add_argument("--language", default="ko", help="언어 코드 (기본값: ko)")
    parser.add_argument("--basic", action="store_true", help="기본 전사 사용 (신뢰도 분석 없음)")
    
    args = parser.parse_args()
    
    # WAV 파일 존재 확인
    if not os.path.exists(args.wav_file):
        logger.error(f"❌ WAV 파일을 찾을 수 없습니다: {args.wav_file}")
        return
    
    # 클라이언트 생성
    server_url = f"http://{args.host}:{args.port}"
    client = WAVSTTClient(server_url)
    
    logger.info(f"🎯 STT 테스트 시작")
    logger.info(f"   서버: {server_url}")
    logger.info(f"   WAV 파일: {args.wav_file}")
    logger.info(f"   언어: {args.language}")
    logger.info(f"=" * 60)
    
    # 서버 상태 확인
    logger.info("🔍 서버 상태 확인 중...")
    if not await client.test_server_health():
        logger.error("❌ 서버에 연결할 수 없습니다.")
        return
    
    # 모델 정보 확인
    logger.info("📊 모델 정보 확인 중...")
    model_info = await client.get_model_info()
    if not model_info:
        logger.warning("⚠️ 모델 정보를 가져올 수 없습니다.")
    
    try:
        # WAV 파일 로드
        logger.info("📁 WAV 파일 로드 중...")
        audio_array, sample_rate = client.load_wav_file(args.wav_file)
        
        # Base64 인코딩
        logger.info("🔄 오디오 인코딩 중...")
        audio_base64 = client.audio_to_base64(audio_array)
        
        # 전사 수행
        if args.basic:
            logger.info("🎤 기본 전사 수행 중...")
            result = await client.basic_transcribe(audio_base64, args.language)
        else:
            logger.info("🎤 신뢰도 분석 전사 수행 중...")
            result = await client.transcribe_with_confidence(audio_base64, args.language)
        
        if result:
            logger.info("=" * 60)
            logger.info("🎉 전사 테스트 성공!")
        else:
            logger.error("❌ 전사 테스트 실패!")
            
    except Exception as e:
        logger.error(f"❌ 테스트 중 오류 발생: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 