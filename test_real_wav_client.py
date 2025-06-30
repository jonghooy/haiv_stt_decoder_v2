#!/usr/bin/env python3
"""
Real WAV File Test Client
실제 WAV 파일을 사용하여 STT 서버를 테스트하는 클라이언트
"""

import asyncio
import aiohttp
import base64
import time
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Optional, List

# 서버 설정
SERVER_URL = "http://localhost:8004"

class RealWAVTestClient:
    def __init__(self, server_url: str = SERVER_URL):
        self.server_url = server_url
    
    async def test_health(self) -> bool:
        """서버 헬스 체크"""
        print("🔍 서버 헬스 체크...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ 서버 상태: {data['status']}")
                        print(f"   GPU: {data['gpu_info']['device']}")
                        print(f"   모델: {data['model_info']['model']}")
                        print(f"   로드됨: {data['model_info']['loaded']}")
                        return True
                    else:
                        print(f"❌ 헬스 체크 실패: {response.status}")
                        return False
        except Exception as e:
            print(f"❌ 서버 연결 실패: {e}")
            return False
    
    def convert_wav_to_pcm_16khz(self, wav_file: str) -> Optional[str]:
        """WAV 파일을 PCM 16kHz base64로 변환"""
        wav_path = Path(wav_file)
        if not wav_path.exists():pm,
            print(f"❌ 파일이 존재하지 않음: {wav_file}")
            return None
        
        try:
            # WAV 파일 로드
            print(f"📁 파일 로드 중: {wav_file}")
            audio_data, sample_rate = sf.read(wav_path)
            
            print(f"   원본 샘플레이트: {sample_rate}Hz")
            print(f"   원본 길이: {len(audio_data)} 샘플 ({len(audio_data)/sample_rate:.2f}초)")
            print(f"   원본 데이터 타입: {audio_data.dtype}")
            print(f"   원본 채널: {'스테레오' if len(audio_data.shape) > 1 else '모노'}")
            
            # 스테레오인 경우 모노로 변환
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                print("   🔄 스테레오 → 모노 변환")
            
            # 16kHz로 리샘플링
            if sample_rate != 16000:
                print(f"   🔄 리샘플링: {sample_rate}Hz → 16000Hz")
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # int16으로 변환
            if audio_data.dtype != np.int16:
                print("   🔄 int16으로 변환")
                # float32/float64 -> int16 변환
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    # 클리핑 방지
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
            
            # base64 인코딩
            audio_bytes = audio_data.tobytes()
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            
            print(f"✅ 변환 완료:")
            print(f"   최종 샘플 수: {len(audio_data)}")
            print(f"   최종 길이: {len(audio_data)/16000:.2f}초")
            print(f"   바이트 크기: {len(audio_bytes)} bytes")
            print(f"   Base64 크기: {len(base64_audio)} chars")
            
            return base64_audio
            
        except Exception as e:
            print(f"❌ 파일 변환 오류: {e}")
            return None
    
    async def transcribe_wav_file(self, wav_file: str, vad_enabled: bool = True) -> Optional[dict]:
        """WAV 파일 전사"""
        print(f"\n🎤 WAV 파일 전사: {wav_file} (VAD: {'ON' if vad_enabled else 'OFF'})")
        
        # WAV를 PCM 16kHz로 변환
        audio_data = self.convert_wav_to_pcm_16khz(wav_file)
        if not audio_data:
            return None
        
        payload = {
            "audio_data": audio_data,
            "language": "ko",
            "audio_format": "pcm_16khz",
            "vad_enabled": vad_enabled
        }
        
        try:
            print("📤 전사 요청 중...")
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.post(
                    f"{self.server_url}/transcribe",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=60)  # 긴 오디오를 위해 타임아웃 증가
                ) as response:
                    
                    request_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ 전사 성공!")
                        print(f"   텍스트: {result.get('text', 'N/A')}")
                        print(f"   RTF: {result.get('rtf', 'N/A'):.4f}")
                        print(f"   처리시간: {result.get('processing_time', 'N/A'):.3f}초")
                        print(f"   요청시간: {request_time:.3f}초")
                        print(f"   오디오 길이: {result.get('audio_duration', 'N/A'):.3f}초")
                        print(f"   VAD 사용: {result.get('vad_enabled', 'N/A')}")
                        
                        return result
                    else:
                        error_text = await response.text()
                        print(f"❌ 전사 실패: {response.status}")
                        print(f"   오류: {error_text}")
                        return None
                        
        except Exception as e:
            print(f"❌ 전사 요청 오류: {e}")
            return None
    
    async def compare_vad_performance(self, wav_file: str):
        """VAD ON/OFF 성능 비교"""
        print(f"\n🧪 VAD 성능 비교: {wav_file}")
        print("=" * 60)
        
        # VAD ON 테스트
        result_vad_on = await self.transcribe_wav_file(wav_file, vad_enabled=True)
        
        # VAD OFF 테스트
        result_vad_off = await self.transcribe_wav_file(wav_file, vad_enabled=False)
        
        # 비교 결과
        if result_vad_on and result_vad_off:
            print(f"\n📊 성능 비교 결과:")
            print("=" * 40)
            
            vad_on_rtf = result_vad_on.get('rtf', 999)
            vad_off_rtf = result_vad_off.get('rtf', 999)
            vad_on_time = result_vad_on.get('processing_time', 999)
            vad_off_time = result_vad_off.get('processing_time', 999)
            
            print(f"VAD ON  - RTF: {vad_on_rtf:.4f}, 처리시간: {vad_on_time:.3f}초")
            print(f"VAD OFF - RTF: {vad_off_rtf:.4f}, 처리시간: {vad_off_time:.3f}초")
            
            if vad_on_rtf < vad_off_rtf:
                improvement = ((vad_off_rtf - vad_on_rtf) / vad_off_rtf) * 100
                print(f"🏆 VAD ON이 {improvement:.1f}% 더 빠름!")
            elif vad_off_rtf < vad_on_rtf:
                improvement = ((vad_on_rtf - vad_off_rtf) / vad_on_rtf) * 100
                print(f"🏆 VAD OFF가 {improvement:.1f}% 더 빠름!")
            else:
                print("➖ RTF 성능 차이 없음")
            
            # 텍스트 비교
            text_on = result_vad_on.get('text', '').strip()
            text_off = result_vad_off.get('text', '').strip()
            
            print(f"\n📝 전사 결과 비교:")
            print(f"VAD ON : \"{text_on}\"")
            print(f"VAD OFF: \"{text_off}\"")
            
            if text_on == text_off:
                print("✅ 전사 결과 동일")
            else:
                print("⚠️ 전사 결과 다름")

async def main():
    """메인 테스트 함수"""
    print("🚀 실제 WAV 파일 STT 테스트 시작")
    print("=" * 60)
    
    client = RealWAVTestClient()
    
    # 서버 상태 확인
    if not await client.test_health():
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        return
    
    # 테스트할 WAV 파일들
    test_files = [
        "test_korean_sample1.wav",
        "test_korean_sample2.wav"
    ]
    
    # 존재하는 파일만 필터링
    existing_files = []
    for file in test_files:
        if Path(file).exists():
            existing_files.append(file)
            print(f"✅ 발견된 테스트 파일: {file}")
        else:
            print(f"⚠️ 파일 없음: {file}")
    
    if not existing_files:
        print("❌ 테스트할 WAV 파일이 없습니다.")
        return
    
    # 각 파일에 대해 테스트 실행
    for wav_file in existing_files:
        await client.compare_vad_performance(wav_file)
        print("\n" + "="*60)
    
    print("✅ 모든 실제 WAV 파일 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main()) 