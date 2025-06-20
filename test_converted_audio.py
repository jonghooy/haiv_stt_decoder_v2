#!/usr/bin/env python3
"""
Convert WAV to PCM 16kHz and Test
WAV 파일을 PCM 16kHz로 변환하여 테스트하는 스크립트
"""

import requests
import base64
import time
import numpy as np
import soundfile as sf
from pathlib import Path

# 서버 설정
SERVER_URL = "http://localhost:8003"

def test_health():
    """헬스 체크 테스트"""
    print("🔍 헬스 체크...")
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 서버 상태: {data['status']}")
            print(f"   GPU: {data['gpu_info']['device']}")
            print(f"   모델: {data['model_info']['model']}")
            print(f"   로드됨: {data['model_info']['loaded']}")
            return True
        else:
            print(f"❌ 헬스 체크 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 헬스 체크 오류: {e}")
        return False

def convert_wav_to_pcm_16khz(audio_file):
    """WAV 파일을 PCM 16kHz로 변환"""
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"❌ 파일이 존재하지 않음: {audio_file}")
        return None
    
    try:
        # WAV 파일 로드
        audio_data, sample_rate = sf.read(audio_path)
        print(f"📁 원본 파일: {audio_file}")
        print(f"   샘플레이트: {sample_rate}Hz")
        print(f"   길이: {len(audio_data)} 샘플 ({len(audio_data)/sample_rate:.2f}초)")
        print(f"   데이터 타입: {audio_data.dtype}")
        
        # 스테레오인 경우 모노로 변환
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            print("   스테레오 → 모노 변환")
        
        # 16kHz로 리샘플링
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            print(f"   리샘플링: {sample_rate}Hz → 16000Hz")
        
        # int16으로 변환
        if audio_data.dtype != np.int16:
            # float32 -> int16 변환
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
            print("   int16으로 변환")
        
        # base64 인코딩
        audio_bytes = audio_data.tobytes()
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"✅ 변환 완료: {len(audio_data)} 샘플, {len(audio_bytes)} bytes")
        return base64_audio
        
    except Exception as e:
        print(f"❌ 파일 변환 오류: {e}")
        return None

def test_transcribe_with_converted_file(audio_file):
    """변환된 오디오 파일로 전사 테스트"""
    print(f"\n🎤 변환된 오디오 전사 테스트: {audio_file}")
    
    # WAV를 PCM 16kHz로 변환
    audio_data = convert_wav_to_pcm_16khz(audio_file)
    if not audio_data:
        return False
    
    payload = {
        "audio_data": audio_data,
        "language": "ko",
        "audio_format": "pcm_16khz",  # PCM 16kHz 형식
        "enable_confidence": True,    # 신뢰도 활성화
        "enable_timestamps": True     # 타임스탬프 활성화
    }
    
    try:
        print("📤 전사 요청 중...")
        start_time = time.time()
        
        response = requests.post(
            f"{SERVER_URL}/infer/utterance",  # 신뢰도 정보 포함한 엔드포인트
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 전사 성공!")
            print(f"   텍스트: {result.get('text', 'N/A')}")
            
            # 메트릭스 정보 출력
            metrics = result.get('metrics', {})
            print(f"   RTF: {metrics.get('rtf', 'N/A')}")
            print(f"   처리시간: {metrics.get('inference_time', 'N/A')}초")
            print(f"   오디오 길이: {metrics.get('audio_duration', 'N/A')}초")
            print(f"   요청시간: {request_time:.3f}초")
            
            # 🎯 신뢰도 정보 출력
            segments = result.get('segments', [])
            if segments:
                print(f"\n📊 신뢰도 정보:")
                for i, segment in enumerate(segments):
                    confidence = segment.get('confidence', 'N/A')
                    if confidence != 'N/A' and confidence is not None:
                        print(f"   세그먼트 {i+1}: {confidence:.3f}")
                    else:
                        print(f"   세그먼트 {i+1}: {confidence}")
                    
                    # 단어별 신뢰도도 출력
                    words = segment.get('words', [])
                    if words:
                        print(f"   단어별 신뢰도:")
                        for word in words:
                            word_confidence = word.get('confidence', 'N/A')
                            if word_confidence != 'N/A' and word_confidence is not None:
                                print(f"     '{word.get('word', '')}': {word_confidence:.3f}")
                            else:
                                print(f"     '{word.get('word', '')}': {word_confidence}")
            else:
                print(f"\n⚠️ 세그먼트 정보가 없습니다.")
            
            return True
        else:
            print(f"❌ 전사 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 전사 요청 오류: {e}")
        return False

def test_vad_comparison_converted(audio_file):
    """변환된 오디오로 VAD ON/OFF 비교 테스트"""
    print(f"\n🧪 VAD 비교 테스트: {audio_file}")
    
    # WAV를 PCM 16kHz로 변환
    audio_data = convert_wav_to_pcm_16khz(audio_file)
    if not audio_data:
        return
    
    # VAD ON 테스트
    print("\n📋 VAD ON 테스트:")
    payload_vad_on = {
        "audio_data": audio_data,
        "language": "ko",
        "audio_format": "pcm_16khz",
        "vad_enabled": True
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{SERVER_URL}/transcribe", json=payload_vad_on, timeout=30)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   텍스트: {result.get('text', 'N/A')}")
            print(f"   RTF: {result.get('rtf', 'N/A')}")
            print(f"   처리시간: {result.get('processing_time', 'N/A')}초")
            vad_on_rtf = result.get('rtf', 999)
        else:
            print(f"   ❌ 실패: {response.status_code}")
            vad_on_rtf = 999
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        vad_on_rtf = 999
    
    # VAD OFF 테스트
    print("\n📋 VAD OFF 테스트:")
    payload_vad_off = {
        "audio_data": audio_data,
        "language": "ko",
        "audio_format": "pcm_16khz",
        "vad_enabled": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{SERVER_URL}/transcribe", json=payload_vad_off, timeout=30)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   텍스트: {result.get('text', 'N/A')}")
            print(f"   RTF: {result.get('rtf', 'N/A')}")
            print(f"   처리시간: {result.get('processing_time', 'N/A')}초")
            vad_off_rtf = result.get('rtf', 999)
        else:
            print(f"   ❌ 실패: {response.status_code}")
            vad_off_rtf = 999
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        vad_off_rtf = 999
    
    # 비교 결과
    print(f"\n📊 VAD 비교 결과:")
    print(f"   VAD ON  RTF: {vad_on_rtf}")
    print(f"   VAD OFF RTF: {vad_off_rtf}")
    
    if vad_on_rtf < vad_off_rtf:
        improvement = ((vad_off_rtf - vad_on_rtf) / vad_off_rtf) * 100
        print(f"   ✅ VAD ON이 {improvement:.1f}% 더 빠름!")
    elif vad_off_rtf < vad_on_rtf:
        improvement = ((vad_on_rtf - vad_off_rtf) / vad_on_rtf) * 100
        print(f"   ✅ VAD OFF가 {improvement:.1f}% 더 빠름!")
    else:
        print("   ➖ 성능 차이 없음")

def main():
    """메인 테스트"""
    print("🚀 Converted Audio Server 테스트 시작")
    print("=" * 50)
    
    # 헬스 체크
    if not test_health():
        print("❌ 서버에 연결할 수 없습니다.")
        return
    
    # 테스트 파일들
    test_files = [
        "test_korean_sample1.wav",
        "test_korean_sample2.wav"
    ]
    
    # 각 파일 테스트
    for test_file in test_files:
        test_transcribe_with_converted_file(test_file)
        test_vad_comparison_converted(test_file)
    
    print("\n✅ 모든 테스트 완료!")

if __name__ == "__main__":
    main() 