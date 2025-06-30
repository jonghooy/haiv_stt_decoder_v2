#!/usr/bin/env python3
"""
모델 선택 기능 테스트 스크립트
Whisper와 NeMo 모델 간 전환을 테스트
"""

import subprocess
import sys
import time
import requests
import base64
import numpy as np


def generate_test_audio(duration=3, sample_rate=16000):
    """테스트용 오디오 생성"""
    # 간단한 사인파 생성
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 음계
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    return audio.astype(np.float32)


def audio_to_base64(audio, sample_rate=16000):
    """오디오를 base64로 인코딩"""
    # PCM 16-bit로 변환
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    return base64.b64encode(audio_bytes).decode('utf-8')


def test_server_health(port=8004):
    """서버 헬스 체크"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def test_model_info(port=8004):
    """모델 정보 테스트"""
    try:
        response = requests.get(f"http://localhost:{port}/models/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def test_transcription(port=8004):
    """전사 테스트"""
    try:
        # 테스트 오디오 생성
        audio = generate_test_audio()
        audio_base64 = audio_to_base64(audio)
        
        # 전사 요청
        data = {
            "audio_data": audio_base64,
            "language": "ko",
            "audio_format": "pcm_16khz"
        }
        
        response = requests.post(f"http://localhost:{port}/transcribe", json=data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"전사 실패: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"전사 중 오류: {e}")
        return None


def start_server(model_type, port=8004):
    """서버 시작"""
    cmd = [
        sys.executable, "gpu_optimized_stt_server.py",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--model", model_type
    ]
    
    print(f"🚀 서버 시작: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def main():
    """메인 테스트 함수"""
    test_configs = [
        # Whisper 모델
        ("whisper", 8001),
        
        # NeMo 모델 (설치된 경우)
        ("nemo", 8002),
    ]
    
    print("🧪 모델 선택 기능 테스트 시작")
    print("=" * 60)
    
    # 지원 모델 목록 확인
    print("\n📋 지원 모델 목록:")
    list_proc = subprocess.run([
        sys.executable, "gpu_optimized_stt_server.py", "--list-models"
    ], capture_output=True, text=True)
    print(list_proc.stdout)
    
    results = []
    
    for model_type, port in test_configs:
        print(f"\n🔧 테스트: {model_type} (포트 {port})")
        print("-" * 40)
        
        # 서버 시작
        process = start_server(model_type, port)
        
        try:
            # 서버 시작 대기
            print("⏱️ 서버 시작 대기 중...")
            max_wait = 120  # 2분 대기
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                if test_server_health(port):
                    print("✅ 서버 시작 완료")
                    break
                time.sleep(2)
            else:
                print("❌ 서버 시작 타임아웃")
                results.append((model_type, "시작 실패"))
                continue
            
            # 모델 정보 확인
            print("📊 모델 정보 확인 중...")
            model_info = test_model_info(port)
            if model_info:
                current_model = model_info.get("current_model", {})
                print(f"   - 모델 타입: {current_model.get('model_type')}")
                print(f"   - 모델 이름: {current_model.get('model_name')}")
                print(f"   - 초기화 상태: {current_model.get('is_initialized')}")
                print(f"   - 헬스 상태: {current_model.get('is_healthy')}")
            else:
                print("❌ 모델 정보 가져오기 실패")
                results.append((model_type, "정보 확인 실패"))
                continue
            
            # 전사 테스트
            print("🎤 전사 테스트 중...")
            transcription = test_transcription(port)
            if transcription:
                print(f"   - 전사 결과: '{transcription.get('text', 'N/A')}'")
                print(f"   - 처리 시간: {transcription.get('processing_time', 0):.3f}초")
                print(f"   - RTF: {transcription.get('rtf', 0):.3f}")
                print(f"   - 모델 타입: {transcription.get('model_type')}")
                results.append((model_type, "성공"))
            else:
                print("❌ 전사 테스트 실패")
                results.append((model_type, "전사 실패"))
            
        except KeyboardInterrupt:
            print("\n⚠️ 사용자 중단")
            break
        except Exception as e:
            print(f"❌ 테스트 중 오류: {e}")
            results.append((model_type, f"오류: {e}"))
        finally:
            # 서버 종료
            print("🛑 서버 종료 중...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 테스트 결과 요약:")
    print("=" * 60)
    
    for model_type, result in results:
        status_emoji = "✅" if result == "성공" else "❌"
        print(f"{status_emoji} {model_type:10} | {result}")
    
    print(f"\n총 {len(results)}개 테스트 완료")
    success_count = sum(1 for _, result in results if result == "성공")
    print(f"성공: {success_count}, 실패: {len(results) - success_count}")


if __name__ == "__main__":
    main() 