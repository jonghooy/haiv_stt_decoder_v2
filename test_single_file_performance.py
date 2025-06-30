#!/usr/bin/env python3
"""
단일 파일 STT 성능 테스트
"""

import requests
import time
import base64

def test_single_file(filename):
    """단일 파일 STT 테스트"""
    url = "http://localhost:8004/transcribe"
    
    try:
        # 파일을 base64로 인코딩
        with open(filename, 'rb') as f:
            audio_content = f.read()
            audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        
        # STT 요청
        request_data = {
            'audio_data': audio_base64,
            'language': 'ko',
            'audio_format': 'pcm_16khz'
        }
        
        start_time = time.time()
        response = requests.post(url, json=request_data)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            
            rtf = result.get('rtf', 0)
            audio_duration = result.get('audio_duration', 0)
            processing_time = result.get('processing_time', 0)
            text = result.get('text', '')
            
            print(f"✅ 단일 파일 처리 성공!")
            print(f"📁 파일: {filename}")
            print(f"📊 오디오 길이: {audio_duration:.1f}초")
            print(f"⚡ 처리 시간: {processing_time:.3f}초")
            print(f"🚀 RTF: {rtf:.3f}")
            print(f"⏱️ 전체 요청 시간: {end_time - start_time:.3f}초")
            print(f"📝 전사 결과: {text[:100]}...")
            print()
            
            return rtf
            
        else:
            print(f"❌ 요청 실패: {response.status_code}")
            print(f"응답: {response.text}")
            return 0.0
            
    except Exception as e:
        print(f"❌ 오류: {e}")
        return 0.0

if __name__ == "__main__":
    print("🔍 단일 파일 STT 성능 테스트")
    print("=" * 40)
    
    files = ["test_korean_sample1.wav", "test_korean_sample2.wav"]
    rtfs = []
    
    for filename in files:
        rtf = test_single_file(filename)
        if rtf > 0:
            rtfs.append(rtf)
    
    if rtfs:
        avg_rtf = sum(rtfs) / len(rtfs)
        print(f"📊 단일 파일 평균 RTF: {avg_rtf:.3f}")
        
        # 예상 배치 처리 시간 계산
        print(f"🔄 순차 처리시 예상 총 RTF: {avg_rtf * 2:.3f} (2개 파일)")
        print(f"🚀 병렬 처리시 예상 총 RTF: {avg_rtf:.3f} (동시 처리)")
    else:
        print("❌ 성능 측정 실패") 