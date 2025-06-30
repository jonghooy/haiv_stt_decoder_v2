#!/usr/bin/env python3
"""
배치 처리 즉시 응답 테스트
배치 요청을 보내고 즉시 batch_id를 받을 수 있는지 확인
"""

import requests
import time
import json

def test_immediate_batch_response():
    """배치 처리가 즉시 응답하는지 테스트"""
    url = "http://localhost:8004/batch/transcribe"
    
    # 테스트용 작은 오디오 파일 사용
    test_files = [
        "test_korean_sample1.wav",
        "test_korean_sample2.wav"
    ]
    
    # 요청 시작 시간 측정
    start_time = time.time()
    
    print("📤 배치 처리 요청 전송 중...")
    print(f"⏰ 요청 시작 시간: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    
    try:
        # multipart/form-data로 파일 업로드
        files = []
        for filename in test_files:
            try:
                with open(filename, 'rb') as f:
                    files.append(('files', (filename, f.read(), 'audio/wav')))
            except FileNotFoundError:
                print(f"⚠️ 파일을 찾을 수 없습니다: {filename}, 건너뜀")
                continue
        
        if not files:
            print("❌ 테스트할 파일이 없습니다.")
            return
        
        # 배치 처리 요청
        data = {
            'language': 'ko',
            'enable_word_timestamps': True,
            'enable_confidence': True,
            'enable_keyword_boosting': False
        }
        
        response = requests.post(url, files=files, data=data)
        
        # 응답 받은 시간 측정
        response_time = time.time()
        elapsed_time = response_time - start_time
        
        print(f"📥 응답 받음: {time.strftime('%H:%M:%S', time.localtime(response_time))}")
        print(f"⚡ 응답 시간: {elapsed_time:.3f}초")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("✅ 배치 처리 요청 성공!")
                print(f"🆔 Batch ID: {result.get('batch_id', 'N/A')}")
                print(f"📊 상태: {result.get('status', 'N/A')}")
                print(f"📝 메시지: {result.get('message', 'N/A')}")
                print(f"📁 총 파일 수: {result.get('total_files', 'N/A')}")
                print(f"🔗 진행률 URL: {result.get('progress_url', 'N/A')}")
                print(f"🔗 상태 URL: {result.get('status_url', 'N/A')}")
                
                # batch_id가 있으면 진행 상황 추적 테스트
                batch_id = result.get('batch_id')
                if batch_id:
                    print(f"\n🔄 배치 {batch_id} 진행 상황 추적 테스트...")
                    test_progress_tracking(batch_id)
            except Exception as json_error:
                print(f"❌ JSON 파싱 오류: {json_error}")
                print(f"응답 내용: {response.text}")
                
                # 응답이 문자열인 경우 (batch_id만 반환)
                if response.text and len(response.text) > 0:
                    print(f"🆔 Batch ID (문자열): {response.text}")
                    batch_id = response.text.strip('"')  # 따옴표 제거
                    if batch_id:
                        print(f"\n🔄 배치 {batch_id} 진행 상황 추적 테스트...")
                        test_progress_tracking(batch_id)
        else:
            print(f"❌ 요청 실패: {response.status_code}")
            print(f"응답: {response.text}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

def test_progress_tracking(batch_id):
    """배치 진행 상황 추적 테스트"""
    status_url = f"http://localhost:8004/batch/status/{batch_id}"
    
    print(f"📍 상태 확인 URL: {status_url}")
    
    for i in range(3):  # 3번 상태 확인
        try:
            time.sleep(1)  # 1초 대기
            
            response = requests.get(status_url)
            if response.status_code == 200:
                status_data = response.json()
                progress = status_data.get('progress', 0.0)
                status = status_data.get('status', 'unknown')
                processed = status_data.get('processed_files', 0)
                total = status_data.get('total_files', 0)
                
                print(f"  📊 진행률: {progress:.1%} ({processed}/{total}) - 상태: {status}")
                
                if status in ['completed', 'failed']:
                    print(f"  ✅ 배치 처리 완료: {status}")
                    break
            else:
                print(f"  ❌ 상태 확인 실패: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ 상태 확인 오류: {e}")

if __name__ == "__main__":
    print("🧪 배치 처리 즉시 응답 테스트")
    print("=" * 50)
    test_immediate_batch_response() 