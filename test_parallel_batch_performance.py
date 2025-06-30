#!/usr/bin/env python3
"""
배치 처리 병렬 성능 테스트
순차 처리 vs 병렬 처리 RTF 비교
"""

import requests
import time
import json
import asyncio
import aiohttp

def test_single_file_rtf():
    """단일 파일 RTF 측정 (기준점)"""
    url = "http://localhost:8004/transcribe/file"
    
    test_files = [
        "test_korean_sample1.wav",
        "test_korean_sample2.wav"
    ]
    
    print("🔍 단일 파일 RTF 측정...")
    single_file_rtfs = []
    
    for filename in test_files:
        try:
            start_time = time.time()
            
            with open(filename, 'rb') as f:
                files = {'audio': (filename, f, 'audio/wav')}
                data = {'language': 'ko', 'vad_filter': False}
                
                response = requests.post(url, files=files, data=data)
                
            end_time = time.time()
            request_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                rtf = result.get('rtf', 0)
                audio_duration = result.get('audio_duration', 0)
                processing_time = result.get('processing_time', 0)
                
                single_file_rtfs.append(rtf)
                print(f"  📁 {filename}: RTF={rtf:.3f}, 오디오={audio_duration:.1f}초, 처리={processing_time:.3f}초")
            else:
                print(f"  ❌ {filename}: 요청 실패 ({response.status_code})")
                
        except Exception as e:
            print(f"  ❌ {filename}: 오류 - {e}")
    
    if single_file_rtfs:
        avg_rtf = sum(single_file_rtfs) / len(single_file_rtfs)
        print(f"📊 단일 파일 평균 RTF: {avg_rtf:.3f}")
        return avg_rtf
    
    return 0.0

def test_batch_processing_rtf():
    """배치 처리 RTF 측정 (병렬 처리)"""
    url = "http://localhost:8004/batch/transcribe"
    
    test_files = [
        "test_korean_sample1.wav",
        "test_korean_sample2.wav"
    ]
    
    print("\n🚀 배치 처리 RTF 측정...")
    
    try:
        # 배치 요청 시작
        batch_start_time = time.time()
        
        # multipart/form-data로 파일 업로드
        files = []
        for filename in test_files:
            try:
                with open(filename, 'rb') as f:
                    files.append(('files', (filename, f.read(), 'audio/wav')))
            except FileNotFoundError:
                print(f"⚠️ 파일을 찾을 수 없습니다: {filename}")
                continue
        
        if not files:
            print("❌ 테스트할 파일이 없습니다.")
            return 0.0, 0.0
        
        data = {
            'language': 'ko',
            'enable_word_timestamps': True,
            'enable_confidence': True,
            'enable_keyword_boosting': False
        }
        
        # 배치 처리 요청
        response = requests.post(url, files=files, data=data)
        
        if response.status_code != 200:
            print(f"❌ 배치 요청 실패: {response.status_code}")
            return 0.0, 0.0
        
        result = response.json()
        batch_id = result.get('batch_id')
        
        if not batch_id:
            print("❌ batch_id를 받지 못했습니다.")
            return 0.0, 0.0
        
        print(f"✅ 배치 시작: {batch_id}")
        
        # 배치 완료까지 대기하면서 진행 상황 모니터링
        status_url = f"http://localhost:8004/batch/status/{batch_id}"
        
        while True:
            time.sleep(0.5)  # 0.5초마다 상태 확인
            
            status_response = requests.get(status_url)
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data.get('status', 'unknown')
                progress = status_data.get('progress', 0.0)
                processed = status_data.get('processed_files', 0)
                total = status_data.get('total_files', 0)
                
                print(f"  📊 진행률: {progress:.1%} ({processed}/{total}) - 상태: {status}")
                
                if status == 'completed':
                    batch_end_time = time.time()
                    batch_total_time = batch_end_time - batch_start_time
                    
                    # 결과 조회
                    result_url = f"http://localhost:8004/batch/result/{batch_id}"
                    result_response = requests.get(result_url)
                    
                    if result_response.status_code == 200:
                        batch_result = result_response.json()
                        total_duration = batch_result.get('total_duration', 0)
                        total_processing_time = batch_result.get('total_processing_time', 0)
                        
                        if total_duration > 0:
                            batch_rtf = total_processing_time / total_duration
                            print(f"🎯 배치 처리 완료!")
                            print(f"  📊 총 오디오 길이: {total_duration:.1f}초")
                            print(f"  ⚡ 총 처리 시간: {total_processing_time:.3f}초")
                            print(f"  🚀 배치 RTF: {batch_rtf:.3f}")
                            print(f"  ⏱️ 전체 소요 시간: {batch_total_time:.3f}초")
                            
                            return batch_rtf, batch_total_time
                    break
                    
                elif status in ['failed', 'cancelled']:
                    print(f"❌ 배치 처리 실패: {status}")
                    break
            else:
                print(f"❌ 상태 확인 실패: {status_response.status_code}")
                break
        
    except Exception as e:
        print(f"❌ 배치 처리 오류: {e}")
    
    return 0.0, 0.0

def compare_performance():
    """성능 비교"""
    print("🧪 배치 처리 병렬 성능 테스트")
    print("=" * 60)
    
    # 1. 단일 파일 RTF 측정
    single_rtf = test_single_file_rtf()
    
    # 2. 배치 처리 RTF 측정
    batch_rtf, batch_time = test_batch_processing_rtf()
    
    # 3. 성능 비교 분석
    print("\n📊 성능 비교 결과")
    print("=" * 60)
    
    if single_rtf > 0 and batch_rtf > 0:
        print(f"📈 단일 파일 평균 RTF: {single_rtf:.3f}")
        print(f"🚀 배치 처리 RTF: {batch_rtf:.3f}")
        
        if batch_rtf <= single_rtf * 1.1:  # 오차 범위 10% 이내
            improvement = (single_rtf - batch_rtf) / single_rtf * 100
            print(f"✅ 병렬 처리 효과: RTF가 {improvement:.1f}% 개선됨")
            print("🎯 **병렬 처리가 성공적으로 적용됨!**")
        elif batch_rtf <= single_rtf * 1.5:  # 50% 이내 증가
            increase = (batch_rtf - single_rtf) / single_rtf * 100
            print(f"⚠️ 약간의 오버헤드: RTF가 {increase:.1f}% 증가")
            print("🔧 병렬 처리 중이지만 약간의 오버헤드 있음")
        else:  # 50% 이상 증가 = 순차 처리
            increase = (batch_rtf - single_rtf) / single_rtf * 100
            print(f"❌ 순차 처리 의심: RTF가 {increase:.1f}% 증가")
            print("🚨 **배치 처리가 순차적으로 실행되고 있을 가능성 높음**")
        
        print(f"\n📋 상세 분석:")
        print(f"  - 단일 파일 1개 처리 예상 시간: {single_rtf * 5:.3f}초 (5초 오디오 기준)")
        print(f"  - 배치 파일 2개 처리 실제 시간: {batch_time:.3f}초")
        print(f"  - 순차 처리 예상 시간: {single_rtf * 10:.3f}초 (2×5초 오디오)")
        print(f"  - 병렬 처리 예상 시간: {single_rtf * 5:.3f}초 (동시 처리)")
        
        efficiency = (single_rtf * 10 - batch_time) / (single_rtf * 10) * 100
        print(f"  - 전체 처리 효율성: {efficiency:.1f}%")
        
    else:
        print("❌ 성능 측정 실패")

if __name__ == "__main__":
    compare_performance() 