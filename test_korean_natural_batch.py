#!/usr/bin/env python3
"""
배치 STT 처리 테스트 프로그램
사용법: python test_korean_natural_batch.py [파일명] --port [포트번호]
"""

import requests
import time
import os
import zipfile
import json
import argparse
from pathlib import Path

class KoreanNaturalBatchTest:
    def __init__(self, base_url="http://localhost:8004"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def submit_batch(self, audio_file, language="ko", vad_filter=False, 
                    enable_word_timestamps=True, enable_confidence=True,
                    priority="medium"):
        """배치 처리 제출"""
        url = f"{self.base_url}/batch/transcribe"
        
        if not os.path.exists(audio_file):
            print(f"❌ 파일을 찾을 수 없습니다: {audio_file}")
            return None
        
        # 파일 정보 출력
        file_size = os.path.getsize(audio_file)
        print(f"📁 파일: {audio_file}")
        print(f"📊 크기: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        try:
            # 파일을 multipart/form-data로 준비
            with open(audio_file, 'rb') as f:
                files = [('files', (os.path.basename(audio_file), f, 'audio/wav'))]
                
                data = {
                    'language': language,
                    'vad_filter': vad_filter,
                    'enable_word_timestamps': enable_word_timestamps,
                    'enable_confidence': enable_confidence,
                    'priority': priority
                }
                
                print(f"📤 배치 제출 중... (파일: {os.path.basename(audio_file)})")
                response = self.session.post(url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                batch_id = result.get('batch_id')
                print(f"✅ 배치 제출 성공: {batch_id}")
                return batch_id
            else:
                print(f"❌ 배치 제출 실패: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   오류 세부사항: {error_detail}")
                except:
                    print(f"   응답 내용: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"❌ 요청 중 오류: {e}")
            return None
    
    def monitor_progress(self, batch_id, max_wait_time=600):
        """배치 진행률 모니터링"""
        url = f"{self.base_url}/batch/status/{batch_id}"
        start_time = time.time()
        
        print(f"⏳ 배치 처리 완료까지 대기 중... (최대 {max_wait_time}초)")
        
        while time.time() - start_time < max_wait_time:
            try:
                response = self.session.get(url)
                
                if response.status_code == 200:
                    status = response.json()
                    progress = status.get('progress', 0) * 100
                    processed = status.get('processed_files', 0)
                    total = status.get('total_files', 0)
                    current_status = status.get('status', 'unknown')
                    
                    print(f"🔄 진행률: {progress:.1f}% ({processed}/{total}) - 상태: {current_status}")
                    
                    if current_status == 'completed':
                        print("✅ 배치 처리 완료!")
                        return True
                    elif current_status == 'failed':
                        error_msg = status.get('error_message', '알 수 없는 오류')
                        print(f"❌ 배치 처리 실패: {error_msg}")
                        return False
                else:
                    print(f"⚠️ 상태 조회 실패: {response.status_code}")
                
            except Exception as e:
                print(f"⚠️ 상태 조회 중 오류: {e}")
            
            time.sleep(3)  # 3초마다 확인
        
        print("⏰ 대기 시간 초과")
        return False
    
    def download_results(self, batch_id, save_dir="batch_results"):
        """결과 다운로드"""
        url = f"{self.base_url}/batch/download/{batch_id}"
        
        try:
            response = self.session.get(url)
            
            if response.status_code == 200:
                # 저장 디렉토리 생성
                os.makedirs(save_dir, exist_ok=True)
                
                # ZIP 파일 저장
                zip_path = os.path.join(save_dir, f"batch_results_{batch_id}.zip")
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"📥 결과 다운로드 완료: {zip_path}")
                
                # 압축 해제
                extract_path = os.path.join(save_dir, f"extracted_{batch_id}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                
                print(f"📂 압축 해제 완료: {extract_path}")
                
                return zip_path, extract_path
            else:
                print(f"❌ 다운로드 실패: {response.status_code}")
                return None, None
                
        except Exception as e:
            print(f"❌ 다운로드 중 오류: {e}")
            return None, None
    
    def analyze_results(self, extract_path):
        """결과 분석"""
        if not extract_path or not os.path.exists(extract_path):
            print("❌ 결과 경로가 존재하지 않습니다.")
            return
        
        # JSON 결과 파일 읽기
        json_path = os.path.join(extract_path, "batch_results.json")
        if not os.path.exists(json_path):
            print("❌ 결과 JSON 파일을 찾을 수 없습니다.")
            return
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            print("\n" + "="*60)
            print("📊 배치 처리 결과 분석")
            print("="*60)
            print(f"배치 ID: {results.get('batch_id')}")
            print(f"총 파일 수: {results.get('total_files')}")
            print(f"처리 완료: {results.get('processed_files')}")
            print(f"처리 실패: {results.get('failed_files')}")
            print(f"총 오디오 길이: {results.get('total_duration', 0):.2f}초")
            print(f"총 처리 시간: {results.get('total_processing_time', 0):.2f}초")
            print(f"시작 시간: {results.get('created_at')}")
            print(f"완료 시간: {results.get('completed_at')}")
            
            # 파일별 결과
            files = results.get('files', [])
            if files:
                print(f"\n📝 파일별 결과:")
                print("-" * 60)
                
                for file_info in files:
                    filename = file_info.get('filename', 'unknown')
                    size_bytes = file_info.get('size_bytes', 0)
                    duration = file_info.get('duration_seconds', 0)
                    processing_time = file_info.get('processing_time_seconds', 0)
                    text = file_info.get('text', '')
                    language = file_info.get('language', 'unknown')
                    confidence = file_info.get('confidence', 0)
                    
                    print(f"\n🎵 {filename}")
                    print(f"   크기: {size_bytes:,} bytes ({size_bytes/1024/1024:.2f} MB)")
                    print(f"   길이: {duration:.2f}초")
                    print(f"   처리시간: {processing_time:.2f}초")
                    print(f"   RTF: {processing_time/duration:.3f}x" if duration > 0 else "   RTF: N/A")
                    print(f"   언어: {language}")
                    print(f"   신뢰도: {confidence:.3f}")
                    print(f"   📄 전사 결과: {text[:100]}{'...' if len(text) > 100 else ''}")
                    
                    # 세그먼트 정보
                    segments = file_info.get('segments', [])
                    if segments:
                        print(f"   🔍 세그먼트 수: {len(segments)}개")
                        
                        # 처음 3개 세그먼트만 출력
                        for i, segment in enumerate(segments[:3]):
                            seg_text = segment.get('text', '')
                            seg_start = segment.get('start', 0)
                            seg_end = segment.get('end', 0)
                            seg_confidence = segment.get('confidence', 0)
                            
                            # confidence가 None이면 기본값 사용
                            if seg_confidence is None:
                                seg_confidence = 0.0
                            
                            print(f"      [{i+1}] {seg_start:.1f}s-{seg_end:.1f}s: {seg_text} (신뢰도: {seg_confidence:.3f})")
                        
                        if len(segments) > 3:
                            print(f"      ... (총 {len(segments)}개 세그먼트)")
            
            # 텍스트 파일 확인
            transcripts_dir = os.path.join(extract_path, "transcripts")
            if os.path.exists(transcripts_dir):
                txt_files = [f for f in os.listdir(transcripts_dir) if f.endswith('.txt')]
                if txt_files:
                    print(f"\n📁 생성된 텍스트 파일: {len(txt_files)}개")
                    for txt_file in txt_files:
                        txt_path = os.path.join(transcripts_dir, txt_file)
                        if os.path.exists(txt_path):
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                            print(f"   📄 {txt_file}: {len(content)}자")
                            print(f"      내용: {content[:100]}{'...' if len(content) > 100 else ''}")
            
        except Exception as e:
            print(f"❌ 결과 분석 중 오류: {e}")

def main():
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(
        description="배치 STT 처리 테스트 프로그램",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python test_korean_natural_batch.py                          # 기본 파일과 포트 사용
  python test_korean_natural_batch.py audio.wav               # 특정 파일 사용
  python test_korean_natural_batch.py --port 8001             # 특정 포트 사용  
  python test_korean_natural_batch.py audio.wav --port 8001   # 파일과 포트 모두 지정
        """
    )
    
    parser.add_argument(
        'audio_file', 
        nargs='?', 
        default='korean_natural_1min.wav',
        help='테스트할 오디오 파일 경로 (기본값: korean_natural_1min.wav)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8004,
        help='STT 서버 포트 번호 (기본값: 8004)'
    )
    
    parser.add_argument(
        '--language', '-l',
        default='ko',
        help='음성 언어 코드 (기본값: ko)'
    )
    
    parser.add_argument(
        '--priority',
        default='medium',
        choices=['high', 'medium', 'low'],
        help='처리 우선순위 (기본값: medium)'
    )
    
    parser.add_argument(
        '--max-wait',
        type=int,
        default=300,
        help='최대 대기 시간(초) (기본값: 300)'
    )
    
    args = parser.parse_args()
    
    # 서버 URL 구성
    base_url = f"http://localhost:{args.port}"
    
    # 오디오 파일 확인
    if not os.path.exists(args.audio_file):
        print(f"❌ 테스트 파일을 찾을 수 없습니다: {args.audio_file}")
        print("현재 디렉토리의 오디오 파일들:")
        audio_files = []
        for f in os.listdir('.'):
            if f.endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
                audio_files.append(f)
                print(f"  - {f}")
        
        if not audio_files:
            print("  (오디오 파일이 없습니다)")
        return
    
    # 설정 정보 출력
    print(f"🎯 배치 STT 처리 테스트")
    print(f"📁 파일: {args.audio_file}")
    print(f"🌐 서버: {base_url}")
    print(f"🗣️ 언어: {args.language}")
    print(f"⚡ 우선순위: {args.priority}")
    print(f"⏰ 최대 대기: {args.max_wait}초")
    print("-" * 50)
    
    # 배치 처리 테스트 시작
    tester = KoreanNaturalBatchTest(base_url=base_url)
    
    # 1. 배치 제출
    batch_id = tester.submit_batch(
        args.audio_file, 
        language=args.language,
        priority=args.priority
    )
    if not batch_id:
        return
    
    # 2. 진행률 모니터링
    success = tester.monitor_progress(batch_id, max_wait_time=args.max_wait)
    if not success:
        return
    
    # 3. 결과 다운로드
    zip_path, extract_path = tester.download_results(batch_id)
    if not zip_path:
        return
    
    # 4. 결과 분석
    tester.analyze_results(extract_path)
    
    print(f"\n🎉 배치 처리 성공!")
    print(f"📁 결과 파일: {zip_path}")
    print(f"📂 압축 해제: {extract_path}")

if __name__ == "__main__":
    main() 