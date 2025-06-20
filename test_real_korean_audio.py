#!/usr/bin/env python3
"""
실제 한국어 음성 파일로 20개 클라이언트 동시 STT 테스트
디코딩 결과 정확성 검증 포함
"""

import asyncio
import aiohttp
import base64
import json
import time
import os
from typing import Dict, List, Any, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import statistics
import difflib
import Levenshtein  # pip install python-Levenshtein

# 서버 설정
SERVER_URL = "http://localhost:8001"
NUM_CLIENTS = 20

class AccuracyMetrics:
    """정확성 지표 수집 클래스"""
    def __init__(self):
        self.wer_scores = []  # Word Error Rate
        self.cer_scores = []  # Character Error Rate
        self.exact_matches = 0
        self.similarity_scores = []
        self.lock = threading.Lock()
    
    def add_result(self, expected: str, actual: str):
        with self.lock:
            # 정확도 계산
            wer = self.calculate_wer(expected, actual)
            cer = self.calculate_cer(expected, actual)
            similarity = self.calculate_similarity(expected, actual)
            
            self.wer_scores.append(wer)
            self.cer_scores.append(cer)
            self.similarity_scores.append(similarity)
            
            if expected.strip().lower() == actual.strip().lower():
                self.exact_matches += 1
    
    def calculate_wer(self, expected: str, actual: str) -> float:
        """단어 오류율 계산"""
        expected_words = expected.strip().split()
        actual_words = actual.strip().split()
        
        if len(expected_words) == 0:
            return 0.0 if len(actual_words) == 0 else 1.0
        
        distance = Levenshtein.distance(expected_words, actual_words)
        return distance / len(expected_words)
    
    def calculate_cer(self, expected: str, actual: str) -> float:
        """문자 오류율 계산"""
        if len(expected) == 0:
            return 0.0 if len(actual) == 0 else 1.0
        
        distance = Levenshtein.distance(expected, actual)
        return distance / len(expected)
    
    def calculate_similarity(self, expected: str, actual: str) -> float:
        """유사도 계산 (0~1)"""
        return difflib.SequenceMatcher(None, expected, actual).ratio()
    
    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            if not self.wer_scores:
                return {}
            
            return {
                "total_samples": len(self.wer_scores),
                "exact_matches": self.exact_matches,
                "exact_match_rate": self.exact_matches / len(self.wer_scores) * 100,
                "avg_wer": statistics.mean(self.wer_scores),
                "avg_cer": statistics.mean(self.cer_scores),
                "avg_similarity": statistics.mean(self.similarity_scores),
                "min_similarity": min(self.similarity_scores),
                "max_similarity": max(self.similarity_scores)
            }

class PerformanceMetrics:
    """성능 지표 수집 클래스"""
    def __init__(self):
        self.request_times = []
        self.processing_times = []
        self.wait_times = []
        self.rtf_values = []
        self.success_count = 0
        self.failure_count = 0
        self.total_requests = 0
        self.start_time = None
        self.end_time = None
        self.lock = threading.Lock()
    
    def add_result(self, success: bool, request_time: float = 0, 
                   processing_time: float = 0, wait_time: float = 0, rtf: float = 0):
        with self.lock:
            self.total_requests += 1
            if success:
                self.success_count += 1
                self.request_times.append(request_time)
                self.processing_times.append(processing_time)
                self.wait_times.append(wait_time)
                self.rtf_values.append(rtf)
            else:
                self.failure_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            duration = (self.end_time - self.start_time) if self.start_time and self.end_time else 0
            
            return {
                "duration_seconds": duration,
                "total_requests": self.total_requests,
                "successful_requests": self.success_count,
                "failed_requests": self.failure_count,
                "success_rate": (self.success_count / self.total_requests * 100) if self.total_requests > 0 else 0,
                "throughput_rps": self.success_count / duration if duration > 0 else 0,
                "avg_request_time": statistics.mean(self.request_times) if self.request_times else 0,
                "median_request_time": statistics.median(self.request_times) if self.request_times else 0,
                "avg_processing_time": statistics.mean(self.processing_times) if self.processing_times else 0,
                "avg_wait_time": statistics.mean(self.wait_times) if self.wait_times else 0,
                "avg_rtf": statistics.mean(self.rtf_values) if self.rtf_values else 0,
                "median_rtf": statistics.median(self.rtf_values) if self.rtf_values else 0
            }

def load_korean_audio_samples() -> List[Tuple[str, str, str]]:
    """실제 한국어 오디오 샘플과 정답 텍스트 로드"""
    samples = []
    
    # 샘플 1
    if os.path.exists("test_korean_sample1.wav") and os.path.exists("test_korean_sample1.txt"):
        with open("test_korean_sample1.wav", "rb") as f:
            audio_data1 = base64.b64encode(f.read()).decode('utf-8')
        with open("test_korean_sample1.txt", "r", encoding="utf-8") as f:
            expected_text1 = f.read().strip()
        samples.append(("sample1", audio_data1, expected_text1))
    
    # 샘플 2
    if os.path.exists("test_korean_sample2.wav") and os.path.exists("test_korean_sample2.txt"):
        with open("test_korean_sample2.wav", "rb") as f:
            audio_data2 = base64.b64encode(f.read()).decode('utf-8')
        with open("test_korean_sample2.txt", "r", encoding="utf-8") as f:
            expected_text2 = f.read().strip()
        samples.append(("sample2", audio_data2, expected_text2))
    
    return samples

async def single_client_test_real_audio(session: aiohttp.ClientSession, 
                                      client_id: int, 
                                      performance_metrics: PerformanceMetrics,
                                      accuracy_metrics: AccuracyMetrics,
                                      audio_samples: List[Tuple[str, str, str]]) -> None:
    """단일 클라이언트 실제 오디오 테스트"""
    
    for i, (sample_name, audio_data, expected_text) in enumerate(audio_samples):
        request_start = time.time()
        
        try:
            # 요청 제출
            queue_request = {
                "audio_data": audio_data,
                "language": "ko",
                "client_id": f"client_{client_id:02d}",
                "priority": "medium"
            }
            
            async with session.post(f"{SERVER_URL}/queue/transcribe", 
                                  json=queue_request) as response:
                if response.status == 200:
                    queue_response = await response.json()
                    request_id = queue_response["request_id"]
                    
                    # 결과 대기
                    max_wait = 30  # 30초 최대 대기
                    wait_start = time.time()
                    
                    while time.time() - wait_start < max_wait:
                        try:
                            async with session.get(f"{SERVER_URL}/queue/result/{request_id}") as result_response:
                                if result_response.status == 200:
                                    result = await result_response.json()
                                    request_end = time.time()
                                    
                                    # 성능 지표 수집
                                    request_time = request_end - request_start
                                    wait_time = time.time() - wait_start
                                    processing_time = result.get("processing_time", 0)
                                    rtf = result.get("rtf", 0)
                                    actual_text = result.get("text", "")
                                    
                                    performance_metrics.add_result(
                                        success=True,
                                        request_time=request_time,
                                        processing_time=processing_time,
                                        wait_time=wait_time,
                                        rtf=rtf
                                    )
                                    
                                    # 정확성 지표 수집
                                    accuracy_metrics.add_result(expected_text, actual_text)
                                    
                                    # 유사도 계산
                                    similarity = accuracy_metrics.calculate_similarity(expected_text, actual_text)
                                    
                                    print(f"✅ 클라이언트 {client_id:02d}-{sample_name}: "
                                          f"RTF={rtf:.3f}, 유사도={similarity:.3f}")
                                    print(f"   예상: '{expected_text}'")
                                    print(f"   실제: '{actual_text}'")
                                    
                                    break
                                    
                        except Exception as e:
                            await asyncio.sleep(0.1)
                    else:
                        # 타임아웃
                        performance_metrics.add_result(success=False)
                        print(f"⏰ 클라이언트 {client_id:02d}-{sample_name}: 타임아웃")
                        
                else:
                    performance_metrics.add_result(success=False)
                    print(f"❌ 클라이언트 {client_id:02d}-{sample_name}: 요청 실패 ({response.status})")
                    
        except Exception as e:
            performance_metrics.add_result(success=False)
            print(f"❌ 클라이언트 {client_id:02d}-{sample_name}: 예외 - {e}")
        
        # 클라이언트 간 요청 간격
        await asyncio.sleep(0.2)

def monitor_system_resources(duration: int, interval: float = 1.0) -> Dict[str, List]:
    """시스템 리소스 모니터링"""
    cpu_usage = []
    memory_usage = []
    
    start_time = time.time()
    while time.time() - start_time < duration:
        cpu_usage.append(psutil.cpu_percent())
        memory_usage.append(psutil.virtual_memory().percent)
        time.sleep(interval)
    
    return {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "avg_cpu": statistics.mean(cpu_usage),
        "max_cpu": max(cpu_usage),
        "avg_memory": statistics.mean(memory_usage),
        "max_memory": max(memory_usage)
    }

async def test_real_korean_audio_20_clients():
    """실제 한국어 음성으로 20개 클라이언트 테스트"""
    print("🎤 실제 한국어 음성 20개 클라이언트 동시 STT 테스트")
    print("=" * 60)
    
    # 한국어 오디오 샘플 로드
    audio_samples = load_korean_audio_samples()
    if not audio_samples:
        print("❌ 한국어 오디오 샘플을 찾을 수 없습니다!")
        return
    
    print(f"📊 테스트 설정:")
    print(f"   클라이언트 수: {NUM_CLIENTS}개")
    print(f"   오디오 샘플: {len(audio_samples)}개")
    print(f"   총 예상 요청: {NUM_CLIENTS * len(audio_samples)}개")
    
    for i, (sample_name, _, expected_text) in enumerate(audio_samples):
        print(f"   샘플 {i+1} ({sample_name}): '{expected_text}'")
    
    # 지표 수집기
    performance_metrics = PerformanceMetrics()
    accuracy_metrics = AccuracyMetrics()
    
    # 1. 서버 상태 확인
    print("\n1️⃣ 서버 상태 확인:")
    connector = aiohttp.TCPConnector(limit=50, limit_per_host=30)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        try:
            async with session.get(f"{SERVER_URL}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"   ✅ 서버: {health['status']}")
                    print(f"   🔥 GPU: {health['gpu_name']}")
                    print(f"   🚀 cuDNN: {health['cudnn_enabled']}")
                else:
                    print("   ❌ 서버 상태 불량")
                    return
        except Exception as e:
            print(f"   ❌ 서버 연결 실패: {e}")
            return
        
        # 2. 초기 큐 상태
        print("\n2️⃣ 초기 큐 상태:")
        try:
            async with session.get(f"{SERVER_URL}/queue/stats") as response:
                stats = await response.json()
                print(f"   완료: {stats['completed_requests']}, "
                      f"실패: {stats['failed_requests']}, "
                      f"최대 동시처리: {stats['max_concurrent']}")
        except Exception as e:
            print(f"   ❌ 큐 상태 조회 실패: {e}")
        
        # 3. 동시 테스트 실행
        print(f"\n3️⃣ {NUM_CLIENTS}개 클라이언트 실제 음성 테스트:")
        print("   (디코딩 결과 실시간 표시)")
        
        # 시스템 리소스 모니터링 시작
        estimated_duration = len(audio_samples) * 3
        resource_future = asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), 
            monitor_system_resources, 
            estimated_duration + 10
        )
        
        # 테스트 시작
        performance_metrics.start_time = time.time()
        
        # 20개 클라이언트 동시 실행
        client_tasks = []
        for client_id in range(NUM_CLIENTS):
            task = asyncio.create_task(
                single_client_test_real_audio(session, client_id, performance_metrics, 
                                            accuracy_metrics, audio_samples)
            )
            client_tasks.append(task)
        
        # 모든 클라이언트 테스트 완료 대기
        await asyncio.gather(*client_tasks)
        
        performance_metrics.end_time = time.time()
        
        # 리소스 모니터링 중지
        try:
            resource_stats = await resource_future
        except:
            resource_stats = {"avg_cpu": 0, "max_cpu": 0, "avg_memory": 0, "max_memory": 0}
        
        # 4. 최종 큐 상태
        print("\n4️⃣ 최종 큐 상태:")
        try:
            async with session.get(f"{SERVER_URL}/queue/stats") as response:
                final_stats = await response.json()
                print(f"   📊 총 요청: {final_stats['total_requests']}개")
                print(f"   ✅ 완료: {final_stats['completed_requests']}개")
                print(f"   ❌ 실패: {final_stats['failed_requests']}개")
        except Exception as e:
            print(f"   ❌ 최종 통계 조회 실패: {e}")

    # 5. 성능 분석 결과
    print("\n5️⃣ 성능 분석 결과:")
    perf_summary = performance_metrics.get_summary()
    acc_summary = accuracy_metrics.get_summary()
    
    print("=" * 60)
    print("📊 전체 성능 요약:")
    print(f"   🕐 테스트 시간: {perf_summary['duration_seconds']:.1f}초")
    print(f"   📝 총 요청수: {perf_summary['total_requests']}개")
    print(f"   ✅ 성공: {perf_summary['successful_requests']}개 ({perf_summary['success_rate']:.1f}%)")
    print(f"   ❌ 실패: {perf_summary['failed_requests']}개")
    print(f"   ⚡ 처리량: {perf_summary['throughput_rps']:.2f} 요청/초")
    
    print("\n🔄 처리 성능:")
    print(f"   평균 처리시간: {perf_summary['avg_processing_time']:.3f}초")
    print(f"   평균 대기시간: {perf_summary['avg_wait_time']:.3f}초")
    print(f"   평균 RTF: {perf_summary['avg_rtf']:.3f}x")
    print(f"   중간값 RTF: {perf_summary['median_rtf']:.3f}x")
    
    if acc_summary:
        print("\n🎯 디코딩 정확성 분석:")
        print(f"   정확한 일치: {acc_summary['exact_matches']}/{acc_summary['total_samples']}개 ({acc_summary['exact_match_rate']:.1f}%)")
        print(f"   평균 유사도: {acc_summary['avg_similarity']:.3f} (0~1)")
        print(f"   평균 WER: {acc_summary['avg_wer']:.3f} (낮을수록 좋음)")
        print(f"   평균 CER: {acc_summary['avg_cer']:.3f} (낮을수록 좋음)")
        print(f"   유사도 범위: {acc_summary['min_similarity']:.3f} ~ {acc_summary['max_similarity']:.3f}")
    
    print("\n💻 시스템 리소스:")
    print(f"   평균 CPU: {resource_stats['avg_cpu']:.1f}%")
    print(f"   최대 CPU: {resource_stats['max_cpu']:.1f}%")
    print(f"   평균 메모리: {resource_stats['avg_memory']:.1f}%")
    print(f"   최대 메모리: {resource_stats['max_memory']:.1f}%")
    
    # 6. 정확성 등급 평가
    if acc_summary:
        print("\n🏆 디코딩 품질 등급:")
        
        # 유사도 기준 평가
        similarity_grade = "🔴 D"
        avg_sim = acc_summary['avg_similarity']
        if avg_sim >= 0.95:
            similarity_grade = "🟢 S+"
        elif avg_sim >= 0.90:
            similarity_grade = "🟢 S"
        elif avg_sim >= 0.85:
            similarity_grade = "🟡 A"
        elif avg_sim >= 0.80:
            similarity_grade = "🟠 B"
        elif avg_sim >= 0.70:
            similarity_grade = "🔴 C"
        
        # WER 기준 평가
        wer_grade = "🔴 D"
        avg_wer = acc_summary['avg_wer']
        if avg_wer <= 0.05:
            wer_grade = "🟢 S+"
        elif avg_wer <= 0.10:
            wer_grade = "🟢 S"
        elif avg_wer <= 0.20:
            wer_grade = "🟡 A"
        elif avg_wer <= 0.30:
            wer_grade = "🟠 B"
        elif avg_wer <= 0.50:
            wer_grade = "🔴 C"
        
        print(f"   유사도: {similarity_grade} ({avg_sim:.3f})")
        print(f"   WER: {wer_grade} ({avg_wer:.3f})")
        print(f"   정확 일치: {acc_summary['exact_match_rate']:.1f}%")
    
    print("\n" + "=" * 60)
    print("🎯 실제 한국어 음성 테스트 완료!")
    
    return perf_summary, acc_summary

if __name__ == "__main__":
    # 의존성 설치 확인
    try:
        import Levenshtein
    except ImportError:
        print("❌ python-Levenshtein 패키지가 필요합니다.")
        print("   설치: pip install python-Levenshtein")
        exit(1)
    
    asyncio.run(test_real_korean_audio_20_clients()) 