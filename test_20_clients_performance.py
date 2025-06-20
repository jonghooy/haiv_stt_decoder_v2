#!/usr/bin/env python3
"""
20개 동시 클라이언트 STT 성능 테스트
현실적인 시나리오로 다양한 오디오 길이와 우선순위 테스트
"""

import asyncio
import aiohttp
import base64
import json
import time
import numpy as np
from typing import Dict, List, Any
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import statistics

# 서버 설정
SERVER_URL = "http://localhost:8001"
NUM_CLIENTS = 20
TEST_DURATION = 60  # 60초간 테스트

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
                "p95_request_time": np.percentile(self.request_times, 95) if self.request_times else 0,
                "p99_request_time": np.percentile(self.request_times, 99) if self.request_times else 0,
                "avg_processing_time": statistics.mean(self.processing_times) if self.processing_times else 0,
                "avg_wait_time": statistics.mean(self.wait_times) if self.wait_times else 0,
                "avg_rtf": statistics.mean(self.rtf_values) if self.rtf_values else 0,
                "median_rtf": statistics.median(self.rtf_values) if self.rtf_values else 0
            }

def create_test_audio(duration_seconds: float = 1.0, frequency: int = 440) -> str:
    """다양한 길이의 테스트 오디오 생성"""
    sample_rate = 16000
    samples = int(sample_rate * duration_seconds)
    
    # 사인파 생성
    t = np.linspace(0, duration_seconds, samples, False)
    audio = np.sin(2 * np.pi * frequency * t)
    
    # 16비트 PCM으로 변환
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    
    return base64.b64encode(audio_bytes).decode('utf-8')

async def single_client_test(session: aiohttp.ClientSession, 
                           client_id: int, 
                           metrics: PerformanceMetrics,
                           test_configs: List[Dict]) -> None:
    """단일 클라이언트 테스트 실행"""
    
    for i, config in enumerate(test_configs):
        request_start = time.time()
        
        try:
            # 요청 제출
            queue_request = {
                "audio_data": config["audio_data"],
                "language": "ko",
                "client_id": f"client_{client_id:02d}",
                "priority": config["priority"]
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
                                    
                                    request_time = request_end - request_start
                                    wait_time = time.time() - wait_start
                                    processing_time = result.get("processing_time", 0)
                                    rtf = result.get("rtf", 0)
                                    
                                    metrics.add_result(
                                        success=True,
                                        request_time=request_time,
                                        processing_time=processing_time,
                                        wait_time=wait_time,
                                        rtf=rtf
                                    )
                                    
                                    print(f"✅ 클라이언트 {client_id:02d}-{i+1:02d}: "
                                          f"총시간={request_time:.3f}s, "
                                          f"처리={processing_time:.3f}s, "
                                          f"RTF={rtf:.3f}, "
                                          f"우선순위={config['priority']}")
                                    break
                                    
                        except Exception as e:
                            await asyncio.sleep(0.1)
                    else:
                        # 타임아웃
                        metrics.add_result(success=False)
                        print(f"⏰ 클라이언트 {client_id:02d}-{i+1:02d}: 타임아웃")
                        
                else:
                    metrics.add_result(success=False)
                    print(f"❌ 클라이언트 {client_id:02d}-{i+1:02d}: 요청 실패 ({response.status})")
                    
        except Exception as e:
            metrics.add_result(success=False)
            print(f"❌ 클라이언트 {client_id:02d}-{i+1:02d}: 예외 - {e}")
        
        # 클라이언트 간 요청 간격 (현실적인 시뮬레이션)
        await asyncio.sleep(np.random.uniform(0.1, 0.5))

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

async def test_queue_stats_monitoring(session: aiohttp.ClientSession, duration: int):
    """큐 통계 모니터링"""
    stats_history = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            async with session.get(f"{SERVER_URL}/queue/stats") as response:
                if response.status == 200:
                    stats = await response.json()
                    stats["timestamp"] = time.time() - start_time
                    stats_history.append(stats)
        except Exception as e:
            pass
        
        await asyncio.sleep(2)  # 2초마다 통계 수집
    
    return stats_history

async def run_20_client_performance_test():
    """20개 클라이언트 동시 성능 테스트 실행"""
    print("🚀 20개 클라이언트 동시 STT 성능 테스트 시작")
    print("=" * 60)
    
    # 테스트 설정 생성 (다양한 시나리오)
    test_scenarios = []
    audio_durations = [0.5, 1.0, 2.0, 3.0, 5.0]  # 다양한 오디오 길이
    priorities = ["high", "medium", "low"]
    frequencies = [220, 440, 880, 1320]  # 다양한 주파수
    
    for i in range(5):  # 클라이언트당 5개 요청
        duration = np.random.choice(audio_durations)
        priority = np.random.choice(priorities)
        frequency = np.random.choice(frequencies)
        
        test_scenarios.append({
            "audio_data": create_test_audio(duration, frequency),
            "priority": priority,
            "duration": duration,
            "frequency": frequency
        })
    
    print(f"📊 테스트 시나리오:")
    print(f"   클라이언트 수: {NUM_CLIENTS}개")
    print(f"   클라이언트당 요청: {len(test_scenarios)}개")
    print(f"   총 예상 요청: {NUM_CLIENTS * len(test_scenarios)}개")
    print(f"   오디오 길이: {min(audio_durations)}-{max(audio_durations)}초")
    print(f"   우선순위: {', '.join(priorities)}")
    
    # 성능 지표 수집기
    metrics = PerformanceMetrics()
    
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
        
        # 2. 초기 큐 상태 확인
        print("\n2️⃣ 초기 큐 상태:")
        try:
            async with session.get(f"{SERVER_URL}/queue/stats") as response:
                stats = await response.json()
                print(f"   대기: {stats['queued_requests']}, "
                      f"처리중: {stats['processing_requests']}, "
                      f"완료: {stats['completed_requests']}, "
                      f"실패: {stats['failed_requests']}")
                print(f"   최대 동시처리: {stats['max_concurrent']}, "
                      f"큐 용량: {stats['queue_capacity']}")
        except Exception as e:
            print(f"   ❌ 큐 상태 조회 실패: {e}")
        
        # 3. 동시 테스트 실행
        print(f"\n3️⃣ {NUM_CLIENTS}개 클라이언트 동시 테스트 실행:")
        print("   (실시간 결과 표시)")
        
        # 시스템 리소스 모니터링 시작
        estimated_duration = len(test_scenarios) * 2  # 예상 테스트 시간
        resource_future = asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), 
            monitor_system_resources, 
            estimated_duration + 10
        )
        
        # 큐 통계 모니터링 시작
        stats_future = asyncio.create_task(
            test_queue_stats_monitoring(session, estimated_duration + 10)
        )
        
        # 테스트 시작
        metrics.start_time = time.time()
        
        # 20개 클라이언트 동시 실행
        client_tasks = []
        for client_id in range(NUM_CLIENTS):
            task = asyncio.create_task(
                single_client_test(session, client_id, metrics, test_scenarios)
            )
            client_tasks.append(task)
        
        # 모든 클라이언트 테스트 완료 대기
        await asyncio.gather(*client_tasks)
        
        metrics.end_time = time.time()
        
        # 리소스 모니터링 중지
        try:
            resource_stats = await resource_future
        except:
            resource_stats = {"avg_cpu": 0, "max_cpu": 0, "avg_memory": 0, "max_memory": 0}
        
        # 큐 통계 모니터링 중지
        stats_future.cancel()
        try:
            queue_stats_history = await stats_future
        except:
            queue_stats_history = []
        
        # 4. 최종 큐 상태 확인
        print("\n4️⃣ 최종 큐 상태:")
        try:
            async with session.get(f"{SERVER_URL}/queue/stats") as response:
                final_stats = await response.json()
                print(f"   📊 총 요청: {final_stats['total_requests']}개")
                print(f"   ✅ 완료: {final_stats['completed_requests']}개")
                print(f"   ❌ 실패: {final_stats['failed_requests']}개")
                print(f"   ⏳ 대기중: {final_stats['queued_requests']}개")
                print(f"   🔄 처리중: {final_stats['processing_requests']}개")
                print(f"   ⚡ 처리량: {final_stats['current_throughput']:.2f} 요청/분")
        except Exception as e:
            print(f"   ❌ 최종 통계 조회 실패: {e}")

    # 5. 성능 분석 결과
    print("\n5️⃣ 성능 분석 결과:")
    summary = metrics.get_summary()
    
    print("=" * 60)
    print("📊 전체 성능 요약:")
    print(f"   🕐 테스트 시간: {summary['duration_seconds']:.1f}초")
    print(f"   📝 총 요청수: {summary['total_requests']}개")
    print(f"   ✅ 성공: {summary['successful_requests']}개 ({summary['success_rate']:.1f}%)")
    print(f"   ❌ 실패: {summary['failed_requests']}개")
    print(f"   ⚡ 처리량: {summary['throughput_rps']:.2f} 요청/초")
    
    print("\n⏱️ 응답 시간 분석:")
    print(f"   평균: {summary['avg_request_time']:.3f}초")
    print(f"   중간값: {summary['median_request_time']:.3f}초")
    print(f"   95%: {summary['p95_request_time']:.3f}초")
    print(f"   99%: {summary['p99_request_time']:.3f}초")
    
    print("\n🔄 처리 성능:")
    print(f"   평균 처리시간: {summary['avg_processing_time']:.3f}초")
    print(f"   평균 대기시간: {summary['avg_wait_time']:.3f}초")
    print(f"   평균 RTF: {summary['avg_rtf']:.3f}x")
    print(f"   중간값 RTF: {summary['median_rtf']:.3f}x")
    
    print("\n💻 시스템 리소스:")
    print(f"   평균 CPU: {resource_stats['avg_cpu']:.1f}%")
    print(f"   최대 CPU: {resource_stats['max_cpu']:.1f}%")
    print(f"   평균 메모리: {resource_stats['avg_memory']:.1f}%")
    print(f"   최대 메모리: {resource_stats['max_memory']:.1f}%")
    
    # 6. 성능 등급 평가
    print("\n🏆 성능 등급 평가:")
    
    # RTF 기준 평가
    rtf_grade = "🔴 D"
    if summary['avg_rtf'] < 0.05:
        rtf_grade = "🟢 S+"
    elif summary['avg_rtf'] < 0.1:
        rtf_grade = "🟢 S"
    elif summary['avg_rtf'] < 0.2:
        rtf_grade = "🟡 A"
    elif summary['avg_rtf'] < 0.5:
        rtf_grade = "🟠 B"
    elif summary['avg_rtf'] < 1.0:
        rtf_grade = "🔴 C"
    
    # 성공률 기준 평가
    success_grade = "🔴 D"
    if summary['success_rate'] >= 99:
        success_grade = "🟢 S+"
    elif summary['success_rate'] >= 95:
        success_grade = "🟢 S"
    elif summary['success_rate'] >= 90:
        success_grade = "🟡 A"
    elif summary['success_rate'] >= 80:
        success_grade = "🟠 B"
    elif summary['success_rate'] >= 70:
        success_grade = "🔴 C"
    
    # 처리량 기준 평가 (20개 클라이언트 기준)
    throughput_grade = "🔴 D"
    if summary['throughput_rps'] >= 15:
        throughput_grade = "🟢 S+"
    elif summary['throughput_rps'] >= 10:
        throughput_grade = "🟢 S"
    elif summary['throughput_rps'] >= 7:
        throughput_grade = "🟡 A"
    elif summary['throughput_rps'] >= 5:
        throughput_grade = "🟠 B"
    elif summary['throughput_rps'] >= 3:
        throughput_grade = "🔴 C"
    
    print(f"   RTF 성능: {rtf_grade} (평균 {summary['avg_rtf']:.3f}x)")
    print(f"   안정성: {success_grade} ({summary['success_rate']:.1f}% 성공)")
    print(f"   처리량: {throughput_grade} ({summary['throughput_rps']:.2f} 요청/초)")
    
    print("\n" + "=" * 60)
    print("🎯 20개 클라이언트 동시 성능 테스트 완료!")
    
    return summary

if __name__ == "__main__":
    asyncio.run(run_20_client_performance_test()) 