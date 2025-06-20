#!/usr/bin/env python3
"""
🔄 다중 클라이언트 동시성 성능 테스트
여러 워커가 동시에 STT 디코더에 요청할 때의 처리 성능 확인
"""

import asyncio
import aiohttp
import time
import json
import numpy as np
import base64
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import statistics
import concurrent.futures
from threading import Lock
import psutil
import os

@dataclass
class ConcurrentTestResult:
    """동시성 테스트 결과"""
    client_id: int
    audio_duration: float
    processing_time: float
    wait_time: float  # 요청 대기 시간
    total_time: float  # 전체 소요 시간
    rtf: float
    text: str
    success: bool
    error_message: str = ""
    timestamp: float = 0.0

class ConcurrentSTTTester:
    """동시성 STT 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8001"):
        self.server_url = server_url
        self.results: List[ConcurrentTestResult] = []
        self.results_lock = Lock()
        self.start_time = None
    
    def generate_test_audio(self, duration_seconds: float, frequency: float = 440.0, sample_rate: int = 16000) -> np.ndarray:
        """테스트용 오디오 생성 (주파수별로 다른 신호)"""
        samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, samples)
        
        # 클라이언트별로 다른 주파수 패턴
        audio = (
            0.4 * np.sin(2 * np.pi * frequency * t) +
            0.2 * np.sin(2 * np.pi * (frequency * 1.5) * t) +
            0.15 * np.sin(2 * np.pi * (frequency * 0.75) * t) +
            0.1 * np.random.normal(0, 0.1, samples)
        )
        
        # 16-bit PCM으로 변환
        audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        return audio
    
    async def check_server_health(self) -> Dict[str, Any]:
        """서버 상태 및 리소스 확인"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.server_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        return {"status": "error", "code": response.status}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def single_client_request(self, client_id: int, audio_duration: float, frequency: float) -> ConcurrentTestResult:
        """단일 클라이언트 요청"""
        request_start = time.time()
        
        try:
            # 테스트 오디오 생성 (클라이언트별 다른 주파수)
            audio_data = self.generate_test_audio(audio_duration, frequency)
            audio_bytes = audio_data.tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # API 요청 준비
            request_data = {
                "audio_data": audio_b64,
                "language": "ko",
                "audio_format": "pcm_16khz"
            }
            
            # 처리 시작 시간
            processing_start = time.time()
            wait_time = processing_start - request_start
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.post(f"{self.server_url}/transcribe", 
                                      json=request_data) as response:
                    
                    processing_end = time.time()
                    processing_time = processing_end - processing_start
                    total_time = processing_end - request_start
                    
                    if response.status == 200:
                        data = await response.json()
                        rtf = processing_time / audio_duration
                        
                        result = ConcurrentTestResult(
                            client_id=client_id,
                            audio_duration=audio_duration,
                            processing_time=processing_time,
                            wait_time=wait_time,
                            total_time=total_time,
                            rtf=rtf,
                            text=data.get("text", ""),
                            success=True,
                            timestamp=processing_end - self.start_time
                        )
                        
                        return result
                    else:
                        error_text = await response.text()
                        return ConcurrentTestResult(
                            client_id=client_id,
                            audio_duration=audio_duration,
                            processing_time=processing_time,
                            wait_time=wait_time,
                            total_time=total_time,
                            rtf=float('inf'),
                            text="",
                            success=False,
                            error_message=f"HTTP {response.status}: {error_text}",
                            timestamp=processing_end - self.start_time
                        )
                        
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - processing_start if 'processing_start' in locals() else 0
            total_time = end_time - request_start
            wait_time = 0
            
            return ConcurrentTestResult(
                client_id=client_id,
                audio_duration=audio_duration,
                processing_time=processing_time,
                wait_time=wait_time,
                total_time=total_time,
                rtf=float('inf'),
                text="",
                success=False,
                error_message=str(e),
                timestamp=end_time - self.start_time if self.start_time else 0
            )
    
    async def concurrent_batch_test(self, num_clients: int, audio_duration: float, test_name: str) -> List[ConcurrentTestResult]:
        """동시 배치 테스트"""
        print(f"\n🔄 {test_name}")
        print(f"   클라이언트 수: {num_clients}개, 오디오 길이: {audio_duration:.1f}초")
        print("-" * 60)
        
        # 클라이언트별로 다른 주파수 사용
        frequencies = [440 + (i * 50) for i in range(num_clients)]
        
        # 동시 요청 시작
        batch_start = time.time()
        tasks = []
        
        for i in range(num_clients):
            task = self.single_client_request(i+1, audio_duration, frequencies[i])
            tasks.append(task)
        
        # 모든 요청 대기
        results = await asyncio.gather(*tasks, return_exceptions=True)
        batch_end = time.time()
        batch_duration = batch_end - batch_start
        
        # 결과 처리
        valid_results = []
        for result in results:
            if isinstance(result, ConcurrentTestResult):
                valid_results.append(result)
                with self.results_lock:
                    self.results.append(result)
        
        # 배치 결과 출력
        successful = [r for r in valid_results if r.success]
        failed = [r for r in valid_results if not r.success]
        
        print(f"   배치 완료 시간: {batch_duration:.3f}초")
        print(f"   성공: {len(successful)}개, 실패: {len(failed)}개")
        
        if successful:
            avg_processing = statistics.mean([r.processing_time for r in successful])
            avg_rtf = statistics.mean([r.rtf for r in successful])
            avg_wait = statistics.mean([r.wait_time for r in successful])
            throughput = len(successful) / batch_duration
            
            print(f"   평균 처리시간: {avg_processing:.3f}초")
            print(f"   평균 RTF: {avg_rtf:.4f}")
            print(f"   평균 대기시간: {avg_wait:.3f}초")
            print(f"   처리량: {throughput:.2f} 요청/초")
        
        if failed:
            print(f"   실패 원인: {failed[0].error_message}")
        
        return valid_results
    
    def get_system_resources(self) -> Dict[str, Any]:
        """시스템 리소스 사용량 확인"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU 메모리 사용량 (nvidia-smi 없이 대략적 추정)
            gpu_info = {}
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        used, total = map(int, lines[0].split(', '))
                        gpu_info = {
                            "gpu_memory_used_mb": used,
                            "gpu_memory_total_mb": total,
                            "gpu_memory_used_percent": (used / total) * 100
                        }
            except:
                gpu_info = {"gpu_memory_info": "unavailable"}
            
            return {
                "cpu_percent": cpu_percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "memory_percent": memory.percent,
                **gpu_info
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def run_concurrent_tests(self):
        """포괄적인 동시성 테스트 실행"""
        print("🔄 다중 클라이언트 동시성 성능 테스트")
        print("=" * 70)
        print(f"서버: {self.server_url}")
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 서버 상태 확인
        health = await self.check_server_health()
        if health.get("status") != "healthy":
            print(f"❌ 서버가 준비되지 않았습니다: {health}")
            return
        
        print(f"✅ 서버 상태: {health.get('status')}")
        if health.get('gpu_available'):
            print(f"   GPU: {health.get('gpu_name', 'Unknown')}")
        
        # 시스템 리소스 확인
        resources = self.get_system_resources()
        print(f"💻 시스템 리소스:")
        print(f"   CPU: {resources.get('cpu_percent', 'N/A'):.1f}%")
        print(f"   RAM: {resources.get('memory_used_gb', 0):.1f}GB / {resources.get('memory_total_gb', 0):.1f}GB ({resources.get('memory_percent', 0):.1f}%)")
        if 'gpu_memory_used_mb' in resources:
            print(f"   GPU RAM: {resources['gpu_memory_used_mb']:.0f}MB / {resources['gpu_memory_total_mb']:.0f}MB ({resources['gpu_memory_used_percent']:.1f}%)")
        
        self.start_time = time.time()
        
        # 테스트 시나리오들
        test_scenarios = [
            (2, 5.0, "2개 클라이언트 동시 요청 (5초 오디오)"),
            (4, 5.0, "4개 클라이언트 동시 요청 (5초 오디오)"),
            (8, 3.0, "8개 클라이언트 동시 요청 (3초 오디오)"),
            (5, 10.0, "5개 클라이언트 동시 요청 (10초 오디오)"),
            (10, 2.0, "10개 클라이언트 동시 요청 (2초 오디오)"),
            (3, 15.0, "3개 클라이언트 동시 요청 (15초 오디오)"),
        ]
        
        for num_clients, audio_duration, test_name in test_scenarios:
            try:
                await self.concurrent_batch_test(num_clients, audio_duration, test_name)
                
                # 테스트 간 잠시 대기 (서버 부하 감소)
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"❌ {test_name} 실패: {e}")
        
        print("\n📊 전체 동시성 성능 분석")
        await self.analyze_concurrent_results()
    
    async def analyze_concurrent_results(self):
        """동시성 결과 분석"""
        if not self.results:
            print("❌ 분석할 결과가 없습니다.")
            return
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        print("=" * 70)
        print("📈 동시성 성능 분석 결과")
        print("=" * 70)
        
        print(f"✅ 총 성공 요청: {len(successful_results)}개")
        print(f"❌ 총 실패 요청: {len(failed_results)}개")
        print(f"🎯 성공률: {len(successful_results) / len(self.results) * 100:.1f}%")
        
        if not successful_results:
            print("❌ 성공한 요청이 없습니다.")
            return
        
        # RTF 분석
        rtfs = [r.rtf for r in successful_results]
        processing_times = [r.processing_time for r in successful_results]
        wait_times = [r.wait_time for r in successful_results]
        total_times = [r.total_time for r in successful_results]
        
        print(f"\n🎯 RTF 성능:")
        print(f"   평균 RTF: {statistics.mean(rtfs):.4f}")
        print(f"   중간값 RTF: {statistics.median(rtfs):.4f}")
        print(f"   최소 RTF: {min(rtfs):.4f}")
        print(f"   최대 RTF: {max(rtfs):.4f}")
        if len(rtfs) > 1:
            print(f"   표준편차: {statistics.stdev(rtfs):.4f}")
        
        print(f"\n⏱️ 시간 분석:")
        print(f"   평균 처리시간: {statistics.mean(processing_times):.3f}초")
        print(f"   평균 대기시간: {statistics.mean(wait_times):.3f}초")
        print(f"   평균 전체시간: {statistics.mean(total_times):.3f}초")
        print(f"   최대 대기시간: {max(wait_times):.3f}초")
        
        # 처리량 분석
        test_duration = max([r.timestamp for r in successful_results])
        throughput = len(successful_results) / test_duration
        print(f"\n🚀 처리량:")
        print(f"   전체 테스트 시간: {test_duration:.3f}초")
        print(f"   평균 처리량: {throughput:.2f} 요청/초")
        
        # 동시성 품질 분석
        avg_rtf = statistics.mean(rtfs)
        max_wait = max(wait_times)
        
        print(f"\n🏆 동시성 품질 평가:")
        
        # RTF 성능 유지도
        if avg_rtf < 0.1:
            rtf_grade = "S+ (초고속 유지)"
        elif avg_rtf < 0.3:
            rtf_grade = "S (고속 유지)"
        elif avg_rtf < 0.5:
            rtf_grade = "A (양호)"
        elif avg_rtf < 1.0:
            rtf_grade = "B (실시간 유지)"
        else:
            rtf_grade = "C (성능 저하)"
        
        print(f"   RTF 성능: {rtf_grade}")
        
        # 대기시간 품질
        if max_wait < 0.1:
            wait_grade = "S+ (즉시 처리)"
        elif max_wait < 0.5:
            wait_grade = "S (빠른 응답)"
        elif max_wait < 1.0:
            wait_grade = "A (양호한 응답)"
        elif max_wait < 2.0:
            wait_grade = "B (보통 응답)"
        else:
            wait_grade = "C (느린 응답)"
        
        print(f"   응답속도: {wait_grade}")
        
        # 처리량 품질
        if throughput > 10:
            throughput_grade = "S+ (매우 높음)"
        elif throughput > 5:
            throughput_grade = "S (높음)"
        elif throughput > 2:
            throughput_grade = "A (양호)"
        elif throughput > 1:
            throughput_grade = "B (보통)"
        else:
            throughput_grade = "C (낮음)"
        
        print(f"   처리량: {throughput_grade}")
        
        # 클라이언트별 분석
        client_stats = {}
        for result in successful_results:
            if result.client_id not in client_stats:
                client_stats[result.client_id] = []
            client_stats[result.client_id].append(result)
        
        print(f"\n👥 클라이언트별 성능:")
        print("┌──────────┬──────────┬──────────┬──────────┬──────────┐")
        print("│ 클라이언트│ 요청수   │ 평균RTF  │ 평균대기 │ 성공률   │")
        print("├──────────┼──────────┼──────────┼──────────┼──────────┤")
        
        for client_id in sorted(client_stats.keys()):
            client_results = client_stats[client_id]
            client_rtfs = [r.rtf for r in client_results if r.success]
            client_waits = [r.wait_time for r in client_results if r.success]
            
            if client_rtfs:
                avg_rtf = statistics.mean(client_rtfs)
                avg_wait = statistics.mean(client_waits)
                success_rate = len(client_rtfs) / len(client_results) * 100
                
                print(f"│ 클라{client_id:3d}      │ {len(client_results):6d}개  │ {avg_rtf:8.4f} │ {avg_wait:8.3f}초│ {success_rate:7.1f}% │")
        
        print("└──────────┴──────────┴──────────┴──────────┴──────────┘")
        
        # 실패 분석
        if failed_results:
            print(f"\n❌ 실패 분석:")
            failure_reasons = {}
            for result in failed_results:
                reason = result.error_message.split(':')[0] if ':' in result.error_message else result.error_message
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            for reason, count in failure_reasons.items():
                print(f"   • {reason}: {count}건")
        
        # 권장사항
        print(f"\n💡 동시성 처리 권장사항:")
        if avg_rtf < 0.3 and max_wait < 1.0:
            print("   • 🎉 우수한 동시성 성능! 실시간 다중 클라이언트 처리에 적합합니다.")
            print("   • 현재 설정으로 프로덕션 환경에서 사용 가능합니다.")
        elif avg_rtf < 1.0 and max_wait < 2.0:
            print("   • ✅ 양호한 동시성 성능입니다.")
            print("   • 클라이언트 수 제한 또는 큐잉 시스템 고려하세요.")
        else:
            print("   • ⚠️ 동시성 처리 개선이 필요합니다.")
            print("   • 서버 스케일링 또는 로드 밸런싱을 고려하세요.")
        
        # 최종 시스템 리소스 확인
        final_resources = self.get_system_resources()
        print(f"\n💻 테스트 후 시스템 리소스:")
        print(f"   CPU: {final_resources.get('cpu_percent', 'N/A'):.1f}%")
        print(f"   RAM: {final_resources.get('memory_used_gb', 0):.1f}GB ({final_resources.get('memory_percent', 0):.1f}%)")
        if 'gpu_memory_used_mb' in final_resources:
            print(f"   GPU RAM: {final_resources['gpu_memory_used_mb']:.0f}MB ({final_resources['gpu_memory_used_percent']:.1f}%)")
    
    def save_results(self):
        """결과를 JSON 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"concurrent_stt_performance_{timestamp}.json"
        
        data = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "server_url": self.server_url,
                "total_requests": len(self.results),
                "successful_requests": len([r for r in self.results if r.success])
            },
            "results": [
                {
                    "client_id": r.client_id,
                    "audio_duration": r.audio_duration,
                    "processing_time": r.processing_time,
                    "wait_time": r.wait_time,
                    "total_time": r.total_time,
                    "rtf": r.rtf if r.rtf != float('inf') else None,
                    "text": r.text,
                    "success": r.success,
                    "error_message": r.error_message,
                    "timestamp": r.timestamp
                }
                for r in self.results
            ]
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\n💾 결과 저장: {filename}")
        except Exception as e:
            print(f"\n❌ 결과 저장 실패: {e}")


async def main():
    """메인 함수"""
    tester = ConcurrentSTTTester()
    
    try:
        await tester.run_concurrent_tests()
        tester.save_results()
        
        # 최종 요약
        successful = [r for r in tester.results if r.success]
        if successful:
            avg_rtf = statistics.mean([r.rtf for r in successful])
            avg_throughput = len(successful) / max([r.timestamp for r in successful])
            
            print(f"\n🎯 최종 동시성 성능 요약:")
            print(f"   성공률: {len(successful) / len(tester.results) * 100:.1f}%")
            print(f"   평균 RTF: {avg_rtf:.4f}")
            print(f"   평균 처리량: {avg_throughput:.2f} 요청/초")
            
    except KeyboardInterrupt:
        print("\n⚠️ 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 