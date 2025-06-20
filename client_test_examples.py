#!/usr/bin/env python3
"""
한국어 STT 디코더 v2 클라이언트 테스트 예제
외부 클라이언트들이 STT 서버 기능을 테스트할 수 있는 종합 스크립트
"""

import requests
import base64
import time
import json
import os
import asyncio
import aiohttp
from typing import Dict, Any, List

# 서버 설정
STT_SERVER_URL = "http://localhost:8005"  # 외부 접속시 실제 IP로 변경

class STTClient:
    """STT 서버 클라이언트"""
    
    def __init__(self, server_url: str = STT_SERVER_URL):
        self.server_url = server_url
    
    def health_check(self) -> Dict[str, Any]:
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unreachable"}
    
    def transcribe_direct(self, audio_file_path: str, language: str = "ko") -> Dict[str, Any]:
        """직접 음성 인식"""
        try:
            # 오디오 파일을 Base64로 인코딩
            with open(audio_file_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
            
            payload = {
                "audio_data": audio_data,
                "language": language,
                "audio_format": "wav"
            }
            
            start_time = time.time()
            response = requests.post(f"{self.server_url}/transcribe", json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            result["client_total_time"] = time.time() - start_time
            return result
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def transcribe_queue(self, audio_file_path: str, priority: str = "medium", language: str = "ko") -> Dict[str, Any]:
        """큐를 통한 음성 인식"""
        try:
            # 오디오 파일을 Base64로 인코딩
            with open(audio_file_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
            
            payload = {
                "audio_data": audio_data,
                "language": language,
                "audio_format": "wav",
                "priority": priority
            }
            
            # 큐에 요청 제출
            start_time = time.time()
            response = requests.post(f"{self.server_url}/queue/transcribe", json=payload, timeout=30)
            response.raise_for_status()
            
            request_data = response.json()
            request_id = request_data["request_id"]
            
            print(f"⏳ 큐에 추가됨 - ID: {request_id}, 위치: {request_data.get('position', '?')}")
            
            # 결과 대기
            while True:
                result_response = requests.get(f"{self.server_url}/queue/result/{request_id}", timeout=10)
                result_response.raise_for_status()
                result = result_response.json()
                
                if result["status"] == "completed":
                    total_time = time.time() - start_time
                    result["result"]["client_total_time"] = total_time
                    return result["result"]
                elif result["status"] == "failed":
                    return {"error": result.get("error", "Unknown error"), "success": False}
                
                time.sleep(0.5)  # 0.5초 대기
                
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def get_queue_status(self) -> Dict[str, Any]:
        """큐 상태 조회"""
        try:
            response = requests.get(f"{self.server_url}/queue/status", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_server_stats(self) -> Dict[str, Any]:
        """서버 통계 조회"""
        try:
            response = requests.get(f"{self.server_url}/stats", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

class STTTester:
    """STT 서버 종합 테스트"""
    
    def __init__(self):
        self.client = STTClient()
        self.test_results = []
    
    def run_basic_tests(self):
        """기본 기능 테스트"""
        print("=" * 60)
        print("🧪 한국어 STT 디코더 v2 기본 기능 테스트")
        print("=" * 60)
        
        # 1. 서버 상태 확인
        print("\n1️⃣ 서버 상태 확인")
        health = self.client.health_check()
        if "error" in health:
            print(f"❌ 서버 연결 실패: {health['error']}")
            return False
        else:
            print(f"✅ 서버 정상 - 모델: {health.get('model', 'unknown')}")
        
        # 2. 큐 상태 확인
        print("\n2️⃣ 큐 상태 확인")
        queue_status = self.client.get_queue_status()
        if "error" in queue_status:
            print(f"❌ 큐 상태 조회 실패: {queue_status['error']}")
        else:
            print(f"📊 큐 길이: {queue_status.get('queue_length', 0)}")
            print(f"📈 처리량: {queue_status.get('throughput', 'N/A')}")
            print(f"⚡ 평균 RTF: {queue_status.get('average_rtf', 'N/A')}")
        
        # 3. 서버 통계 확인
        print("\n3️⃣ 서버 통계 확인")
        stats = self.client.get_server_stats()
        if "error" not in stats and "server_info" in stats:
            print(f"🖥️ GPU: {stats['server_info'].get('gpu', 'unknown')}")
            print(f"💾 GPU 메모리: {stats['server_info'].get('gpu_memory', 'unknown')}")
            print(f"📊 총 요청 수: {stats['performance'].get('total_requests', 0)}")
            print(f"✅ 성공률: {stats['performance'].get('success_rate', 'N/A')}")
        
        return True
    
    def test_audio_files(self):
        """오디오 파일 테스트"""
        print("\n=" * 60)
        print("🎵 오디오 파일 음성 인식 테스트")
        print("=" * 60)
        
        # 테스트할 오디오 파일들
        test_files = [
            {
                "file": "test_korean_sample1.wav",
                "expected": "김화영이 번역하고 책세상에서 출간된 카뮈의 전집"
            },
            {
                "file": "test_korean_sample2.wav", 
                "expected": "그친구 이름이 되게 흔했는데"
            }
        ]
        
        for i, test_case in enumerate(test_files, 1):
            if not os.path.exists(test_case["file"]):
                print(f"⚠️ {i}. {test_case['file']} 파일이 없습니다.")
                continue
            
            print(f"\n{i}. 🎵 {test_case['file']} 테스트")
            print(f"   📝 예상 텍스트: \"{test_case['expected']}\"")
            
            # 직접 API 테스트
            print("   📡 직접 API 테스트...")
            direct_result = self.client.transcribe_direct(test_case["file"])
            
            if "error" in direct_result:
                print(f"   ❌ 직접 API 실패: {direct_result['error']}")
            else:
                print(f"   ✅ 인식 결과: \"{direct_result.get('text', '')}\"")
                print(f"   ⚡ RTF: {direct_result.get('rtf', 'N/A')}")
                print(f"   ⏱️ 처리 시간: {direct_result.get('processing_time', 'N/A')}초")
                print(f"   🕐 클라이언트 총 시간: {direct_result.get('client_total_time', 'N/A'):.3f}초")
                
                self.test_results.append({
                    "file": test_case["file"],
                    "method": "direct",
                    "success": True,
                    "rtf": direct_result.get('rtf'),
                    "processing_time": direct_result.get('processing_time'),
                    "text": direct_result.get('text', '')
                })
            
            # 큐 API 테스트
            print("   📬 큐 API 테스트...")
            queue_result = self.client.transcribe_queue(test_case["file"], priority="high")
            
            if "error" in queue_result:
                print(f"   ❌ 큐 API 실패: {queue_result['error']}")
            else:
                print(f"   ✅ 인식 결과: \"{queue_result.get('text', '')}\"")
                print(f"   ⚡ RTF: {queue_result.get('rtf', 'N/A')}")
                print(f"   ⏱️ 처리 시간: {queue_result.get('processing_time', 'N/A')}초")
                print(f"   🕐 클라이언트 총 시간: {queue_result.get('client_total_time', 'N/A'):.3f}초")
                
                self.test_results.append({
                    "file": test_case["file"],
                    "method": "queue",
                    "success": True,
                    "rtf": queue_result.get('rtf'),
                    "processing_time": queue_result.get('processing_time'),
                    "text": queue_result.get('text', '')
                })
    
    def test_concurrent_requests(self, num_clients: int = 5):
        """동시 요청 테스트"""
        print(f"\n=" * 60)
        print(f"🚀 동시 {num_clients}개 클라이언트 테스트")
        print("=" * 60)
        
        if not os.path.exists("test_korean_sample1.wav"):
            print("⚠️ 테스트 오디오 파일이 없어 동시 요청 테스트를 건너뜁니다.")
            return
        
        async def single_request(session, client_id):
            """단일 비동기 요청"""
            try:
                with open("test_korean_sample1.wav", "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode('utf-8')
                
                payload = {
                    "audio_data": audio_data,
                    "language": "ko",
                    "audio_format": "wav"
                }
                
                start_time = time.time()
                async with session.post(f"{STT_SERVER_URL}/transcribe", json=payload) as response:
                    result = await response.json()
                    total_time = time.time() - start_time
                    
                    return {
                        "client_id": client_id,
                        "success": response.status == 200,
                        "rtf": result.get('rtf'),
                        "processing_time": result.get('processing_time'),
                        "total_time": total_time,
                        "text": result.get('text', '')[:50] + "..." if len(result.get('text', '')) > 50 else result.get('text', '')
                    }
            except Exception as e:
                return {
                    "client_id": client_id,
                    "success": False,
                    "error": str(e)
                }
        
        async def run_concurrent_test():
            """동시 요청 실행"""
            connector = aiohttp.TCPConnector(limit=100)
            timeout = aiohttp.ClientTimeout(total=60)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                tasks = [single_request(session, i) for i in range(num_clients)]
                results = await asyncio.gather(*tasks)
                return results
        
        print(f"📡 {num_clients}개 클라이언트로 동시 요청 전송 중...")
        start_time = time.time()
        
        try:
            results = asyncio.run(run_concurrent_test())
            total_time = time.time() - start_time
            
            # 결과 분석
            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]
            
            print(f"\n📊 동시 요청 테스트 결과:")
            print(f"   ✅ 성공: {len(successful)}/{num_clients} ({len(successful)/num_clients*100:.1f}%)")
            print(f"   ❌ 실패: {len(failed)}")
            print(f"   🕐 전체 소요 시간: {total_time:.3f}초")
            print(f"   🚀 처리량: {len(successful)/total_time:.2f} 요청/초")
            
            if successful:
                rtfs = [r["rtf"] for r in successful if r["rtf"] is not None]
                proc_times = [r["processing_time"] for r in successful if r["processing_time"] is not None]
                
                if rtfs:
                    print(f"   ⚡ 평균 RTF: {sum(rtfs)/len(rtfs):.4f}")
                    print(f"   ⚡ 최고 RTF: {min(rtfs):.4f}")
                
                if proc_times:
                    print(f"   ⏱️ 평균 처리 시간: {sum(proc_times)/len(proc_times):.3f}초")
            
            # 샘플 결과 출력
            print(f"\n📋 샘플 결과 (처음 3개):")
            for result in results[:3]:
                if result["success"]:
                    print(f"   클라이언트 {result['client_id']}: RTF {result.get('rtf', 'N/A')}, \"{result.get('text', '')}\"")
                else:
                    print(f"   클라이언트 {result['client_id']}: 실패 - {result.get('error', 'Unknown')}")
                    
        except Exception as e:
            print(f"❌ 동시 요청 테스트 실패: {e}")
    
    def print_summary(self):
        """테스트 결과 요약"""
        print("\n" + "=" * 60)
        print("📋 테스트 결과 요약")
        print("=" * 60)
        
        if not self.test_results:
            print("⚠️ 수집된 테스트 결과가 없습니다.")
            return
        
        # RTF 통계
        rtfs = [r["rtf"] for r in self.test_results if r["rtf"] is not None]
        if rtfs:
            print(f"⚡ RTF 성능:")
            print(f"   평균: {sum(rtfs)/len(rtfs):.4f}")
            print(f"   최고: {min(rtfs):.4f}")
            print(f"   최저: {max(rtfs):.4f}")
        
        # 처리 시간 통계
        proc_times = [r["processing_time"] for r in self.test_results if r["processing_time"] is not None]
        if proc_times:
            print(f"⏱️ 처리 시간:")
            print(f"   평균: {sum(proc_times)/len(proc_times):.3f}초")
            print(f"   최단: {min(proc_times):.3f}초")
            print(f"   최장: {max(proc_times):.3f}초")
        
        # 성공률
        successful = len([r for r in self.test_results if r["success"]])
        print(f"✅ 성공률: {successful}/{len(self.test_results)} ({successful/len(self.test_results)*100:.1f}%)")
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🎯 한국어 STT 디코더 v2 종합 테스트 시작")
        print(f"📡 서버 주소: {STT_SERVER_URL}")
        
        # 기본 기능 테스트
        if not self.run_basic_tests():
            print("❌ 서버 연결 실패로 테스트를 중단합니다.")
            return
        
        # 오디오 파일 테스트
        self.test_audio_files()
        
        # 동시 요청 테스트
        self.test_concurrent_requests(5)
        
        # 결과 요약
        self.print_summary()
        
        print("\n🎉 모든 테스트 완료!")

def create_sample_audio():
    """테스트용 샘플 오디오 파일 생성 안내"""
    print("=" * 60)
    print("🎵 테스트용 오디오 파일 안내")
    print("=" * 60)
    print("다음 파일들이 있으면 더 정확한 테스트가 가능합니다:")
    print("- test_korean_sample1.wav: \"김화영이 번역하고 책세상에서 출간된 카뮈의 전집\"")
    print("- test_korean_sample2.wav: \"그친구 이름이 되게 흔했는데\"")
    print()
    print("테스트용 오디오가 없다면 직접 녹음하거나 TTS로 생성하여 사용하세요.")

if __name__ == "__main__":
    print("🚀 한국어 STT 디코더 v2 클라이언트 테스트")
    print("이 스크립트는 STT 서버의 모든 기능을 테스트합니다.")
    print()
    
    # 기본 테스트 실행
    tester = STTTester()
    tester.run_all_tests()
    
    # 추가 정보
    print("\n" + "=" * 60)
    print("🔧 추가 테스트 옵션")
    print("=" * 60)
    print("1. 더 많은 동시 클라이언트 테스트:")
    print("   tester.test_concurrent_requests(20)")
    print()
    print("2. 개별 API 테스트:")
    print("   client = STTClient()")
    print("   result = client.transcribe_direct('your_audio.wav')")
    print()
    print("3. 서버 주소 변경:")
    print("   STT_SERVER_URL = 'http://YOUR_IP:8005'")
    
    create_sample_audio() 