#!/usr/bin/env python3
"""
Large-v3 모델 전용 극한 RTF 최적화 테스트
인식률 최우선 - RTF < 0.05x 목표 검증
"""

import asyncio
import aiohttp
import base64
import time
import os
import statistics
from typing import List, Tuple, Dict, Any

# 서버 설정
LARGE_ONLY_SERVER_URL = "http://localhost:8005"

class LargeOnlyRTFTester:
    """Large 모델 전용 RTF 테스트 클래스"""
    
    def load_korean_audio_files(self) -> List[Tuple[str, str, str]]:
        """한국어 오디오 파일 로드"""
        audio_files = []
        
        # 샘플 1
        if os.path.exists("test_korean_sample1.wav"):
            with open("test_korean_sample1.wav", "rb") as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
            expected_text = "김화영이 번역하고 책세상에서 출간된 카뮈의 전집"
            audio_files.append(("sample1", audio_data, expected_text))
        
        # 샘플 2  
        if os.path.exists("test_korean_sample2.wav"):
            with open("test_korean_sample2.wav", "rb") as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
            expected_text = "그친구 이름이 되게 흔했는데"
            audio_files.append(("sample2", audio_data, expected_text))
        
        return audio_files
    
    async def test_health_check(self):
        """헬스 체크 테스트"""
        print("🔍 Large 모델 전용 서버 헬스 체크...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{LARGE_ONLY_SERVER_URL}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print("✅ 서버 상태: 정상")
                        print(f"   모델: {data['model_info']['model_name']}")
                        print(f"   GPU: {data['gpu_info']['name']}")
                        print(f"   메모리: {data['gpu_info']['memory_allocated_gb']}GB")
                        print(f"   최적화: cuDNN={data['optimization_status']['cudnn_enabled']}, "
                              f"TF32={data['optimization_status']['tf32_enabled']}")
                        return True
                    else:
                        print(f"❌ 헬스 체크 실패: {response.status}")
                        return False
            except Exception as e:
                print(f"❌ 연결 실패: {e}")
                return False
    
    async def test_single_request(self, session, audio_data: str, name: str, expected: str) -> Dict[str, Any]:
        """단일 요청 테스트"""
        payload = {
            "audio_data": audio_data,
            "language": "ko",
            "audio_format": "wav"
        }
        
        start_time = time.time()
        
        try:
            async with session.post(f"{LARGE_ONLY_SERVER_URL}/transcribe", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    total_time = time.time() - start_time
                    
                    return {
                        'success': True,
                        'name': name,
                        'text': result['text'],
                        'expected': expected,
                        'rtf': result['rtf'],
                        'processing_time': result['processing_time'],
                        'total_time': total_time,
                        'confidence': result.get('confidence', 0.0)
                    }
                else:
                    return {
                        'success': False,
                        'name': name,
                        'error': f"HTTP {response.status}",
                        'total_time': time.time() - start_time
                    }
        except Exception as e:
            return {
                'success': False,
                'name': name,
                'error': str(e),
                'total_time': time.time() - start_time
            }
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """문자열 유사도 계산 (간단한 버전)"""
        words1 = set(text1.replace(" ", ""))
        words2 = set(text2.replace(" ", ""))
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def test_accuracy_and_performance(self):
        """인식률과 RTF 성능 종합 테스트"""
        print("\n🎯 Large-v3 모델 인식률 & RTF 성능 테스트")
        print("=" * 60)
        
        audio_files = self.load_korean_audio_files()
        if not audio_files:
            print("❌ 테스트 오디오 파일을 찾을 수 없습니다")
            return
        
        async with aiohttp.ClientSession() as session:
            all_results = []
            
            # 각 오디오 파일을 여러 번 테스트
            for sample_name, audio_data, expected_text in audio_files:
                print(f"\n📄 {sample_name} 테스트 (5회 반복)")
                print(f"   예상 텍스트: {expected_text}")
                
                sample_results = []
                
                for i in range(5):
                    result = await self.test_single_request(
                        session, audio_data, f"{sample_name}_{i+1}", expected_text
                    )
                    sample_results.append(result)
                    
                    if result['success']:
                        similarity = self.calculate_similarity(result['text'], expected_text)
                        print(f"   회차 {i+1}: RTF={result['rtf']:.4f}, "
                              f"시간={result['processing_time']:.3f}s, "
                              f"유사도={similarity:.2%}")
                        print(f"        인식: {result['text']}")
                    else:
                        print(f"   회차 {i+1}: ❌ 실패 - {result['error']}")
                
                all_results.extend(sample_results)
            
            # 결과 분석
            self.analyze_results(all_results)
    
    async def test_concurrent_performance(self):
        """동시 요청 성능 테스트"""
        print("\n🚀 Large-v3 모델 동시 처리 성능 테스트")
        print("=" * 60)
        
        audio_files = self.load_korean_audio_files()
        if not audio_files:
            print("❌ 테스트 오디오 파일을 찾을 수 없습니다")
            return
        
        # 10개 동시 요청
        concurrent_requests = 10
        print(f"📊 {concurrent_requests}개 동시 요청 테스트")
        
        async with aiohttp.ClientSession() as session:
            # 요청 생성
            tasks = []
            sample_name, audio_data, expected_text = audio_files[0]  # 첫 번째 샘플 사용
            
            start_time = time.time()
            
            for i in range(concurrent_requests):
                task = self.test_single_request(
                    session, audio_data, f"concurrent_{i+1}", expected_text
                )
                tasks.append(task)
            
            # 모든 요청 실행
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # 결과 분석
            successful = [r for r in results if r['success']]
            failed = [r for r in results if not r['success']]
            
            print(f"\n📈 동시 처리 결과:")
            print(f"   성공: {len(successful)}/{concurrent_requests}")
            print(f"   실패: {len(failed)}")
            print(f"   전체 소요시간: {total_time:.3f}초")
            
            if successful:
                rtfs = [r['rtf'] for r in successful]
                processing_times = [r['processing_time'] for r in successful]
                
                print(f"   평균 RTF: {statistics.mean(rtfs):.4f}")
                print(f"   최저 RTF: {min(rtfs):.4f}")
                print(f"   최고 RTF: {max(rtfs):.4f}")
                print(f"   평균 처리시간: {statistics.mean(processing_times):.3f}초")
                print(f"   처리량: {len(successful)/total_time:.2f} 요청/초")
    
    def analyze_results(self, results: List[Dict[str, Any]]):
        """결과 분석 및 출력"""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\n📊 종합 결과 분석")
        print("=" * 60)
        print(f"✅ 성공: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"❌ 실패: {len(failed)}")
        
        if successful:
            rtfs = [r['rtf'] for r in successful]
            processing_times = [r['processing_time'] for r in successful]
            similarities = []
            
            for r in successful:
                similarity = self.calculate_similarity(r['text'], r['expected'])
                similarities.append(similarity)
            
            print(f"\n🏆 RTF 성능:")
            print(f"   평균 RTF: {statistics.mean(rtfs):.4f}")
            print(f"   최저 RTF: {min(rtfs):.4f} (최고 성능)")
            print(f"   최고 RTF: {max(rtfs):.4f}")
            print(f"   표준편차: {statistics.stdev(rtfs) if len(rtfs) > 1 else 0:.4f}")
            
            # RTF 등급 분류
            excellent = sum(1 for rtf in rtfs if rtf < 0.05)
            great = sum(1 for rtf in rtfs if 0.05 <= rtf < 0.10)
            good = sum(1 for rtf in rtfs if 0.10 <= rtf < 0.15)
            fair = sum(1 for rtf in rtfs if rtf >= 0.15)
            
            print(f"\n📈 RTF 등급 분포:")
            print(f"   🏆 EXCELLENT (<0.05): {excellent}/{len(rtfs)} ({excellent/len(rtfs)*100:.1f}%)")
            print(f"   ✨ GREAT (0.05-0.10): {great}/{len(rtfs)} ({great/len(rtfs)*100:.1f}%)")
            print(f"   ✅ GOOD (0.10-0.15): {good}/{len(rtfs)} ({good/len(rtfs)*100:.1f}%)")
            print(f"   ⚠️ FAIR (≥0.15): {fair}/{len(rtfs)} ({fair/len(rtfs)*100:.1f}%)")
            
            print(f"\n🎯 인식률 성능:")
            print(f"   평균 유사도: {statistics.mean(similarities):.2%}")
            print(f"   최고 유사도: {max(similarities):.2%}")
            print(f"   최저 유사도: {min(similarities):.2%}")
            
            print(f"\n⏱️ 처리 시간:")
            print(f"   평균: {statistics.mean(processing_times):.3f}초")
            print(f"   최단: {min(processing_times):.3f}초")
            print(f"   최장: {max(processing_times):.3f}초")
            
            # 목표 달성 여부
            target_achieved = excellent > 0
            high_accuracy = statistics.mean(similarities) > 0.85
            
            print(f"\n🎯 목표 달성 현황:")
            print(f"   RTF < 0.05x 달성: {'✅ 달성' if target_achieved else '❌ 미달성'}")
            print(f"   높은 인식률 (>85%): {'✅ 달성' if high_accuracy else '❌ 미달성'}")
            
            if target_achieved and high_accuracy:
                print(f"\n🏆 축하합니다! Large-v3 모델로 RTF < 0.05x와 높은 인식률을 모두 달성했습니다!")
            elif target_achieved:
                print(f"\n✨ RTF 목표는 달성했지만 인식률 개선이 필요합니다.")
            elif high_accuracy:
                print(f"\n✅ 높은 인식률은 달성했지만 RTF 최적화가 더 필요합니다.")
            else:
                print(f"\n⚠️ RTF와 인식률 모두 추가 최적화가 필요합니다.")

async def main():
    """메인 테스트 실행"""
    tester = LargeOnlyRTFTester()
    
    # 1. 헬스 체크
    if not await tester.test_health_check():
        print("❌ 서버가 응답하지 않습니다. 서버를 먼저 시작해주세요.")
        return
    
    # 2. 인식률 & RTF 테스트
    await tester.test_accuracy_and_performance()
    
    # 3. 동시 처리 성능 테스트
    await tester.test_concurrent_performance()
    
    print(f"\n🎉 Large-v3 전용 극한 최적화 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main()) 