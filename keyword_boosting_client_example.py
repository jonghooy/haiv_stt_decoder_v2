#!/usr/bin/env python3
"""
Keyword Boosting Client Example
키워드 부스팅 시스템 전용 클라이언트 예제
"""

import asyncio
import aiohttp
import base64
import json
import time
import numpy as np
import logging
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeywordBoostingClient:
    def __init__(self, server_url: str = "http://localhost:8004"):
        self.server_url = server_url
        self.call_id = "comprehensive_test"  # 기본 call_id
        
    async def test_server_health(self):
        """서버 상태 확인"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/health") as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ 서버 상태: 정상")
                        logger.info(f"   GPU: {result.get('gpu_available', 'N/A')}")
                        logger.info(f"   모델: {result.get('model_loaded', 'N/A')}")
                        return True
                    else:
                        logger.error(f"❌ 서버 오류: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ 서버 연결 실패: {e}")
            return False

    async def setup_comprehensive_keywords(self):
        """종합적인 키워드 데이터베이스 설정"""
        # 통합 키워드 리스트 (올바른 API 스펙에 맞춤)
        all_keywords = [
            {"keyword": "카뮈", "aliases": ["카뮤", "까뮤", "알베르 카뮤"], "category": "authors", "confidence_threshold": 0.8},
            {"keyword": "도스토예프스키", "aliases": ["도스또예프스키", "도스토예프스끼"], "category": "authors", "confidence_threshold": 0.8},
            {"keyword": "톨스토이", "aliases": ["똘스또이", "톨스또이"], "category": "authors", "confidence_threshold": 0.8},
            {"keyword": "헤밍웨이", "aliases": ["헤밍웨이", "어니스트 헤밍웨이"], "category": "authors", "confidence_threshold": 0.8},
            {"keyword": "서울대학교", "aliases": ["서울대", "에스엔유", "SNU"], "category": "universities", "confidence_threshold": 0.8},
            {"keyword": "연세대학교", "aliases": ["연세대", "연대"], "category": "universities", "confidence_threshold": 0.8},
            {"keyword": "고려대학교", "aliases": ["고려대", "고대"], "category": "universities", "confidence_threshold": 0.8},
            {"keyword": "KAIST", "aliases": ["카이스트", "한국과학기술원"], "category": "universities", "confidence_threshold": 0.8},
            {"keyword": "딥러닝", "aliases": ["딥 러닝", "Deep Learning"], "category": "technology", "confidence_threshold": 0.8},
            {"keyword": "머신러닝", "aliases": ["머신 러닝", "Machine Learning"], "category": "technology", "confidence_threshold": 0.8},
            {"keyword": "인공지능", "aliases": ["AI", "에이아이"], "category": "technology", "confidence_threshold": 0.8},
            {"keyword": "블록체인", "aliases": ["블록 체인", "Blockchain"], "category": "technology", "confidence_threshold": 0.8},
            {"keyword": "네이버", "aliases": ["NAVER"], "category": "companies", "confidence_threshold": 0.8},
            {"keyword": "카카오", "aliases": ["Kakao"], "category": "companies", "confidence_threshold": 0.8},
            {"keyword": "삼성전자", "aliases": ["삼성", "Samsung"], "category": "companies", "confidence_threshold": 0.8},
            {"keyword": "LG전자", "aliases": ["엘지전자", "LG"], "category": "companies", "confidence_threshold": 0.8}
        ]
        
        # Call ID 사용 (후처리 시스템은 call_id 기반)
        call_id = "comprehensive_test"
        
        payload = {
            "call_id": call_id,
            "keywords": all_keywords
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.server_url}/keywords/register",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ 키워드 등록 성공 (Call ID: {call_id})")
                        logger.info(f"🎯 총 {len(all_keywords)}개 키워드 등록 완료")
                        
                        # 클래스 인스턴스 변수에 call_id 저장
                        self.call_id = call_id
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ 키워드 등록 실패: {response.status} - {error_text}")
                        return False
            except Exception as e:
                logger.error(f"❌ 키워드 등록 오류: {e}")
                return False

    async def get_all_keywords(self):
        """전체 키워드 조회"""
        try:
            call_id = getattr(self, 'call_id', 'comprehensive_test')
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/keywords/{call_id}") as response:
                    if response.status == 200:
                        result = await response.json()
                        keywords_dict = result.get('keywords', {})
                        
                        # 카테고리별 분류
                        categories = {}
                        for keyword, details in keywords_dict.items():
                            category = details.get('category', 'unknown')
                            if category not in categories:
                                categories[category] = []
                            categories[category].append(details)
                        
                        logger.info(f"📋 등록된 키워드 현황 (총 {len(keywords_dict)}개):")
                        for category, cat_keywords in categories.items():
                            logger.info(f"\n  📂 {category} ({len(cat_keywords)}개):")
                            for keyword_data in cat_keywords:
                                aliases = ", ".join(keyword_data.get('aliases', []))
                                logger.info(f"     - {keyword_data['keyword']} (별칭: {aliases})")
                        
                        return result
                    else:
                        logger.error(f"❌ 키워드 조회 실패: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"❌ 키워드 조회 오류: {e}")
            return None

    async def test_comprehensive_corrections(self):
        """종합적인 키워드 교정 테스트"""
        test_cases = [
            {
                "name": "문학 작품",
                "texts": [
                    "김화영이 번역한 카뮤의 이방인을 읽었습니다",
                    "도스또예프스키의 죄와 벌은 명작입니다",
                    "똘스또이의 전쟁과 평화는 긴 소설입니다",
                    "헤밍웨이의 노인과 바다를 추천합니다"
                ]
            },
            {
                "name": "대학교",
                "texts": [
                    "서울대 컴퓨터공학과에 진학하고 싶습니다",
                    "연세대에서 경영학을 전공했습니다",
                    "고려대 의과대학이 유명합니다",
                    "카이스트에서 로봇공학을 연구합니다"
                ]
            },
            {
                "name": "기술",
                "texts": [
                    "딥 러닝으로 이미지 분류를 합니다",
                    "머신 러닝 알고리즘을 구현했습니다",
                    "에이아이 기술이 발전하고 있습니다",
                    "블록 체인 기술을 공부하고 있습니다"
                ]
            },
            {
                "name": "기업",
                "texts": [
                    "네이버에서 검색 엔진을 개발합니다",
                    "카카오톡을 많이 사용합니다",
                    "삼성 스마트폰을 구매했습니다",
                    "엘지전자 냉장고가 좋습니다"
                ]
            },
            {
                "name": "복합 문장",
                "texts": [
                    "서울대에서 딥 러닝을 연구하는 카뮤 전공자입니다",
                    "카이스트 출신이 네이버에서 에이아이 개발을 합니다",
                    "연세대 교수가 도스또예프스키의 작품을 분석했습니다",
                    "삼성에서 머신 러닝 기술로 블록 체인을 연구합니다"
                ]
            }
        ]
        
        total_tests = 0
        successful_corrections = 0
        
        for test_case in test_cases:
            logger.info(f"\n🧪 {test_case['name']} 테스트:")
            
            for i, text in enumerate(test_case['texts'], 1):
                total_tests += 1
                logger.info(f"\n📝 테스트 {i}: {text}")
                
                correction_result = await self.correct_text(text)
                
                if correction_result:
                    original = correction_result.get('original_text', '')
                    corrected = correction_result.get('corrected_text', '')
                    corrections = correction_result.get('corrections', [])
                    
                    if corrections:
                        successful_corrections += 1
                        logger.info(f"   ✅ 교정됨: {corrected}")
                        for correction in corrections:
                            logger.info(f"      '{correction['original']}' → '{correction['corrected']}' (신뢰도: {correction['confidence']:.3f})")
                    else:
                        logger.info(f"   ➡️ 교정 불필요: {original}")
                else:
                    logger.error(f"   ❌ 교정 실패")
        
        logger.info(f"\n📊 종합 테스트 결과:")
        logger.info(f"   총 테스트: {total_tests}개")
        logger.info(f"   교정 성공: {successful_corrections}개")
        logger.info(f"   성공률: {(successful_corrections/total_tests)*100:.1f}%")

    async def correct_text(self, text: str, confidence_threshold: float = 0.8) -> dict:
        """텍스트 키워드 교정"""
        call_id = getattr(self, 'call_id', 'comprehensive_test')
        
        payload = {
            "call_id": call_id,
            "text": text,
            "enable_fuzzy_matching": True,
            "min_similarity": 0.7
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.post(
                    f"{self.server_url}/keywords/correct",
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ 교정 실패 ({response.status}): {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ 교정 요청 실패: {e}")
            return None

    async def test_performance_benchmark(self):
        """키워드 교정 성능 벤치마크"""
        test_texts = [
            "카뮤의 작품을 서울대에서 연구합니다",
            "딥 러닝으로 네이버 검색을 개선합니다",
            "도스또예프스키를 카이스트에서 분석합니다"
        ]
        
        logger.info("🚀 성능 벤치마크 테스트:")
        
        total_time = 0
        test_count = len(test_texts) * 10  # 각 텍스트를 10번씩 테스트
        
        for text in test_texts:
            logger.info(f"\n📝 테스트 텍스트: {text}")
            
            times = []
            for i in range(10):
                start_time = time.time()
                result = await self.correct_text(text)
                end_time = time.time()
                
                if result:
                    processing_time = end_time - start_time
                    times.append(processing_time)
                    total_time += processing_time
            
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                logger.info(f"   평균: {avg_time*1000:.1f}ms")
                logger.info(f"   최소: {min_time*1000:.1f}ms")
                logger.info(f"   최대: {max_time*1000:.1f}ms")
        
        if test_count > 0 and total_time > 0:
            overall_avg = (total_time / test_count) * 1000
            logger.info(f"\n📊 전체 성능:")
            logger.info(f"   평균 처리시간: {overall_avg:.1f}ms")
            logger.info(f"   처리량: {1000/overall_avg:.1f} 요청/초")
        else:
            logger.warning(f"\n⚠️ 성능 측정 실패: 유효한 결과가 없습니다.")

    async def test_real_audio_with_correction(self):
        """실제 오디오 파일 + 키워드 교정 테스트"""
        audio_file = "test_korean_sample1.wav"
        
        if not os.path.exists(audio_file):
            logger.warning(f"⚠️ 오디오 파일 {audio_file}이 없습니다. 샘플 오디오로 대체합니다.")
            return await self.test_with_sample_audio()
        
        try:
            # WAV 파일을 PCM 16kHz로 변환 (간단한 예제)
            import wave
            
            with wave.open(audio_file, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                
            # Base64 인코딩
            audio_b64 = base64.b64encode(frames).decode('utf-8')
            
            logger.info(f"🎵 실제 오디오 파일 테스트: {audio_file}")
            
            # STT 전사
            async with aiohttp.ClientSession() as session:
                stt_payload = {
                    "audio_data": audio_b64,
                    "language": "ko",
                    "audio_format": "pcm_16khz",
                    "enable_confidence": True
                }
                
                async with session.post(
                    f"{self.server_url}/infer/utterance",
                    json=stt_payload
                ) as response:
                    
                    if response.status == 200:
                        stt_result = await response.json()
                        logger.info(f"🎤 STT 결과: {stt_result['text']}")
                        logger.info(f"⚡ STT 시간: {stt_result['processing_time']:.3f}초")
                        
                        # 키워드 교정
                        correction_result = await self.correct_text(stt_result['text'])
                        
                        if correction_result:
                            logger.info(f"✅ 교정 결과: {correction_result['corrected_text']}")
                            logger.info(f"⚡ 교정 시간: {correction_result['processing_time']:.3f}초")
                            
                            total_time = stt_result['processing_time'] + correction_result['processing_time']
                            logger.info(f"📊 총 처리시간: {total_time:.3f}초")
                        else:
                            logger.error("❌ 키워드 교정 실패")
                    else:
                        logger.error(f"❌ STT 실패: {response.status}")
                        
        except Exception as e:
            logger.error(f"❌ 실제 오디오 테스트 실패: {e}")

    async def test_with_sample_audio(self):
        """실제 샘플 오디오 파일로 STT + 키워드 교정 테스트"""
        # 사용 가능한 샘플 오디오 파일들
        sample_files = [
            "test_korean_sample1.wav",  # "김화영이 번역하고 책세상에서 출간된 카뮤의 전집"
            "test_korean_sample2.wav"   # 다른 샘플이 있다면
        ]
        
        # 존재하는 파일 찾기
        audio_file = None
        for file in sample_files:
            if os.path.exists(file):
                audio_file = file
                break
        
        if not audio_file:
            logger.error("❌ 사용 가능한 한국어 샘플 오디오 파일이 없습니다.")
            logger.info("   필요한 파일: test_korean_sample1.wav 또는 test_korean_sample2.wav")
            return
        
        logger.info(f"🎵 실제 샘플 오디오 테스트: {audio_file}")
        
        try:
            # 오디오 파일 변환 및 로드
            from scipy.io import wavfile
            import librosa
            
            # librosa로 오디오 로드 (자동으로 16kHz 변환)
            audio_data, sample_rate = librosa.load(audio_file, sr=16000, dtype=np.float32)
            
            # float32를 int16으로 변환
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Base64 인코딩
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
            
            logger.info(f"   오디오 정보: {len(audio_data)/sample_rate:.2f}초, {sample_rate}Hz")
            
            # STT 전사
            async with aiohttp.ClientSession() as session:
                stt_payload = {
                    "audio_data": audio_b64,
                    "language": "ko",
                    "audio_format": "pcm_16khz",
                    "enable_confidence": True
                }
                
                start_time = time.time()
                async with session.post(
                    f"{self.server_url}/infer/utterance",
                    json=stt_payload
                ) as response:
                    
                    if response.status == 200:
                        stt_result = await response.json()
                        stt_time = time.time() - start_time
                        
                        original_text = stt_result['text']
                        logger.info(f"🎤 STT 원본: {original_text}")
                        logger.info(f"⚡ STT 처리시간: {stt_time:.3f}초")
                        logger.info(f"📊 STT 신뢰도: {stt_result.get('confidence', 'N/A')}")
                        
                        # 키워드 교정 적용
                        correction_result = await self.correct_text(original_text)
                        
                        if correction_result:
                            corrected_text = correction_result['corrected_text']
                            corrections = correction_result.get('corrections', [])
                            correction_time = correction_result['processing_time']
                            
                            logger.info(f"✅ 교정 결과: {corrected_text}")
                            logger.info(f"⚡ 교정 처리시간: {correction_time:.6f}초")
                            
                            if corrections:
                                logger.info(f"🔧 적용된 교정:")
                                for correction in corrections:
                                    logger.info(f"   '{correction['original']}' → '{correction['corrected']}' "
                                              f"(신뢰도: {correction['confidence']:.3f}, "
                                              f"방법: {correction.get('method', 'unknown')})")
                            else:
                                logger.info("   교정이 필요한 키워드가 없습니다.")
                            
                            total_time = stt_time + correction_time
                            logger.info(f"📊 총 처리시간: {total_time:.3f}초")
                            
                            # 원본과 교정본 비교
                            if original_text != corrected_text:
                                logger.info(f"\n📝 비교:")
                                logger.info(f"   원본: {original_text}")
                                logger.info(f"   교정: {corrected_text}")
                            else:
                                logger.info(f"   ➡️ 교정이 필요하지 않았습니다.")
                        else:
                            logger.error("❌ 키워드 교정 실패")
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ STT 실패: {response.status} - {error_text}")
                        
        except ImportError as e:
            logger.error(f"❌ 필요한 라이브러리가 없습니다: {e}")
            logger.info("   설치 명령: pip install librosa scipy")
        except Exception as e:
            logger.error(f"❌ 샘플 오디오 테스트 실패: {e}")

    async def get_keyword_stats(self):
        """키워드 시스템 통계"""
        try:
            async with aiohttp.ClientSession() as session:
                # 후처리 교정기의 통계는 따로 구현되지 않았으므로 간단한 정보만 표시
                call_id = getattr(self, 'call_id', 'comprehensive_test')
                
                # 등록된 키워드 정보로 통계 생성
                keywords_response = await session.get(f"{self.server_url}/keywords/{call_id}")
                if keywords_response.status == 200:
                    result = await keywords_response.json()
                    keywords_dict = result.get('keywords', {})
                    
                    # 카테고리별 통계
                    categories = {}
                    for keyword, details in keywords_dict.items():
                        category = details.get('category', 'unknown')
                        categories[category] = categories.get(category, 0) + 1
                    
                    logger.info(f"📊 키워드 시스템 통계:")
                    logger.info(f"   Call ID: {call_id}")
                    logger.info(f"   총 키워드: {len(keywords_dict)}개")
                    
                    if categories:
                        logger.info(f"   카테고리별:")
                        for category, count in categories.items():
                            logger.info(f"     - {category}: {count}개")
                    
                    return {
                        "call_id": call_id,
                        "total_keywords": len(keywords_dict),
                        "categories": categories,
                        "keywords": keywords_dict
                    }
                else:
                    logger.error(f"❌ 통계 조회 실패: {keywords_response.status}")
                    return None
        except Exception as e:
            logger.error(f"❌ 통계 조회 오류: {e}")
            return None

async def main():
    """메인 실행 함수"""
    client = KeywordBoostingClient()
    
    # 서버 상태 확인
    logger.info("🔍 서버 연결 테스트:")
    if not await client.test_server_health():
        logger.error("❌ 서버에 연결할 수 없습니다.")
        return
    
    # 키워드 데이터베이스 설정
    logger.info("\n🚀 키워드 데이터베이스 설정:")
    await client.setup_comprehensive_keywords()
    
    # 등록된 키워드 확인
    logger.info("\n📋 등록된 키워드 현황:")
    await client.get_all_keywords()
    
    # 종합적인 교정 테스트
    await client.test_comprehensive_corrections()
    
    # 성능 벤치마크
    logger.info("\n🚀 성능 벤치마크:")
    await client.test_performance_benchmark()
    
    # 실제 오디오 + 키워드 교정 테스트
    logger.info("\n🎵 오디오 + 키워드 교정 통합 테스트:")
    await client.test_real_audio_with_correction()
    
    # 키워드 시스템 통계
    logger.info("\n📊 최종 통계:")
    await client.get_keyword_stats()
    
    logger.info("\n✅ 모든 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main()) 