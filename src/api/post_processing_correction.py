"""
Post-processing Based Keyword Correction System
후처리 기반 키워드 교정 시스템

Whisper 전사 후 텍스트를 분석하여 키워드를 감지하고 교정하는 시스템입니다.
기존의 모델 레벨 부스팅 대신, 출력된 텍스트를 후처리하여 정확도를 향상시킵니다.
"""

import logging
import re
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
import threading
from collections import defaultdict, Counter

import numpy as np
from fuzzywuzzy import fuzz, process

logger = logging.getLogger(__name__)


@dataclass
class KeywordEntry:
    """키워드 정보"""
    keyword: str
    aliases: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7
    category: str = "general"
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CorrectionResult:
    """교정 결과"""
    original_text: str
    corrected_text: str
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    keywords_detected: List[str] = field(default_factory=list)


class PostProcessingCorrector:
    """후처리 기반 키워드 교정기"""
    
    def __init__(self, storage_path: str = "./data/keywords"):
        """
        초기화
        
        Args:
            storage_path: 키워드 데이터 저장 경로
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 키워드 사전 캐시
        self.keyword_cache: Dict[str, Dict[str, KeywordEntry]] = {}
        self.cache_lock = threading.RLock()
        
        # 교정 통계
        self.correction_stats = {
            'total_corrections': 0,
            'successful_corrections': 0,
            'false_positives': 0,
            'processing_time_total': 0.0
        }
        self.stats_lock = threading.Lock()
        
        # 한국어 특화 패턴
        self.korean_patterns = {
            # 조사 패턴 (키워드 + 조사)
            'josa_patterns': [
                r'이$', r'가$', r'을$', r'를$', r'은$', r'는$',
                r'에$', r'의$', r'와$', r'과$', r'로$', r'으로$',
                r'만$', r'도$', r'부터$', r'까지$', r'처럼$', r'같이$'
            ],
            # 어미 패턴
            'ending_patterns': [
                r'습니다$', r'입니다$', r'였습니다$', r'있습니다$',
                r'했습니다$', r'됩니다$', r'합니다$'
            ]
        }
        
        logger.info(f"후처리 교정기 초기화 완료 - 저장 경로: {self.storage_path}")
    
    async def register_keywords(self, call_id: str, keywords: List[Dict[str, Any]]) -> bool:
        """
        키워드 등록
        
        Args:
            call_id: 호출 ID
            keywords: 키워드 목록
            
        Returns:
            성공 여부
        """
        try:
            keyword_dict = {}
            
            for kw_data in keywords:
                if isinstance(kw_data, dict):
                    keyword = kw_data.get('keyword', '')
                    aliases = kw_data.get('aliases', [])
                    confidence_threshold = kw_data.get('confidence_threshold', 0.7)
                    category = kw_data.get('category', 'general')
                elif isinstance(kw_data, str):
                    keyword = kw_data
                    aliases = []
                    confidence_threshold = 0.7
                    category = 'general'
                else:
                    continue
                
                if keyword:
                    keyword_dict[keyword.lower()] = KeywordEntry(
                        keyword=keyword,
                        aliases=aliases,
                        confidence_threshold=confidence_threshold,
                        category=category
                    )
            
            # 캐시에 저장
            with self.cache_lock:
                self.keyword_cache[call_id] = keyword_dict
            
            # 파일에 저장
            await self._save_keywords_to_file(call_id, keyword_dict)
            
            logger.info(f"키워드 등록 완료 - Call ID: {call_id}, 키워드 수: {len(keyword_dict)}")
            return True
            
        except Exception as e:
            logger.error(f"키워드 등록 실패: {e}")
            return False
    
    async def apply_correction(self, 
                             call_id: str, 
                             text: str,
                             enable_fuzzy_matching: bool = True,
                             min_similarity: float = 0.8) -> CorrectionResult:
        """
        텍스트 교정 적용
        
        Args:
            call_id: 호출 ID
            text: 원본 텍스트
            enable_fuzzy_matching: 퍼지 매칭 활성화
            min_similarity: 최소 유사도 임계값
            
        Returns:
            교정 결과
        """
        start_time = datetime.now()
        
        try:
            # 키워드 사전 로드
            keywords = await self._load_keywords(call_id)
            if not keywords:
                return CorrectionResult(
                    original_text=text,
                    corrected_text=text,
                    confidence_score=1.0,
                    processing_time=0.0
                )
            
            corrected_text = text
            corrections = []
            detected_keywords = []
            
            # 1. 정확 매칭 교정
            corrected_text, exact_corrections, exact_detected = await self._apply_exact_matching(
                corrected_text, keywords
            )
            corrections.extend(exact_corrections)
            detected_keywords.extend(exact_detected)
            
            # 2. 퍼지 매칭 교정 (활성화된 경우)
            if enable_fuzzy_matching:
                corrected_text, fuzzy_corrections, fuzzy_detected = await self._apply_fuzzy_matching(
                    corrected_text, keywords, min_similarity
                )
                corrections.extend(fuzzy_corrections)
                detected_keywords.extend(fuzzy_detected)
            
            # 3. 한국어 특화 교정
            corrected_text, korean_corrections, korean_detected = await self._apply_korean_correction(
                corrected_text, keywords
            )
            corrections.extend(korean_corrections)
            detected_keywords.extend(korean_detected)
            
            # 중복 제거
            detected_keywords = list(set(detected_keywords))
            
            # 신뢰도 계산
            confidence_score = self._calculate_confidence(text, corrected_text, corrections)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 통계 업데이트
            with self.stats_lock:
                self.correction_stats['total_corrections'] += len(corrections)
                if corrections:
                    self.correction_stats['successful_corrections'] += 1
                self.correction_stats['processing_time_total'] += processing_time
            
            result = CorrectionResult(
                original_text=text,
                corrected_text=corrected_text,
                corrections=corrections,
                confidence_score=confidence_score,
                processing_time=processing_time,
                keywords_detected=detected_keywords
            )
            
            logger.info(f"교정 완료 - 교정 수: {len(corrections)}, 키워드 감지: {len(detected_keywords)}")
            return result
            
        except Exception as e:
            logger.error(f"교정 처리 실패: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                confidence_score=0.0,
                processing_time=processing_time
            )
    
    async def _apply_exact_matching(self, 
                                  text: str, 
                                  keywords: Dict[str, KeywordEntry]) -> Tuple[str, List[Dict], List[str]]:
        """정확 매칭 교정 적용"""
        corrected_text = text
        corrections = []
        detected_keywords = []
        
        for keyword, entry in keywords.items():
            if not entry.enabled:
                continue
            
            # 원본 키워드 매칭
            if keyword in text.lower():
                detected_keywords.append(entry.keyword)
                # 대소문자 교정
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                if pattern.search(corrected_text):
                    old_text = corrected_text
                    corrected_text = pattern.sub(entry.keyword, corrected_text)
                    if old_text != corrected_text:
                        corrections.append({
                            'type': 'exact_match',
                            'original': keyword,
                            'corrected': entry.keyword,
                            'confidence': 1.0,
                            'method': 'case_correction'
                        })
            
            # 별칭 매칭
            for alias in entry.aliases:
                if alias.lower() in text.lower():
                    detected_keywords.append(entry.keyword)
                    pattern = re.compile(re.escape(alias), re.IGNORECASE)
                    if pattern.search(corrected_text):
                        old_text = corrected_text
                        corrected_text = pattern.sub(entry.keyword, corrected_text)
                        if old_text != corrected_text:
                            corrections.append({
                                'type': 'alias_match',
                                'original': alias,
                                'corrected': entry.keyword,
                                'confidence': 0.95,
                                'method': 'alias_replacement'
                            })
        
        return corrected_text, corrections, detected_keywords
    
    async def _apply_fuzzy_matching(self, 
                                  text: str, 
                                  keywords: Dict[str, KeywordEntry],
                                  min_similarity: float = 0.8) -> Tuple[str, List[Dict], List[str]]:
        """퍼지 매칭 교정 적용"""
        corrected_text = text
        corrections = []
        detected_keywords = []
        
        # 텍스트를 단어 단위로 분리
        words = re.findall(r'\S+', text)
        
        for keyword, entry in keywords.items():
            if not entry.enabled:
                continue
            
            # 각 단어와 키워드 비교
            for word in words:
                # 구두점 제거
                clean_word = re.sub(r'[^\w\s]', '', word).lower()
                if len(clean_word) < 2:  # 너무 짧은 단어는 스킵
                    continue
                
                # 유사도 계산
                similarity = fuzz.ratio(clean_word, keyword.lower()) / 100.0
                
                if similarity >= min_similarity:
                    detected_keywords.append(entry.keyword)
                    
                    # 원본 텍스트에서 해당 단어를 키워드로 교체
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    if pattern.search(corrected_text):
                        old_text = corrected_text
                        corrected_text = pattern.sub(entry.keyword, corrected_text, count=1)
                        if old_text != corrected_text:
                            corrections.append({
                                'type': 'fuzzy_match',
                                'original': word,
                                'corrected': entry.keyword,
                                'confidence': similarity,
                                'method': 'fuzzy_replacement',
                                'similarity': similarity
                            })
        
        return corrected_text, corrections, detected_keywords
    
    async def _apply_korean_correction(self, 
                                     text: str, 
                                     keywords: Dict[str, KeywordEntry]) -> Tuple[str, List[Dict], List[str]]:
        """한국어 특화 교정 적용"""
        corrected_text = text
        corrections = []
        detected_keywords = []
        
        for keyword, entry in keywords.items():
            if not entry.enabled:
                continue
            
            # 조사가 붙은 형태 찾기
            for josa_pattern in self.korean_patterns['josa_patterns']:
                # 키워드 + 조사 패턴
                pattern = keyword + josa_pattern.replace('$', '')
                if pattern in text.lower():
                    detected_keywords.append(entry.keyword)
                    
                    # 정확한 키워드 형태로 교정 (조사는 유지)
                    full_pattern = re.compile(
                        re.escape(keyword) + josa_pattern.replace('$', ''),
                        re.IGNORECASE
                    )
                    
                    matches = full_pattern.findall(text)
                    for match in matches:
                        # 키워드 부분만 정확한 형태로 교정
                        josa_part = match[len(keyword):]
                        corrected_form = entry.keyword + josa_part
                        
                        old_text = corrected_text
                        corrected_text = corrected_text.replace(match, corrected_form)
                        if old_text != corrected_text:
                            corrections.append({
                                'type': 'korean_josa',
                                'original': match,
                                'corrected': corrected_form,
                                'confidence': 0.9,
                                'method': 'korean_morphology'
                            })
            
            # 어미가 붙은 형태 처리
            for ending_pattern in self.korean_patterns['ending_patterns']:
                # 키워드가 어미와 함께 나타나는 경우
                words = text.split()
                for i, word in enumerate(words):
                    if keyword.lower() in word.lower() and re.search(ending_pattern, word):
                        detected_keywords.append(entry.keyword)
                        # 필요시 교정 로직 추가
        
        return corrected_text, corrections, detected_keywords
    
    def _calculate_confidence(self, 
                            original: str, 
                            corrected: str, 
                            corrections: List[Dict]) -> float:
        """교정 신뢰도 계산"""
        if not corrections:
            return 1.0
        
        # 기본 신뢰도
        base_confidence = 0.8
        
        # 교정 품질 점수
        quality_scores = []
        for correction in corrections:
            correction_confidence = correction.get('confidence', 0.5)
            quality_scores.append(correction_confidence)
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            # 교정 수가 적을수록 신뢰도 높음
            correction_penalty = min(len(corrections) * 0.05, 0.3)
            final_confidence = min(base_confidence * avg_quality - correction_penalty, 1.0)
        else:
            final_confidence = base_confidence
        
        return max(final_confidence, 0.0)
    
    async def _load_keywords(self, call_id: str) -> Dict[str, KeywordEntry]:
        """키워드 로드"""
        # 캐시에서 먼저 확인
        with self.cache_lock:
            if call_id in self.keyword_cache:
                return self.keyword_cache[call_id]
        
        # 파일에서 로드
        file_path = self.storage_path / f"{call_id}.json"
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                keywords = {}
                for keyword, entry_data in data.items():
                    keywords[keyword] = KeywordEntry(
                        keyword=entry_data['keyword'],
                        aliases=entry_data.get('aliases', []),
                        confidence_threshold=entry_data.get('confidence_threshold', 0.7),
                        category=entry_data.get('category', 'general'),
                        enabled=entry_data.get('enabled', True)
                    )
                
                # 캐시에 저장
                with self.cache_lock:
                    self.keyword_cache[call_id] = keywords
                
                return keywords
                
            except Exception as e:
                logger.error(f"키워드 파일 로드 실패 {file_path}: {e}")
        
        return {}
    
    async def _save_keywords_to_file(self, call_id: str, keywords: Dict[str, KeywordEntry]):
        """키워드를 파일에 저장"""
        file_path = self.storage_path / f"{call_id}.json"
        
        try:
            data = {}
            for keyword, entry in keywords.items():
                data[keyword] = {
                    'keyword': entry.keyword,
                    'aliases': entry.aliases,
                    'confidence_threshold': entry.confidence_threshold,
                    'category': entry.category,
                    'enabled': entry.enabled,
                    'created_at': entry.created_at.isoformat()
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"키워드 파일 저장 실패 {file_path}: {e}")
    
    async def delete_keywords(self, call_id: str) -> bool:
        """키워드 삭제"""
        try:
            # 캐시에서 제거
            with self.cache_lock:
                if call_id in self.keyword_cache:
                    del self.keyword_cache[call_id]
            
            # 파일 삭제
            file_path = self.storage_path / f"{call_id}.json"
            if file_path.exists():
                file_path.unlink()
            
            logger.info(f"키워드 삭제 완료 - Call ID: {call_id}")
            return True
            
        except Exception as e:
            logger.error(f"키워드 삭제 실패: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """교정 통계 조회"""
        with self.stats_lock:
            stats = self.correction_stats.copy()
            
        # 평균 처리 시간 계산
        if stats['successful_corrections'] > 0:
            stats['avg_processing_time'] = stats['processing_time_total'] / stats['successful_corrections']
        else:
            stats['avg_processing_time'] = 0.0
        
        # 성공률 계산
        total_attempts = stats['successful_corrections'] + stats['false_positives']
        if total_attempts > 0:
            stats['success_rate'] = stats['successful_corrections'] / total_attempts
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    async def get_keywords(self, call_id: str) -> Optional[Dict[str, KeywordEntry]]:
        """키워드 조회"""
        return await self._load_keywords(call_id)


# 전역 인스턴스
_post_processing_corrector: Optional[PostProcessingCorrector] = None


def get_post_processing_corrector() -> PostProcessingCorrector:
    """후처리 교정기 인스턴스 반환"""
    global _post_processing_corrector
    if _post_processing_corrector is None:
        _post_processing_corrector = PostProcessingCorrector()
    return _post_processing_corrector


async def apply_keyword_correction(call_id: str, 
                                 text: str,
                                 **kwargs) -> CorrectionResult:
    """키워드 교정 적용 (편의 함수)"""
    corrector = get_post_processing_corrector()
    return await corrector.apply_correction(call_id, text, **kwargs) 