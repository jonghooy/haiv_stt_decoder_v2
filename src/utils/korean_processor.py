#!/usr/bin/env python3
"""
Korean Language Processing Module for STT
Specialized processing for Korean speech recognition and text processing
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Korean-specific constants  
KOREAN_VAD_PARAMETERS = {
    "threshold": 0.3,  # Lower threshold for Korean speech patterns
    "min_speech_duration_ms": 100,  # Much shorter minimum for Korean syllables
    "max_speech_duration_s": float('inf'),
    "min_silence_duration_ms": 1000,  # Shorter silence for Korean rhythm
    "speech_pad_ms": 200,  # Reduced padding for Korean
}

# Optimized VAD parameters for different audio durations
DURATION_OPTIMIZED_VAD = {
    "short": {  # < 2 seconds
        "threshold": 0.2,
        "min_speech_duration_ms": 50,
        "max_speech_duration_s": float('inf'),
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 100,
    },
    "medium": {  # 2-5 seconds
        "threshold": 0.3,
        "min_speech_duration_ms": 100,
        "max_speech_duration_s": float('inf'),
        "min_silence_duration_ms": 800,
        "speech_pad_ms": 150,
    },
    "long": {  # > 5 seconds
        "threshold": 0.35,
        "min_speech_duration_ms": 150,
        "max_speech_duration_s": float('inf'),
        "min_silence_duration_ms": 1200,
        "speech_pad_ms": 250,
    }
}

KOREAN_WHISPER_PARAMETERS = {
    "beam_size": 5,
    "patience": 1.0,
    "length_penalty": 1.0,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": True,
}

# Korean text normalization patterns
KOREAN_NORMALIZATION_PATTERNS = [
    # Remove extra whitespace
    (r'\s+', ' '),
    # Normalize Korean punctuation
    (r'[，]', ','),
    (r'[。]', '.'),
    (r'[；]', ';'),
    (r'[：]', ':'),
    (r'[！]', '!'),
    (r'[？]', '?'),
    # Remove English letters mixed inappropriately
    (r'([가-힣])\s*([a-zA-Z])\s*([가-힣])', r'\1\3'),
    # Normalize multiple periods
    (r'\.{2,}', '...'),
    # Fix spacing around punctuation
    (r'\s*([,.!?;:])\s*', r'\1 '),
    # Remove leading/trailing whitespace from lines
    (r'^\s+|\s+$', ''),
]

@dataclass
class KoreanProcessingConfig:
    """Configuration for Korean language processing"""
    use_korean_vad: bool = True
    normalize_text: bool = True
    enhance_word_boundaries: bool = True
    filter_confidence_threshold: float = 0.3
    min_word_length: int = 1
    enable_post_processing: bool = True
    batch_processing_mode: bool = False


class KoreanSTTProcessor:
    """Korean-specialized STT processor with enhanced features"""
    
    def __init__(self, config: Optional[KoreanProcessingConfig] = None):
        self.config = config or KoreanProcessingConfig()
        logger.info("Korean STT Processor initialized")
    
    def get_korean_vad_parameters(self, audio_duration: Optional[float] = None) -> Dict[str, Any]:
        """Get optimized VAD parameters for Korean speech, optionally duration-based"""
        if not self.config.use_korean_vad:
            # Default parameters
            return {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "max_speech_duration_s": float('inf'),
                "min_silence_duration_ms": 2000,
                "speech_pad_ms": 400,
            }
        
        # Use duration-optimized parameters if duration is provided
        if audio_duration is not None:
            if audio_duration < 2.0:
                vad_params = DURATION_OPTIMIZED_VAD["short"].copy()
                logger.debug(f"Using short-audio VAD parameters for {audio_duration}s")
            elif audio_duration <= 5.0:
                vad_params = DURATION_OPTIMIZED_VAD["medium"].copy()
                logger.debug(f"Using medium-audio VAD parameters for {audio_duration}s")
            else:
                vad_params = DURATION_OPTIMIZED_VAD["long"].copy()
                logger.debug(f"Using long-audio VAD parameters for {audio_duration}s")
            
            return vad_params
        
        # Default Korean VAD parameters
        return KOREAN_VAD_PARAMETERS.copy()
    
    def get_korean_whisper_parameters(self) -> Dict[str, Any]:
        """Get optimized Whisper parameters for Korean language"""
        return KOREAN_WHISPER_PARAMETERS.copy()
    
    def normalize_korean_text(self, text: str) -> str:
        """Normalize Korean text with improved formatting"""
        if not self.config.normalize_text:
            return text
        
        normalized = text
        
        # Apply normalization patterns
        for pattern, replacement in KOREAN_NORMALIZATION_PATTERNS:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Specific Korean text improvements
        normalized = self._improve_korean_spacing(normalized)
        normalized = self._fix_korean_punctuation(normalized)
        
        return normalized.strip()
    
    def _improve_korean_spacing(self, text: str) -> str:
        """Improve spacing for Korean text"""
        # Add space after common Korean particles/endings
        patterns = [
            # Space after common endings
            (r'([가-힣])([은는이가을를의도])\s*([가-힣])', r'\1\2 \3'),
            # Space around numbers and Korean text
            (r'([0-9])([가-힣])', r'\1 \2'),
            (r'([가-힣])([0-9])', r'\1 \2'),
        ]
        
        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _fix_korean_punctuation(self, text: str) -> str:
        """Fix Korean punctuation marks"""
        # Ensure proper spacing around punctuation
        result = re.sub(r'([가-힣])([,.!?])', r'\1\2', text)
        result = re.sub(r'([,.!?])([가-힣])', r'\1 \2', result)
        
        return result
    
    def enhance_word_timestamps(self, segments: List[Dict]) -> List[Dict]:
        """Enhance word-level timestamp accuracy for Korean"""
        if not self.config.enhance_word_boundaries:
            return segments
        
        enhanced_segments = []
        
        for segment in segments:
            enhanced_segment = segment.copy()
            
            if 'words' in segment and segment['words']:
                enhanced_words = []
                
                for word_info in segment['words']:
                    enhanced_word = self._process_korean_word_timestamp(word_info)
                    
                    # Filter out very low confidence words if enabled
                    if self._should_keep_word(enhanced_word):
                        enhanced_words.append(enhanced_word)
                
                enhanced_segment['words'] = enhanced_words
            
            # Update segment text based on processed words
            if self.config.normalize_text and enhanced_segment.get('words'):
                segment_text = ' '.join(word['word'] for word in enhanced_segment['words'])
                enhanced_segment['text'] = self.normalize_korean_text(segment_text)
            
            enhanced_segments.append(enhanced_segment)
        
        return enhanced_segments
    
    def _process_korean_word_timestamp(self, word_info: Dict) -> Dict:
        """Process individual Korean word timestamp"""
        enhanced = word_info.copy()
        
        # Clean Korean word
        word = enhanced.get('word', '').strip()
        
        # Remove leading/trailing punctuation for better processing
        clean_word = re.sub(r'^[^\w가-힣]+|[^\w가-힣]+$', '', word)
        
        if clean_word:
            enhanced['word'] = clean_word
            
            # Adjust confidence for Korean characteristics
            if 'probability' in enhanced:
                enhanced['probability'] = self._adjust_korean_confidence(
                    enhanced['probability'], clean_word
                )
        
        return enhanced
    
    def _adjust_korean_confidence(self, confidence: float, word: str) -> float:
        """Adjust confidence score for Korean word characteristics"""
        adjusted = confidence
        
        # Boost confidence for common Korean patterns
        if re.match(r'^[가-힣]+$', word):  # Pure Korean characters
            adjusted = min(1.0, confidence * 1.1)
        
        # Reduce confidence for very short uncertain words
        if len(word) == 1 and confidence < 0.5:
            adjusted = confidence * 0.8
        
        # Boost confidence for longer Korean words
        if len(word) >= 3:
            adjusted = min(1.0, confidence * 1.05)
        
        return adjusted
    
    def _should_keep_word(self, word_info: Dict) -> bool:
        """Determine if a Korean word should be kept based on quality metrics"""
        word = word_info.get('word', '')
        probability = word_info.get('probability', 1.0)
        
        # Basic length check
        if len(word) < self.config.min_word_length:
            return False
        
        # Confidence threshold check
        if probability < self.config.filter_confidence_threshold:
            return False
        
        # Korean-specific checks
        if not re.search(r'[가-힣]', word) and len(word) < 2:
            return False  # Filter out single non-Korean characters
        
        return True
    
    def post_process_transcription(self, text: str, segments: List[Dict]) -> Tuple[str, List[Dict]]:
        """Apply Korean-specific post-processing to transcription results"""
        if not self.config.enable_post_processing:
            return text, segments
        
        # Process text
        processed_text = self.normalize_korean_text(text)
        
        # Process segments
        processed_segments = self.enhance_word_timestamps(segments)
        
        # Final quality checks
        processed_text = self._final_korean_cleanup(processed_text)
        
        return processed_text, processed_segments
    
    def _final_korean_cleanup(self, text: str) -> str:
        """Final cleanup pass for Korean text"""
        # Remove common transcription artifacts
        cleaned = text
        
        # Remove repeated short words (common in speech recognition errors)
        cleaned = re.sub(r'\b(\w{1,2})\s+\1\b', r'\1', cleaned)
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        
        # Ensure proper sentence ending
        if cleaned and not cleaned[-1] in '.!?':
            # Don't add period if it ends with Korean particle
            if not re.search(r'[은는이가을를의도요]$', cleaned):
                cleaned += '.'
        
        return cleaned.strip()
    
    def get_language_detection_params(self) -> Dict[str, Any]:
        """Get optimized parameters for Korean language detection"""
        return {
            "language": "ko",
            "task": "transcribe",
            # Boost Korean language probability
            "language_detection_threshold": 0.5,
            "language_detection_segments": 1,
        }
    
    def get_korean_optimized_params(self) -> Dict[str, Any]:
        """Get all Korean-optimized parameters for STT processing"""
        params = {
            "vad_parameters": self.get_korean_vad_parameters(),
            "whisper_parameters": self.get_korean_whisper_parameters(),
            "language_params": self.get_language_detection_params(),
            "processing_config": {
                "normalize_text": self.config.normalize_text,
                "enhance_word_boundaries": self.config.enhance_word_boundaries,
                "post_processing": self.config.enable_post_processing,
                "batch_mode": self.config.batch_processing_mode
            }
        }
        
        # Batch processing optimizations
        if self.config.batch_processing_mode:
            # Lower beam size for faster batch processing
            params["whisper_parameters"]["beam_size"] = 1
            # Simpler temperature schedule
            params["whisper_parameters"]["temperature"] = [0.0]
            # Lower confidence threshold for batch processing
            params["processing_config"]["filter_confidence_threshold"] = 0.25
            logger.debug("Applied batch processing optimizations")
        
        return params


# Global Korean processor instance
_korean_processor: Optional[KoreanSTTProcessor] = None

def get_korean_processor(config: Optional[KoreanProcessingConfig] = None) -> KoreanSTTProcessor:
    """Get global Korean processor instance"""
    global _korean_processor
    if _korean_processor is None:
        _korean_processor = KoreanSTTProcessor(config)
    return _korean_processor

def create_korean_optimized_params() -> Dict[str, Any]:
    """Create Korean-optimized parameters for STT processing"""
    processor = get_korean_processor()
    
    params = {}
    params.update(processor.get_korean_whisper_parameters())
    params.update(processor.get_language_detection_params())
    params["vad_parameters"] = processor.get_korean_vad_parameters()
    
    return params


if __name__ == "__main__":
    # Test Korean processor
    processor = KoreanSTTProcessor()
    
    # Test text normalization
    test_text = "안녕하세요    ，이것은    테스트입니다。"
    normalized = processor.normalize_korean_text(test_text)
    print(f"Original: {test_text}")
    print(f"Normalized: {normalized}")
    
    # Test parameters
    vad_params = processor.get_korean_vad_parameters()
    whisper_params = processor.get_korean_whisper_parameters()
    
    print("\nKorean VAD Parameters:")
    for key, value in vad_params.items():
        print(f"  {key}: {value}")
    
    print("\nKorean Whisper Parameters:")
    for key, value in whisper_params.items():
        print(f"  {key}: {value}") 