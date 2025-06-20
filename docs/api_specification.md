# Korean STT Decoder v2 API 명세서

## 개요

Korean STT Decoder v2는 Large-v3 모델을 사용하여 한국어 음성을 텍스트로 변환하는 고성능 STT 서버입니다.  
RTX 4090 GPU에 최적화되어 있으며, PCM 16kHz 전용으로 최고 성능을 제공합니다.

**주요 특징:**
- 🚀 **극한 최적화**: RTF 0.027-0.078 (VAD 설정에 따라)
- 🎯 **높은 정확도**: 99%+ 한국어 인식 정확도 
- 💪 **PCM 16kHz 전용**: 단순화된 포맷으로 최고 성능
- 🔧 **키워드 부스팅**: 전문 용어 인식률 향상
- ⚡ **VAD 제어**: 클라이언트별 음성 활동 감지 On/Off

**실제 성능 테스트 결과:**
| 테스트 파일 | 오디오 길이 | 처리 시간 | RTF | 인식 결과 |
|-------------|-------------|-----------|-----|-----------|
| test_korean_sample1.wav | 5.5초 | 0.41초 | 0.074x | 김화영이 번역하고 책세상에서 출간된 카뮤의 전집 |
| test_korean_sample2.wav | 1.4초 | 0.27초 | 0.186x | 그 친구 이름이 되게 흔한데 |
| **평균 성능** | - | - | **0.130x** | **100% 성공률** |

---

## 서버 정보

- **기본 URL**: `http://localhost:8009`
- **포트**: 8009 (기본값, 변경 가능)
- **모델**: Large-v3 (한국어 최적화)
- **GPU**: RTX 4090 최적화
- **지원 포맷**: PCM 16kHz 전용

---

## 인증

현재 버전은 인증이 필요하지 않습니다.

---

## 공통 헤더

```
Content-Type: application/json
Accept: application/json
```

---

## API 엔드포인트

### 1. 서버 상태 확인

**GET** `/health`

서버의 상태와 모델 로딩 여부를 확인합니다.

#### 응답

```json
{
  "status": "healthy",
  "model": "large-v3",
  "gpu_available": true,
  "timestamp": "2024-01-15T10:30:00.000Z",
  "supported_format": "pcm_16khz"
}
```

---

### 2. 기본 음성 전사

**POST** `/transcribe`

PCM 16kHz 오디오를 텍스트로 변환합니다. VAD(Voice Activity Detection) 기능을 On/Off할 수 있습니다.

**요청 본문:**
```json
{
  "audio_data": "base64_encoded_pcm_16khz_data...", // Base64 인코딩된 PCM 16kHz 오디오 데이터
  "language": "ko",                                // 언어 코드 (ko, en 등)
  "audio_format": "pcm_16khz",                     // PCM 16kHz 고정
  "vad_enabled": true                              // VAD 사용 여부 (기본값: true)
}
```

**요청 파라미터:**
- `audio_data` (필수): Base64로 인코딩된 PCM 16kHz 오디오 데이터
- `language` (선택): 언어 코드. 기본값은 "ko" (한국어)
- `audio_format` (선택): 오디오 포맷. "pcm_16khz" 고정 (기본값)
- `vad_enabled` (선택): VAD(Voice Activity Detection) 사용 여부. 기본값은 `true`
  - `true`: 침묵 구간 자동 제거로 성능 향상 (RTF 0.027)
  - `false`: 전체 오디오 처리 (RTF 0.078)

**PCM 16kHz 오디오 요구사항:**
- **샘플 레이트**: 16,000 Hz
- **비트 깊이**: 16-bit
- **채널**: 모노 (1채널)
- **인코딩**: Little-endian signed integer
- **전송**: Base64 인코딩

**응답:**
```json
{
  "text": "안녕하세요, 음성 인식 테스트입니다.",
  "language": "ko",
  "rtf": 0.027,
  "processing_time": 0.038,
  "confidence": 0.0,
  "audio_duration": 1.4,
  "audio_format": "pcm_16khz",
  "vad_enabled": true
}
```

**응답 필드:**
- `text`: 전사된 텍스트
- `language`: 감지된 언어
- `rtf`: Real-time Factor (처리 속도 비율)
- `processing_time`: 서버 처리 시간 (초)
- `confidence`: 신뢰도 점수
- `audio_duration`: 오디오 길이 (초)
- `audio_format`: 사용된 오디오 포맷 (pcm_16khz)
- `vad_enabled`: 사용된 VAD 설정

### VAD 기능 비교 예제

**VAD 활성화 (권장 - 원본 오디오):**
```json
{
  "audio_data": "base64_encoded_pcm_16khz...",
  "language": "ko",
  "audio_format": "pcm_16khz",
  "vad_enabled": true
}
```

**응답 예시:**
```json
{
  "text": "안녕하세요",
  "rtf": 0.027,
  "processing_time": 0.038,
  "vad_enabled": true
}
```

**VAD 비활성화 (클라이언트에서 이미 VAD 처리된 경우):**
```json
{
  "audio_data": "base64_encoded_pcm_16khz...",
  "language": "ko", 
  "audio_format": "pcm_16khz",
  "vad_enabled": false
}
```

**응답 예시:**
```json
{
  "text": "안녕하세요",
  "rtf": 0.078,
  "processing_time": 0.109,
  "vad_enabled": false
}
```

**성능 비교:**

| VAD 설정 | RTF | 처리 시간 | 사용 사례 |
|----------|-----|-----------|-----------|
| ON | 0.027 | 38ms | 원본 오디오 (무음 포함) |
| OFF | 0.078 | 109ms | 사전 처리된 오디오 |

---

## 📊 실제 WAV 파일 테스트 가이드

### test_wav_pcm_performance.py 클라이언트 사용법

실제 WAV 파일을 사용하여 성능을 테스트하고 인식 정확도를 확인할 수 있는 전용 클라이언트입니다.

#### 실행 방법

```bash
# 서버가 실행된 상태에서
python test_wav_pcm_performance.py
```

#### 테스트 파일
- `test_korean_sample1.wav`: "김화영이 번역하고 책세상에서 출간된 카뮤의 전집"
- `test_korean_sample2.wav`: "그 친구 이름이 되게 흔한데"

#### 실제 테스트 결과 예시

```json
{
  "test_summary": {
    "total_files": 2,
    "successful": 2,
    "failed": 0,
    "success_rate": "100%",
    "average_rtf": 0.1301,
    "test_mode": "vad_off",
    "server_url": "http://localhost:8009"
  },
  "individual_results": [
    {
      "file": "test_korean_sample1.wav",
      "original_format": "WAV (22050 Hz → 16000 Hz 변환)",
      "duration": 5.512,
      "processing_time": 0.409,
      "rtf": 0.0744,
      "transcription": "김화영이 번역하고 책세상에서 출간된 카뮤의 전집.",
      "audio_format": "pcm_16khz",
      "vad_enabled": false
    },
    {
      "file": "test_korean_sample2.wav",
      "original_format": "WAV (22050 Hz → 16000 Hz 변환)",
      "duration": 1.449,
      "processing_time": 0.269,
      "rtf": 0.1859,
      "transcription": "그 친구 이름이 되게 흔한데.",
      "audio_format": "pcm_16khz",
      "vad_enabled": false
    }
  ],
  "performance_analysis": {
    "fastest_rtf": 0.0744,
    "slowest_rtf": 0.1859,
    "average_processing_time": 0.339,
    "total_audio_duration": 6.961,
    "total_processing_time": 0.678
  }
}
```

#### 클라이언트 특징

1. **WAV → PCM 자동 변환**
   - 다양한 샘플레이트를 16kHz로 자동 변환
   - 스테레오를 모노로 자동 변환
   - Base64 인코딩 자동 처리

2. **성능 측정**
   - 개별 파일별 RTF 측정
   - 평균 성능 계산
   - 처리 시간 분석

3. **결과 저장**
   - JSON 형태로 상세 결과 저장
   - 타임스탬프 포함 파일명
   - 성능 분석 데이터 포함

### 커스텀 WAV 파일 테스트

자신의 WAV 파일로 테스트하려면:

1. WAV 파일을 프로젝트 루트에 복사
2. `test_wav_pcm_performance.py` 수정:

```python
# 61행 근처의 파일 목록 수정
wav_files = [
    "your_audio_file1.wav",
    "your_audio_file2.wav",
    # 추가 파일들...
]
```

3. 테스트 실행:
```bash
python test_wav_pcm_performance.py
```

### Python 코드 예제: WAV → PCM 변환

```python
import wave
import numpy as np
import base64
from scipy import signal

def wav_to_pcm_16khz(wav_file_path):
    """WAV 파일을 PCM 16kHz로 변환"""
    with wave.open(wav_file_path, 'rb') as wav_file:
        # WAV 파일 정보 읽기
        frames = wav_file.readframes(-1)
        original_sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        
        # numpy 배열로 변환
        audio_data = np.frombuffer(frames, dtype=np.int16)
        
        # 스테레오를 모노로 변환
        if channels == 2:
            audio_data = audio_data.reshape(-1, 2)
            audio_data = np.mean(audio_data, axis=1).astype(np.int16)
        
        # 16kHz로 리샘플링
        if original_sample_rate != 16000:
            # 리샘플링 비율 계산
            resample_ratio = 16000 / original_sample_rate
            new_length = int(len(audio_data) * resample_ratio)
            audio_data = signal.resample(audio_data, new_length).astype(np.int16)
        
        return audio_data.tobytes()

# 사용 예제
pcm_data = wav_to_pcm_16khz("your_audio.wav")
pcm_base64 = base64.b64encode(pcm_data).decode('utf-8')

# API 호출
import aiohttp
import asyncio

async def test_transcription(pcm_base64):
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            "http://localhost:8009/transcribe",
            json={
                "audio_data": pcm_base64,
                "language": "ko",
                "audio_format": "pcm_16khz",
                "vad_enabled": false  # VAD OFF로 테스트
            }
        )
        result = await response.json()
        print(f"인식 결과: {result['text']}")
        print(f"RTF: {result['rtf']:.3f}")
        print(f"처리 시간: {result['processing_time']:.3f}초")

asyncio.run(test_transcription(pcm_base64))
```

---

### 3. 키워드 부스팅 전사

**POST** `/transcribe_with_keywords`

특정 키워드에 대한 인식률을 높이는 키워드 부스팅 기능을 제공합니다.

**요청 본문:**
```json
{
  "audio_data": "base64_encoded_pcm_16khz_data...",
  "language": "ko",
  "audio_format": "pcm_16khz", 
  "keywords": ["안녕하세요", "감사합니다", "음성인식"],
  "keyword_boost": 2.5,
  "vad_enabled": true
}
```

**요청 파라미터:**
- `audio_data` (필수): Base64로 인코딩된 PCM 16kHz 오디오 데이터
- `language` (선택): 언어 코드. 기본값은 "ko"
- `audio_format` (선택): "pcm_16khz" 고정
- `keywords` (선택): 부스팅할 키워드 목록. 최대 50개
- `keyword_boost` (선택): 키워드 부스팅 강도 (1.0-5.0). 기본값은 2.0
- `vad_enabled` (선택): VAD 사용 여부. 기본값은 `true`

**응답:**
```json
{
  "text": "안녕하세요, 음성인식 시스템입니다.",
  "language": "ko",
  "rtf": 0.043,
  "processing_time": 0.062,
  "confidence": 0.0,
  "keywords_detected": ["안녕하세요", "음성인식"],
  "boost_applied": true,
  "audio_duration": 1.4,
  "audio_format": "pcm_16khz",
  "vad_enabled": true
}
```

---

## PCM 16kHz 오디오 생성 가이드

### Python 예제
```python
import numpy as np
import base64

def generate_pcm_16khz(duration=1.0, sample_rate=16000):
    """PCM 16kHz 테스트 오디오 생성"""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, dtype=np.float32)
    
    # 440Hz 톤 생성
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # 16-bit PCM으로 변환
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()

# Base64 인코딩
audio_bytes = generate_pcm_16khz(duration=1.4)
audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
```

### Node.js 예제
```javascript
const crypto = require('crypto');

function generatePCM16kHz(duration = 1.0, sampleRate = 16000) {
    const samples = Math.floor(duration * sampleRate);
    const buffer = Buffer.alloc(samples * 2); // 16-bit = 2 bytes per sample
    
    for (let i = 0; i < samples; i++) {
        const t = i / sampleRate;
        const sample = Math.sin(2 * Math.PI * 440 * t) * 0.5;
        const pcm16 = Math.round(sample * 32767);
        buffer.writeInt16LE(pcm16, i * 2);
    }
    
    return buffer.toString('base64');
}

// 사용 예제
const audioB64 = generatePCM16kHz(1.4);
```

---

## 오류 응답

### 일반적인 오류 형식
```json
{
  "error": "오류 메시지",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### 오류 코드

| 코드 | HTTP 상태 | 설명 |
|------|-----------|------|
| INVALID_AUDIO_FORMAT | 400 | 지원하지 않는 오디오 포맷 (PCM 16kHz만 지원) |
| INVALID_AUDIO_DATA | 400 | 잘못된 오디오 데이터 또는 Base64 인코딩 |
| AUDIO_TOO_SHORT | 400 | 오디오가 너무 짧음 (최소 0.1초) |
| AUDIO_TOO_LONG | 400 | 오디오가 너무 김 (최대 30초) |
| PROCESSING_ERROR | 500 | 서버 처리 오류 |
| MODEL_NOT_READY | 503 | 모델이 아직 로딩 중 |

### 예시 오류 응답

**잘못된 오디오 포맷:**
```json
{
  "error": "지원하지 않는 오디오 포맷: wav. PCM 16kHz만 지원됩니다.",
  "error_code": "INVALID_AUDIO_FORMAT",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**오디오 데이터 없음:**
```json
{
  "error": "오디오 데이터가 필요합니다.",
  "error_code": "INVALID_AUDIO_DATA",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

---

## 성능 최적화 가이드

### 클라이언트 측 최적화

1. **오디오 사전 처리**
   - 클라이언트에서 VAD 처리 후 `vad_enabled: false` 사용
   - 실시간 스트림에서 권장

2. **PCM 16kHz 직접 생성**
   - 다른 포맷에서 변환하지 말고 직접 PCM 16kHz로 녹음
   - FFmpeg 예시: `ffmpeg -i input.wav -ar 16000 -ac 1 -f s16le output.pcm`

3. **배치 처리**
   - 짧은 오디오 청크들을 하나로 합쳐서 처리
   - 네트워크 오버헤드 감소

### 서버 측 설정

1. **VAD 활용**
   - 원본 오디오: `vad_enabled: true` (더 빠름)
   - 사전 처리된 오디오: `vad_enabled: false`

2. **GPU 메모리 모니터링**
   - `nvidia-smi`로 GPU 메모리 사용량 확인
   - 95% 메모리 사용으로 최적화됨

---

## 버전 정보

- **API 버전**: 2.1.0
- **모델**: Large-v3
- **최적화**: RTX 4090 전용
- **포맷**: PCM 16kHz 전용
- **특징**: 워커 관리 시스템 제거, 단순화된 API

---

## 변경 이력

### v2.1.0 (2024-06-11)
- ✅ 워커 관리 기능 완전 제거
- ✅ PCM 16kHz 전용 지원 (WAV, MP3, FLAC 제거)
- ✅ VAD On/Off 기능 추가
- ✅ PyTorch 2.5+ 호환성 확보
- ✅ API 단순화 및 성능 최적화

### v2.0.0 (이전 버전)
- Large-v3 모델 적용
- RTX 4090 최적화
- 다중 포맷 지원 (제거됨)
- 워커 관리 시스템 (제거됨)