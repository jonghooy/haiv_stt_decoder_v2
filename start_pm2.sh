#!/bin/bash
# PM2로 GPU 최적화된 STT 서버를 실행하기 위한 스크립트

echo "🚀 PM2 STT 서버 시작 중..."

# 1. 기존 서버 프로세스 정리
echo "🧹 기존 서버 프로세스 정리 중..."
pkill -f "gpu_optimized_stt_server.py" || true
sleep 2

# 2. Conda 환경 활성화
echo "🔧 Conda 환경 활성화 중..."
source /home/jonghooy/miniconda3/etc/profile.d/conda.sh
conda activate stt-decoder

# 3. cuDNN 환경 설정
echo "🔧 cuDNN 환경 설정 중..."
source ./setup_cudnn_env.sh

# 4. 로그 디렉토리 생성
mkdir -p logs

# 5. GPU 최적화된 STT 서버 실행
echo "🚀 GPU 최적화된 STT 서버 시작..."
exec python gpu_optimized_stt_server.py --port 8001
