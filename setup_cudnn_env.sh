#!/bin/bash
"""
cuDNN 환경 설정 스크립트
PyTorch가 올바른 cuDNN 라이브러리를 찾을 수 있도록 경로를 설정합니다.
"""

echo "🔧 cuDNN 환경 설정 중..."

# conda 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate stt-decoder

# cuDNN 라이브러리 경로 설정
export CUDNN_LIB_DIR="/home/jonghooy/miniconda3/envs/stt-decoder/lib/python3.9/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:$LD_LIBRARY_PATH"

# CUDA 라이브러리 경로도 추가
export CUDA_LIB_DIR="/home/jonghooy/miniconda3/envs/stt-decoder/lib/python3.9/site-packages/nvidia/cuda_runtime/lib"
if [ -d "$CUDA_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$CUDA_LIB_DIR:$LD_LIBRARY_PATH"
fi

# cuBLAS 라이브러리 경로도 추가
export CUBLAS_LIB_DIR="/home/jonghooy/miniconda3/envs/stt-decoder/lib/python3.9/site-packages/nvidia/cublas/lib"
if [ -d "$CUBLAS_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$CUBLAS_LIB_DIR:$LD_LIBRARY_PATH"
fi

echo "✅ cuDNN 라이브러리 경로 설정 완료"
echo "CUDNN_LIB_DIR: $CUDNN_LIB_DIR"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 라이브러리 파일 존재 확인
echo ""
echo "🔍 cuDNN 라이브러리 파일 확인:"
ls -la "$CUDNN_LIB_DIR" | grep "libcudnn_ops"

echo ""
echo "✅ cuDNN 환경 설정이 완료되었습니다!"
echo "이제 다음 명령으로 서버를 실행하세요:"
echo "python simple_stt_server.py" 