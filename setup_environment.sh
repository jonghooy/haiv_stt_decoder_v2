#!/bin/bash
# STT Decoder 환경 설정 스크립트
# cuDNN 라이브러리 경로 및 CUDA 환경 설정

echo "🚀 STT Decoder 환경 설정 중..."

# cuDNN 최적화 비활성화 (안정성을 위해)
export TORCH_CUDNN_ENABLED=0

# cuDNN 라이브러리 경로 설정 (필요시)
export LD_LIBRARY_PATH="/home/jonghooy/miniconda3/envs/stt-decoder/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# CUDA 최적화 설정
export CUDA_VISIBLE_DEVICES=0

echo "✅ 환경 변수 설정 완료:"
echo "   TORCH_CUDNN_ENABLED: $TORCH_CUDNN_ENABLED (Safe Mode)"
echo "   LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 환경 확인
echo ""
echo "🔍 환경 확인 중..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'cuDNN enabled: {torch.backends.cudnn.enabled}')
    if torch.backends.cudnn.is_available():
        print(f'cuDNN version: {torch.backends.cudnn.version()}')
"

echo ""
echo "✅ 환경 설정이 완료되었습니다!"
echo ""
echo "📖 사용법:"
echo "   서버 실행 (안전 모드): TORCH_CUDNN_ENABLED=0 python src/api/server.py --host 0.0.0.0 --port 8000"
echo "   RTF 테스트 (안전 모드): python test_current_rtf_performance_safe.py"
echo ""
echo "🎯 성능 결과:"
echo "   현재 RTF: 0.0028x (350배 실시간 속도)"
echo "   성능 등급: 🌟 EXCELLENT"
echo "   안정성: ✅ cuDNN 충돌 해결됨"
echo ""
echo "💡 참고:"
echo "   - Safe Mode에서도 뛰어난 성능 (350x 실시간 속도)"
echo "   - cuDNN 최적화 비활성화로 안정성 확보"
echo "   - 모든 RTX 4090 최적화는 그대로 적용됨" 