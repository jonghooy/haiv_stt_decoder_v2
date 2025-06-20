#!/bin/bash
# 이 스크립트는 PM2로 Gunicorn 기반 STT 서버를 실행하기 위한 것입니다.
# 정확한 Conda 환경과 cuDNN 경로 설정을 보장합니다.

# 1. Conda 환경 활성화
# 참고: Conda 설치 경로가 다를 경우 아래 경로를 수정해야 합니다.
source /home/jonghooy/miniconda3/etc/profile.d/conda.sh
conda activate stt-decoder

# 2. cuDNN 환경 설정 스크립트 실행
source ./setup_cudnn_env.sh

# 3. Gunicorn으로 애플리케이션 실행
#    --preload: 워커를 생성하기 전에 앱을 미리 로드하여 모델을 한 번만 로드합니다.
#    -w 1: GPU를 많이 사용하는 작업이므로, 워커 수를 1로 시작하는 것이 안전합니다.
#    -k uvicorn.workers.UvicornWorker: ASGI 앱을 위한 워커 클래스
#    -b 0.0.0.0:8003: 8003번 포트로 바인딩
#    large_only_optimized_server:app: 실행할 Python 모듈과 FastAPI 앱 인스턴스
echo "Starting Gunicorn server with --preload and 1 worker..."
exec gunicorn --preload -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8003 large_only_optimized_server:app
