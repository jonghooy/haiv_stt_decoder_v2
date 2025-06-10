#!/usr/bin/env python3
"""
GPU STT Server Launcher
RTX 4090 최적화된 STT 서버 실행 스크립트
"""

import os
import sys
import uvicorn
import logging

# 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDNN_ENABLED'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """메인 함수"""
    print("🚀 GPU 최적화 STT 서버 시작...")
    print("📦 CUDA 설정 적용 중...")
    print("⚡ RTX 4090 최적화 활성화...")
    
    try:
        # GPU 서버 실행
        uvicorn.run(
            "gpu_optimized_stt_server:app",
            host="0.0.0.0",
            port=8001,
            reload=False,  # GPU 최적화를 위해 reload 비활성화
            log_level="info",
            access_log=True,
            use_colors=True
        )
    except KeyboardInterrupt:
        print("\\n⏹️  서버 정지됨")
    except Exception as e:
        logger.error(f"❌ 서버 실행 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 