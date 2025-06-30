#!/bin/bash
# PM2로 GPU 최적화된 STT 서버를 관리하는 편의 스크립트

set -e

function print_usage() {
    echo "🔧 GPU STT 서버 PM2 관리 스크립트"
    echo ""
    echo "사용법: $0 {start|stop|restart|status|logs|monitor|delete}"
    echo ""
    echo "명령어:"
    echo "  start    - STT 서버 시작"
    echo "  stop     - STT 서버 중지"
    echo "  restart  - STT 서버 재시작"
    echo "  status   - 서버 상태 확인"
    echo "  logs     - 실시간 로그 보기"
    echo "  monitor  - PM2 모니터링 대시보드"
    echo "  delete   - PM2에서 앱 완전히 제거"
    echo ""
}

function check_pm2() {
    if ! command -v pm2 &> /dev/null; then
        echo "❌ PM2가 설치되지 않았습니다."
        echo "설치: npm install -g pm2"
        exit 1
    fi
}

function start_server() {
    echo "🚀 GPU STT 서버 시작 중..."
    
    # 기존 서버 프로세스 정리
    echo "🧹 기존 프로세스 정리..."
    pkill -f "gpu_optimized_stt_server.py" || true
    sleep 2
    
    # PM2로 시작
    pm2 start stt-decoder.config.js
    
    echo "✅ 서버가 시작되었습니다!"
    echo "📊 상태 확인: ./pm2_control.sh status"
    echo "📝 로그 확인: ./pm2_control.sh logs"
}

function stop_server() {
    echo "🛑 GPU STT 서버 중지 중..."
    pm2 stop gpu-stt-server
    echo "✅ 서버가 중지되었습니다!"
}

function restart_server() {
    echo "🔄 GPU STT 서버 재시작 중..."
    pm2 restart gpu-stt-server
    echo "✅ 서버가 재시작되었습니다!"
}

function show_status() {
    echo "📊 GPU STT 서버 상태:"
    pm2 list
    echo ""
    echo "📈 자세한 정보:"
    pm2 show gpu-stt-server
}

function show_logs() {
    echo "📝 GPU STT 서버 실시간 로그:"
    echo "종료하려면 Ctrl+C를 누르세요"
    pm2 logs gpu-stt-server --lines 50
}

function monitor() {
    echo "📊 PM2 모니터링 대시보드 시작..."
    echo "종료하려면 Ctrl+C를 누르세요"
    pm2 monit
}

function delete_server() {
    echo "🗑️  PM2에서 GPU STT 서버 완전히 제거 중..."
    pm2 stop gpu-stt-server 2>/dev/null || true
    pm2 delete gpu-stt-server 2>/dev/null || true
    echo "✅ 서버가 PM2에서 제거되었습니다!"
}

# 메인 로직
check_pm2

case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    monitor)
        monitor
        ;;
    delete)
        delete_server
        ;;
    *)
        print_usage
        exit 1
        ;;
esac 