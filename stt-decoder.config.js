module.exports = {
  apps : [{
    name: "gpu-stt-server",
    script: "./start_pm2.sh",
    interpreter: "bash",
    instances: 1,                         // GPU 서버는 단일 인스턴스로 실행
    exec_mode: "fork",                    // GPU 메모리 공유 문제로 fork 모드 사용
    log_date_format: "YYYY-MM-DD HH:mm:ss Z",
    out_file: "./logs/pm2-out.log",
    error_file: "./logs/pm2-error.log",
    merge_logs: true,
    autorestart: true,
    watch: false,
    max_memory_restart: '8G',             // GPU 서버는 메모리 사용량이 많아 8GB로 설정
    min_uptime: "10s",                    // 최소 10초 실행 후 안정화로 간주
    max_restarts: 5,                      // 최대 재시작 횟수
    restart_delay: 4000,                  // 재시작 지연 시간 (ms)
    env: {
      NODE_ENV: "production",
      PORT: 8004,
      PYTHONUNBUFFERED: "1"              // Python 출력 버퍼링 비활성화
    },
    error_file: "./logs/gpu-stt-error.log",
    out_file: "./logs/gpu-stt-out.log",
    log_file: "./logs/gpu-stt-combined.log"
  }]
};
