module.exports = {
  apps : [{
    name: "stt-server",
    script: "./start_pm2.sh",
    interpreter: "bash", // 이 스크립트를 실행할 인터프리터
    log_date_format: "YYYY-MM-DD HH:mm Z", // 로그 시간 형식
    out_file: "./logs/pm2-out.log",       // 표준 출력 로그 파일 경로
    error_file: "./logs/pm2-error.log",   // 에러 로그 파일 경로
    merge_logs: true,                     // 모든 로그를 하나의 파일로 병합
    autorestart: true,                    // 앱이 비정상적으로 종료되면 자동 재시작
    watch: false,                         // 파일 변경 감지 비활성화 (프로덕션에서는 false 권장)
    max_memory_restart: '2G'              // 메모리 사용량이 2GB를 초과하면 재시작
  }]
};
