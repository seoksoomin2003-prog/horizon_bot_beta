#!/bin/bash
# 보기 전용 화면. 봇과 별개로 켜고 끌 수 있으며, 꺼도 봇에는 영향이 없습니다.
#   ./run_dashboard.sh         화면 서버를 켜고 브라우저로 엽니다
#   ./run_dashboard.sh stop    화면 서버만 끕니다
set -u
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
RUN="$ROOT/.run"
PID_FILE="$RUN/dashboard.pid"
URL="http://127.0.0.1:8791"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
mkdir -p "$RUN"

is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "${1:-start}" in
    start)
        if ! is_running; then
            nohup "$PY" -I "$ROOT/dashboard/server.py" >"$RUN/dashboard.out" 2>&1 &
            echo $! >"$PID_FILE"
            sleep 1
            if ! is_running; then
                echo "화면 서버를 시작하지 못했습니다:"
                cat "$RUN/dashboard.out"
                rm -f "$PID_FILE"
                exit 1
            fi
        fi
        echo "화면: $URL (이 맥북에서만 열립니다)"
        open "$URL"
        ;;
    stop)
        if is_running; then
            kill "$(cat "$PID_FILE")"
            echo "화면 서버를 껐습니다. 봇은 그대로 돌아갑니다."
        else
            echo "화면 서버가 꺼져 있습니다."
        fi
        rm -f "$PID_FILE"
        ;;
    *)
        echo "사용법: ./run_dashboard.sh [start|stop]"
        exit 1
        ;;
esac
