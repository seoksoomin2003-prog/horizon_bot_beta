#!/bin/bash
# 방향 예측 기록기. 주문을 내지 않고 API 키도 읽지 않습니다. 공개 시세만 씁니다.
# 4시간봉이 마감될 때마다 코인 60개의 방향 예측과 큰 코인 30개의 "얼마나 오를까"를 기록하고, 기간이 끝나면 채점합니다.
#   ./run_predictor.sh start    켜기
#   ./run_predictor.sh stop     끄기
#   ./run_predictor.sh status   실행 여부와 최근 메시지
set -u
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
RUN="$ROOT/.run"
PID_FILE="$RUN/predictor.pid"
PY="$ROOT/.venv/bin/python"
mkdir -p "$RUN"

is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "${1:-}" in
    start)
        if is_running; then
            echo "이미 실행 중입니다 (PID $(cat "$PID_FILE"))."
            exit 0
        fi
        if [ ! -x "$PY" ]; then
            echo "먼저 ./run_mac.sh setup 을 한 번 실행하세요."
            exit 1
        fi
        # -I: 환경 변수와 현재 폴더의 모듈을 무시합니다. 예측기는 config(.env)를 불러오지 않습니다.
        nohup "$PY" -I "$ROOT/predictor/live.py" >>"$RUN/predictor.out" 2>&1 </dev/null &
        echo $! >"$PID_FILE"
        # 예측기가 도는 동안 잠자기를 막고, 예측기가 끝나면 함께 끝납니다. 덮개를 닫는 잠자기까지 항상 막지는 못합니다.
        nohup caffeinate -is -w "$(cat "$PID_FILE")" >/dev/null 2>&1 </dev/null &
        sleep 2
        if is_running; then
            echo "예측 기록기를 켰습니다. 처음에는 과거 시세를 받느라 1~2분 걸립니다."
        else
            echo "켜지 못했습니다:"
            tail -n 20 "$RUN/predictor.out"
            rm -f "$PID_FILE"
            exit 1
        fi
        ;;
    stop)
        if is_running; then
            kill "$(cat "$PID_FILE")"
            echo "예측 기록기를 껐습니다. 지금까지의 기록은 runtime/predictor/에 남습니다."
        else
            echo "예측 기록기가 꺼져 있습니다."
        fi
        rm -f "$PID_FILE"
        ;;
    status)
        if is_running; then echo "상태: 실행 중 (PID $(cat "$PID_FILE"))"; else echo "상태: 꺼짐"; fi
        tail -n 5 "$RUN/predictor.out" 2>/dev/null
        ;;
    *)
        sed -n 2,6p "$0"
        exit 1
        ;;
esac
