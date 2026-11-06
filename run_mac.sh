#!/bin/bash
# 맥북 실행 스크립트. 서버와 systemd 없이 동작합니다. 기본은 가상 매매이며 API 키를 읽지 않습니다.
#   ./run_mac.sh setup    처음 한 번: 패키지 설치
#   ./run_mac.sh start    가상 매매 시작
#   ./run_mac.sh status   실행 여부와 가상 계좌 현황
#   ./run_mac.sh logs     최근 로그
#   ./run_mac.sh stop     중지
set -u
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
RUN="$ROOT/.run"
PY="$ROOT/.venv/bin/python"
PIDFILE="$RUN/bot.pid"
PHRASE="I_ACCEPT_REAL_LOSSES"
mkdir -p "$RUN"

running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

icloud_warning() {
    case "$ROOT" in
        "$HOME/Desktop"*|"$HOME/Documents"*)
            echo "주의: 이 폴더는 iCloud 동기화 대상일 수 있습니다. 파일이 클라우드로 내려가면 실행이 멈추거나 .env가 업로드될 수 있습니다."
            echo "      장시간 실행하려면 ~/Projects 같은 동기화 밖 폴더로 옮기는 것을 권합니다."
            ;;
    esac
}

# Restart=always 대체: 비정상 종료 시 다시 시작하고, 반복 실패하면 간격을 늘립니다.
supervise() {
    local delay=10 child=""
    trap '[ -n "$child" ] && kill "$child" 2>/dev/null; rm -f "$PIDFILE"; exit 0' TERM INT
    echo $$ > "$PIDFILE"
    while true; do
        local started
        started=$(date +%s)
        "$PY" main.py &
        child=$!
        wait "$child"
        local code=$?
        child=""
        if [ $(( $(date +%s) - started )) -gt 300 ]; then delay=10; fi
        echo "$(date '+%F %T') 프로그램 종료(코드 $code). ${delay}초 뒤 다시 시작합니다."
        sleep "$delay" & child=$!; wait "$child"; child=""
        delay=$(( delay * 2 )); [ "$delay" -gt 600 ] && delay=600
    done
}

launch() {
    if running; then
        echo "이미 실행 중입니다 (PID $(cat "$PIDFILE"))."
        exit 1
    fi
    if [ ! -x "$PY" ]; then
        echo "먼저 ./run_mac.sh setup 을 한 번 실행하세요."
        exit 1
    fi
    icloud_warning
    # -i 시스템 잠자기 방지, -s 전원 연결 시 잠자기 방지. 덮개를 닫는 잠자기까지 항상 막지는 못합니다.
    nohup caffeinate -is /bin/bash "$ROOT/run_mac.sh" _supervise >> "$RUN/bot.out" 2>&1 < /dev/null &
    sleep 2
    if running; then
        echo "시작했습니다 (PID $(cat "$PIDFILE"))."
    else
        echo "시작하지 못했습니다. $RUN/bot.out 을 확인하세요."
        exit 1
    fi
}

case "${1:-}" in
    setup)
        python3 -m venv .venv && "$PY" -m pip install -q --upgrade pip && "$PY" -m pip install -q -r requirements.txt pytest && echo "설치 완료"
        ;;
    start)
        if [ "${2:-}" = "--live" ]; then
            echo "실거래 모드입니다. 실제 돈으로 주문이 나가고 원금 이상의 손실이 날 수 있습니다."
            echo "docs/MAC_GUIDE_KO.md 의 실거래 전 점검을 마쳤다면 다음 문구를 그대로 입력하세요: $PHRASE"
            read -r typed < /dev/tty
            if [ "$typed" != "$PHRASE" ]; then
                echo "문구가 달라 취소했습니다."
                exit 1
            fi
            export TRADING_MODE=live LIVE_TRADING_CONFIRM="$PHRASE"
        elif [ -n "${2:-}" ]; then
            echo "알 수 없는 옵션: $2"
            exit 1
        else
            # .env에 TRADING_MODE=live가 있어도 여기서는 가상 매매입니다.
            export TRADING_MODE=paper
            echo "가상 매매 모드: 실제 주문 없음, API 키 사용 안 함, 공개 시세만 사용"
        fi
        launch
        ;;
    _supervise)
        supervise
        ;;
    stop)
        if running; then
            pid=$(cat "$PIDFILE")
            kill "$pid"
            for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
            echo "중지했습니다. 가상 포지션과 손절·익절 주문은 그대로 남아 다음 실행 때 이어집니다."
        else
            rm -f "$PIDFILE"
            echo "실행 중이 아닙니다."
        fi
        ;;
    status)
        if running; then echo "상태: 실행 중 (PID $(cat "$PIDFILE"))"; else echo "상태: 중지됨"; fi
        TRADING_MODE=paper "$PY" scripts/paper_report.py
        ;;
    logs)
        tail -n 40 logs/paper/trades.log; echo "--- 오류 ---"; tail -n 15 logs/paper/errors.log
        ;;
    *)
        sed -n 2,7p "$0"
        exit 1
        ;;
esac
