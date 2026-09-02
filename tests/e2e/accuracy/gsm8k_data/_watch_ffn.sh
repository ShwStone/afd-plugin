#!/bin/bash
# Watch e2e FFN-tagged processes (AFD_E2E_RUN_ID + AFD_E2E_PROCESS_ROLE=ffn)
# once per second: pid, comm, /proc state, and whether SIGKILL is pending
# (SigPnd bit 9 => mask 0x100). Prints NEW/GONE events with timestamps so the
# delay between SIGKILL delivery and actual reap is directly measurable.
# Usage: bash _watch_ffn.sh        (Ctrl-C to stop; also tees to /tmp/ffn_watch.log)

declare -A first_seen
declare -A last_state

fmt_ts() { date '+%H:%M:%S'; }

check_kill_pending() {
    # $1=pid; echo "KILL_PENDING" if signal 9 pending, else "-"
    local pnd
    pnd=$(awk '/^SigPnd:/{print $2}' /proc/$1/status 2>/dev/null)
    if [ -n "$pnd" ] && [ $((16#$pnd & 16#100)) -ne 0 ]; then
        echo "KILL_PENDING"
    else
        echo "-"
    fi
}

{
while true; do
    now_s=$(date +%s)
    ts=$(fmt_ts)
    seen_now=""
    for d in /proc/[0-9]*; do
        pid=${d#/proc/}
        [ "$pid" = "$$" ] && continue
        # cheap pre-filter: environ must contain both markers
        if tr '\0' '\n' < "$d/environ" 2>/dev/null | grep -q '^AFD_E2E_PROCESS_ROLE=ffn$'; then
            run_id=$(tr '\0' '\n' < "$d/environ" 2>/dev/null | grep '^AFD_E2E_RUN_ID=' | cut -d= -f2)
            state=$(awk '{print $3}' "$d/stat" 2>/dev/null)
            comm=$(cat "$d/comm" 2>/dev/null)
            kp=$(check_kill_pending "$pid")
            if [ -z "${first_seen[$pid]}" ]; then
                first_seen[$pid]=$now_s
                echo "$ts NEW    pid=$pid comm=$comm state=$state kill=$kp run_id=$run_id"
            elif [ "${last_state[$pid]}" != "$state$kp" ]; then
                echo "$ts CHANGE pid=$pid comm=$comm state=$state kill=$kp"
            fi
            last_state[$pid]="$state$kp"
            seen_now="$seen_now $pid"
        fi
    done
    for pid in "${!first_seen[@]}"; do
        case " $seen_now " in
            *" $pid "*) ;;
            *)
                echo "$ts GONE   pid=$pid alive_for=$((now_s - first_seen[$pid]))s"
                unset first_seen[$pid]
                unset last_state[$pid]
                ;;
        esac
    done
    sleep 1
done
} 2>&1 | tee -a /tmp/ffn_watch.log
