#!/usr/bin/env bash
# Start every orb service on this machine for local work. Ctrl+C stops all of them.
#
#   scripts/dev.sh            # all services, then open http://127.0.0.1:8100
#   scripts/dev.sh --no-voice # only the agent service and Ollama (text chat)
#
# Logs go to data/logs/<service>.log. Every service listens on 127.0.0.1 only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ORB_PYTHON:-$HOME/miniconda3/envs/orb/bin/python}"
export JEV_ENV_FILE="${JEV_ENV_FILE:-$HOME/jev-test/.env}"
export NEEDLE_TELEMETRY=0
export ONNXRUNTIME_LIB_PATH="${ONNXRUNTIME_LIB_PATH:-/opt/homebrew/lib/libonnxruntime.dylib}"
export CGO_ENABLED=1
VOICE=1
[[ "${1:-}" == "--no-voice" ]] && VOICE=0

LOGS="$ROOT/data/logs"
mkdir -p "$LOGS" "$ROOT/data/bin"
pids=()

cleanup() {
  trap - EXIT INT TERM
  echo
  echo "Stopping the orb services."
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for _ in $(seq 1 20); do  # up to 10 s, then force
    alive=0
    for pid in "${pids[@]:-}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    (( alive )) || break
    sleep 0.5
  done
  for pid in "${pids[@]:-}"; do kill -KILL "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

port_free() { ! lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

start() { # name port command...
  local name=$1 port=$2; shift 2
  if ! port_free "$port"; then
    echo "ERROR: port $port is in use, so $name cannot start. Stop the process on that port, then try again." >&2
    exit 1
  fi
  "$@" >"$LOGS/$name.log" 2>&1 &
  pids+=($!)
  echo "started $name on port $port (pid $!), log: data/logs/$name.log"
}

wait_for() { # name url seconds
  local name=$1 url=$2 limit=$3 i=0
  until curl -s -m 2 -o /dev/null "$url"; do
    i=$((i + 1))
    if (( i > limit * 2 )); then
      echo "ERROR: $name did not answer at $url in $limit s. Read data/logs/$name.log." >&2
      exit 1
    fi
    sleep 0.5
  done
  echo "ready   $name"
}

if curl -s -m 2 -o /dev/null http://127.0.0.1:11434/api/tags; then
  echo "found   ollama (already running)"
else
  start ollama 11434 ollama serve
  wait_for ollama http://127.0.0.1:11434/api/tags 30
fi

start agent 8100 "$PY" -m orb_agent.server
if (( VOICE )); then
  start livekit 7880 livekit-server --dev --bind 127.0.0.1 --node-ip 127.0.0.1
  # The models are on disk after `make models`, so the speech service needs no network.
  start speech 8200 env HF_HUB_OFFLINE=1 "$PY" -m orb_speech.server
  echo "build   apps/voice"
  (cd "$ROOT/apps/voice" && go build -ldflags=-extldflags=-Wl,-no_warn_duplicate_libraries -o "$ROOT/data/bin/orb-voice" .)
  wait_for livekit http://127.0.0.1:7880 20
  wait_for speech http://127.0.0.1:8200/health 60
  start voice 8081 "$ROOT/data/bin/orb-voice" dev --drain-timeout 5s
fi
wait_for agent http://127.0.0.1:8100/health 60

echo
echo "orb is running. Open http://127.0.0.1:8100 . Press Ctrl+C to stop."
wait
