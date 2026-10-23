#!/usr/bin/env bash
# Start every orb service on this machine for local work. Ctrl+C stops all of them.
#
#   scripts/dev.sh              # all services, then open http://127.0.0.1:8100
#   scripts/dev.sh --no-voice   # only the agent service and Ollama (text chat)
#   scripts/dev.sh --no-tunnel  # no ngrok, also when ORB_NGROK_DOMAIN is set
#
# Logs go to data/logs/<service>.log. Every service listens on 127.0.0.1 only.
#
# Voice has three modes (see docs/decisions/0009-sharing.md). The settings come from the environment, else from .env:
#   dev        LIVEKIT_URL is local or unset, and there are no keys: `livekit-server --dev` with the public dev keys.
#   keys       LIVEKIT_URL is local, with LIVEKIT_API_KEY and LIVEKIT_API_SECRET: a LiveKit server with those keys
#              (scripts/livekit-keys.yaml). Use it when other devices connect.
#   elsewhere  LIVEKIT_URL is not local (LiveKit Cloud, a public host), with its keys: no server is started here.
# Sharing: with ORB_NGROK_DOMAIN set (a domain of your ngrok account), this script runs `ngrok http` for the agent service.
# The page and the voice signal (/rtc) both go through that one address. The audio does not: a device needs a network path
# to this computer for it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ORB_PYTHON:-$HOME/miniconda3/envs/orb/bin/python}"
export JEV_ENV_FILE="${JEV_ENV_FILE:-$HOME/jev-test/.env}"
export NEEDLE_TELEMETRY=0
VOICE=1
TUNNEL=1
for arg in "$@"; do
  case "$arg" in
    --no-voice) VOICE=0 ;;
    --no-tunnel) TUNNEL=0 ;;
    *) echo "Unknown option: $arg. Use --no-voice or --no-tunnel." >&2; exit 2 ;;
  esac
done

LOGS="$ROOT/data/logs"
mkdir -p "$LOGS"
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

# A setting from the environment, else from the repo .env (its first line "NAME=value", without quotes or a comment).
setting() {
  local v="${!1:-}"
  if [[ -z "$v" && -f "$ROOT/.env" ]]; then
    v="$(sed -n "s/^$1=\([^[:space:]#]*\).*/\1/p" "$ROOT/.env" | head -1 | tr -d "\"'")"
  fi
  printf '%s' "$v"
}

# A process with no port to check, for example a tunnel.
start_bg() { # name command...
  local name=$1; shift
  "$@" >"$LOGS/$name.log" 2>&1 &
  pids+=($!)
  echo "started $name (pid $!), log: data/logs/$name.log"
}

# Wait until a tunnel reaches the agent service: the service answers 200, or 403 when no key is given. A tunnel with nothing
# behind it, or one that is not up yet, gives something else.
wait_tunnel() { # url seconds
  local url=$1 limit=$2 i=0 code
  until code="$(curl -s -m 3 -o /dev/null -w '%{http_code}' -H 'ngrok-skip-browser-warning: 1' "$url/v1/voice/status")"; [[ "$code" == 200 || "$code" == 403 ]]; do
    i=$((i + 1))
    if (( i > limit )); then
      echo "WARNING: $url did not reach the agent service in $limit s (last answer: $code). Read data/logs/ngrok.log." >&2
      return 0
    fi
    sleep 1
  done
  echo "ready   $url"
}

is_local() { [[ "$1" =~ ^wss?://(127\.0\.0\.1|localhost|\[::1\])([:/]|$) ]]; }

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
  LK_URL="$(setting LIVEKIT_URL)"; LK_URL="${LK_URL:-ws://127.0.0.1:7880}"
  LK_KEY="$(setting LIVEKIT_API_KEY)"; LK_SECRET="$(setting LIVEKIT_API_SECRET)"
  if ! is_local "$LK_URL"; then
    LK_MODE=elsewhere
  elif [[ -n "$LK_KEY" && -n "$LK_SECRET" ]]; then
    LK_MODE=keys
  else
    LK_MODE=dev
  fi
  case "$LK_MODE" in
    dev)
      if [[ -n "$(setting LIVEKIT_PUBLIC_URL)" ]]; then
        echo "WARNING: LIVEKIT_PUBLIC_URL is set, but there are no LIVEKIT_API_KEY and LIVEKIT_API_SECRET: voice stays off for it." >&2
      fi
      start livekit 7880 livekit-server --dev --bind 127.0.0.1 --node-ip 127.0.0.1 ;;
    keys)
      # the keys go in the environment, not on the command line, so `ps` does not show them
      start livekit 7880 env LIVEKIT_KEYS="$LK_KEY: $LK_SECRET" livekit-server --config "$ROOT/scripts/livekit-keys.yaml" ;;
    elsewhere)
      if ! curl -s -m 5 -o /dev/null "${LK_URL/#ws/http}"; then
        echo "WARNING: the LiveKit server at $LK_URL does not answer. Voice will not work until it does." >&2
      else
        echo "using   the LiveKit server at $LK_URL (it is not started here)"
      fi ;;
  esac
  # The models are on disk after `make models`, so the speech service needs no network.
  start speech 8200 env HF_HUB_OFFLINE=1 "$PY" -m orb_speech.server
  if [[ "$LK_MODE" != elsewhere ]]; then wait_for livekit http://127.0.0.1:7880 20; fi
  wait_for speech http://127.0.0.1:8200/health 60
  # The voice worker (LiveKit Agents) joins every new room. In the dev mode it uses the dev server's public
  # credentials. Otherwise it reads LIVEKIT_API_KEY and LIVEKIT_API_SECRET from the environment or .env, and it
  # connects out from this computer (it needs no open port).
  if [[ "$LK_MODE" == dev ]]; then
    start voice 8081 env LIVEKIT_URL="$LK_URL" LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}" \
      LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}" ORB_VOICE_PORT=8081 \
      "$PY" -m orb_voice dev --no-reload --log-level info
  else
    start voice 8081 env LIVEKIT_URL="$LK_URL" ORB_VOICE_PORT=8081 "$PY" -m orb_voice dev --no-reload --log-level info
  fi
  wait_for voice http://127.0.0.1:8081/ 60
fi
wait_for agent http://127.0.0.1:8100/health 60

NGROK_DOMAIN=""
if (( TUNNEL )); then NGROK_DOMAIN="$(setting ORB_NGROK_DOMAIN)"; fi
if [[ -n "$NGROK_DOMAIN" ]]; then
  if command -v ngrok >/dev/null 2>&1; then
    start_bg ngrok ngrok http 127.0.0.1:8100 --url "https://$NGROK_DOMAIN" --log stdout
    wait_tunnel "https://$NGROK_DOMAIN" 30
  else
    echo "WARNING: ORB_NGROK_DOMAIN is set, but ngrok is not installed." >&2
  fi
fi

echo
echo "orb is running. Open http://127.0.0.1:8100 . Press Ctrl+C to stop."
if [[ -n "$NGROK_DOMAIN" ]]; then
  echo "Shared at https://$NGROK_DOMAIN : open it once as /?key=<ORB_SHARE_KEY> on each device (a browser first shows ngrok's notice)."
  echo "Audio needs a network path from the device to this computer (the same network): decision 0009."
fi
if [[ -n "$(setting ORB_SHARE_KEY)" ]]; then
  echo "Sharing is on: a request from a tunnel or another device needs the key (ORB_SHARE_KEY)."
else
  echo "Sharing is off: a request from a tunnel or another device is refused. To share, set ORB_SHARE_KEY in .env."
fi
wait
