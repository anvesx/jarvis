#!/usr/bin/env bash
# Start the whole orb stack, test it, and stop it again. The exit code is 0 when every check passed.
#
#   scripts/check_stack.sh              # unit tests, the stack, the live checks, the voice end-to-end test, then stop
#   scripts/check_stack.sh --quick      # only the stack and the live checks (about 1 minute)
#   scripts/check_stack.sh --no-unit    # skip the unit tests
#   scripts/check_stack.sh --no-e2e     # skip the voice end-to-end test (about 2 minutes)
#   scripts/check_stack.sh --no-tunnel  # no ngrok, also when ORB_NGROK_DOMAIN is set
#
# The live checks use the Jev API (about 10 reads, under $0.001) and the local LLM. No audio leaves this computer.
# It does not touch a stack that already runs: when a port is busy, it stops at once and says which one.
# It stops everything that it started, also after a failed check and after Ctrl+C.
# Logs: data/logs/check/<time>/   Exit codes: 0 every check passed, 1 a check failed, 2 it could not start (nothing changed).
set -u  # not -e: a failed check must not end the script before it has stopped the stack

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ORB_PYTHON:-$HOME/miniconda3/envs/orb/bin/python}"
export JEV_ENV_FILE="${JEV_ENV_FILE:-$HOME/jev-test/.env}"
export NEEDLE_TELEMETRY=0
AGENT=http://127.0.0.1:8100
SPEECH=http://127.0.0.1:8200
PORTS="8100 8200 8081 7880"

UNIT=1; E2E=1; TUNNEL=1
for arg in "$@"; do
  case "$arg" in
    --quick) UNIT=0; E2E=0 ;;
    --no-unit) UNIT=0 ;;
    --no-e2e) E2E=0 ;;
    --no-tunnel) TUNNEL=0 ;;
    -h|--help) awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg. Try --help." >&2; exit 2 ;;
  esac
done

STAMP="$(date +%Y%m%d-%H%M%S)"
LOGS="$ROOT/data/logs/check/$STAMP"  # made after the preflight: a refused run leaves nothing behind
START=$SECONDS

# A setting from the environment, else from the repo .env (its first line "NAME=value", without quotes or a comment).
setting() {
  local v="${!1:-}"
  if [[ -z "$v" && -f "$ROOT/.env" ]]; then
    v="$(sed -n "s/^$1=\([^[:space:]#]*\).*/\1/p" "$ROOT/.env" | head -1 | tr -d "\"'")"
  fi
  printf '%s' "$v"
}

# --- the record of the checks -------------------------------------------------------------------------------------
RESULTS=()  # "PASS|seconds|name" or "FAIL|seconds|name"
FAILED=0
N=0

run_check() { # name command...: run the command, record PASS or FAIL, show what it said; the status is the command's
  local name=$1; shift
  N=$((N + 1))
  local out="$LOGS/check-$(printf '%02d' "$N").log" t0=$SECONDS
  "$@" >"$out" 2>&1
  local rc=$?
  local secs=$((SECONDS - t0))
  if (( rc == 0 )); then
    RESULTS+=("PASS|$secs|$name")
    printf 'PASS  %3d s  %s\n' "$secs" "$name"
    sed 's/^/                /' "$out" | head -n 12
  else
    FAILED=$((FAILED + 1))
    RESULTS+=("FAIL|$secs|$name")
    printf 'FAIL  %3d s  %s\n' "$secs" "$name"
    sed 's/^/                /' "$out" | head -n 20
    echo "                (full output: ${out#$ROOT/})"
  fi
  return $rc
}

# --- the stack ----------------------------------------------------------------------------------------------------
DEV_PID=""
STOPPED=0

stop_stack() {
  (( STOPPED )) && return 0
  STOPPED=1
  [[ -n "$DEV_PID" ]] || return 0
  echo
  echo "Stopping the stack (pid $DEV_PID)."
  kill -TERM "$DEV_PID" 2>/dev/null
  local i
  for i in $(seq 1 90); do
    kill -0 "$DEV_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$DEV_PID" 2>/dev/null; then
    echo "The stack did not stop in 90 s. Stopping it by force." >&2
    pkill -KILL -P "$DEV_PID" 2>/dev/null
    kill -KILL "$DEV_PID" 2>/dev/null
  fi
  wait "$DEV_PID" 2>/dev/null
  return 0
}
trap 'stop_stack; exit 130' INT TERM
trap 'stop_stack' EXIT

preflight() {
  local missing="" tool p busy=""
  [[ -x "$PY" ]] || missing="$missing the-orb-python($PY)"
  for tool in curl lsof livekit-server ollama; do
    command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
  done
  if (( TUNNEL )) && [[ -n "$(setting ORB_NGROK_DOMAIN)" ]]; then
    command -v ngrok >/dev/null 2>&1 || missing="$missing ngrok(--no-tunnel skips it)"
  fi
  if [[ -n "$missing" ]]; then
    echo "Missing:$missing"
    return 1
  fi
  for p in $PORTS; do
    lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1 && busy="$busy $p"
  done
  if [[ -n "$busy" ]]; then
    echo "These ports are in use:$busy. A stack may run already. This script does not touch a stack that it did not start."
    for p in $busy; do
      lsof -nP -iTCP:"$p" -sTCP:LISTEN | awk -v port="$p" 'NR == 2 {print "  port " port ": " $1 " (pid " $2 ")"}'
    done
    return 1
  fi
  return 0
}

start_stack() {
  local args=""
  (( TUNNEL )) || args="--no-tunnel"
  "$ROOT/scripts/dev.sh" $args >"$LOGS/stack.log" 2>&1 &
  DEV_PID=$!
}

wait_ready() {
  local i
  for i in $(seq 1 300); do
    if grep -q '^orb is running' "$LOGS/stack.log"; then
      echo "every service answered after about $i s (log: ${LOGS#$ROOT/}/stack.log)"
      grep -E '^(WARNING|ERROR)' "$LOGS/stack.log" | head -n 5
      return 0
    fi
    if ! kill -0 "$DEV_PID" 2>/dev/null; then
      echo "The stack stopped while it started. Its log:"
      tail -n 15 "$LOGS/stack.log"
      return 1
    fi
    sleep 1
  done
  echo "The stack was not ready in 300 s. Its log:"
  tail -n 15 "$LOGS/stack.log"
  return 1
}

# --- the checks ---------------------------------------------------------------------------------------------------
unit_tests() {
  if ! make -C "$ROOT" test >"$LOGS/unit.log" 2>&1; then
    tail -n 25 "$LOGS/unit.log"
    return 1
  fi
  grep -oE '[0-9]+ passed' "$LOGS/unit.log" | awk '{n += $1} END {print n " tests passed"}'
}

c_ollama() {
  curl -sf -m 5 http://127.0.0.1:11434/api/tags | ORB_LLM_MODEL="$(setting ORB_LLM_MODEL)" \
    ORB_POLISH_MODEL="$(setting ORB_POLISH_MODEL)" "$PY" -I -c '
import json, os, sys
names = [m["name"] for m in json.load(sys.stdin)["models"]]
want = os.environ.get("ORB_LLM_MODEL") or "gemma4:e2b"
polish = os.environ.get("ORB_POLISH_MODEL") or "qwen2.5:1.5b"
assert want in names, f"the LLM {want} is not installed; installed: {names}"
assert polish in names, f"the polish LLM {polish} is not installed (ollama pull {polish}); installed: {names}"
print(f"Ollama answers, and the LLM {want} and the polish LLM {polish} are installed")
'
}

c_gemini_key() { # informational: the value is never printed, and no call goes to Google
  if [[ -n "$(setting GOOGLE_API_KEY)" ]]; then echo "GOOGLE_API_KEY is set: the Gemini Live mode is available"
  else echo "GOOGLE_API_KEY is not set: the Gemini Live mode is off (the orb pipeline works without it)"; fi
}

c_agent() {
  curl -sf -m 30 "$AGENT/health" | "$PY" -I -c '
import json, sys
h = json.load(sys.stdin)
llm = h["agents"]["discussion"]["llm"]
assert h["ok"], h
assert llm.get("ok"), f"the LLM is not ready: {llm}"
print("the agent service answers; the LLM is", llm["model"])
'
}

c_voice_status() {
  curl -sf -m 10 "$AGENT/v1/voice/status" | "$PY" -I -c '
import json, sys
v = json.load(sys.stdin)
assert v, "voice is off"
assert v["livekit"], "the LiveKit server does not answer"
assert v["worker"], "the voice worker does not run"
print("LiveKit and the voice worker answer; the signal is forwarded:", v["proxy"], "; dev keys:", v["dev"])
'
}

c_speech() {
  curl -sf -m 10 "$SPEECH/health" | "$PY" -I -c '
import json, sys
h = json.load(sys.stdin)
assert h.get("ok", True), h
print("the speech service answers")
'
}

c_speech_roundtrip() {
  "$PY" -I - "$SPEECH" <<'EOF'
import re, sys, httpx
speech = sys.argv[1]
text = "This is a test of the speech service. It reads one short sentence."
with httpx.Client(timeout=120) as c:
    tts = c.post(f"{speech}/v1/tts", json={"text": text, "voice": "am_michael"})
    tts.raise_for_status()
    rate = int(tts.headers["x-sample-rate"])
    assert len(tts.content) > rate * 2, "the TTS made less than one second of audio"
    stt = c.post(f"{speech}/v1/stt", params={"sample_rate": rate, "language": "en"}, content=tts.content)
    stt.raise_for_status()
heard = stt.json()["text"]
words = re.sub(r"[^a-z ]", "", heard.lower())
missing = [w for w in ("test", "speech", "service", "sentence") if w not in words]
# one missed word is allowed: Whistle, the default STT, makes more errors than Parakeet (decision 0007)
assert len(missing) <= 1, f"the STT did not hear {missing}: {heard!r}"
print(f"the TTS made {len(tts.content) / 2 / rate:.1f} s of audio; the STT heard {heard!r} in {stt.json()['ms']} ms")
EOF
}

c_gate() {
  local code
  code="$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$AGENT/")"
  [[ "$code" == 200 ]] || { echo "this computer gets $code for /, not 200"; return 1; }
  code="$(curl -s -m 10 -o /dev/null -w '%{http_code}' -H 'Host: check.example.net' "$AGENT/v1/sessions")"
  [[ "$code" == 403 ]] || { echo "a foreign Host gets $code for /v1/sessions, not 403"; return 1; }
  code="$(curl -s -m 10 -o /dev/null -w '%{http_code}' -H 'X-Forwarded-For: 203.0.113.9' "$AGENT/v1/sessions")"
  [[ "$code" == 403 ]] || { echo "a request with a proxy header gets $code, not 403"; return 1; }
  # with no token, LiveKit itself answers with a 4xx. A 404 would mean no route here, a 502 that LiveKit does not answer.
  code="$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$AGENT/rtc/v1/validate")"
  case "$code" in
    400|401|403) ;;
    *) echo "/rtc/v1/validate gives $code: the voice signal does not reach LiveKit"; return 1 ;;
  esac
  echo "this computer gets 200; a foreign Host and a proxy header get 403; the voice signal reaches LiveKit ($code without a token)"
}

c_text_turn() {
  "$PY" -I - "$AGENT" <<'EOF'
import json, sys, time, httpx
agent = sys.argv[1]
with httpx.Client(timeout=180) as c:
    s = c.post(f"{agent}/v1/sessions", json={"topic": "Check of the stack"}).json()
    t0, events = time.monotonic(), []
    body = {"text": "We should store passwords in a plain JSON file.", "channel": "web"}
    with c.stream("POST", f"{agent}/v1/sessions/{s['id']}/turns", json=body) as r:
        r.raise_for_status()
        events = [json.loads(line) for line in r.iter_lines() if line]
kinds = [e["type"] for e in events]
assert "error" not in kinds, [e for e in events if e["type"] == "error"]
reply = "".join(e["data"] for e in events if e["type"] == "token")
assert len(reply.split()) >= 3, f"the reply is too short: {reply!r}; events: {kinds}"
decision = next((e["data"] for e in events if e["type"] == "decision"), {})
print(f"one turn took {time.monotonic() - t0:.1f} s; persona {decision.get('persona')}; reply {reply[:70]!r}")
EOF
}

c_tunnel() {
  local domain key code
  domain="$(setting ORB_NGROK_DOMAIN)"; key="$(setting ORB_SHARE_KEY)"
  [[ -n "$key" ]] || { echo "ORB_SHARE_KEY is not set, so the tunnel refuses every request"; return 1; }
  code="$(curl -s -m 20 -o /dev/null -w '%{http_code}' -H 'ngrok-skip-browser-warning: 1' "https://$domain/v1/voice/status")"
  [[ "$code" == 403 ]] || { echo "through the tunnel without the key: $code, not 403"; return 1; }
  # the key goes to curl on its standard input, so that `ps` does not show it
  printf 'header = "X-Orb-Key: %s"\n' "$key" | curl -s -m 20 -K - -H 'ngrok-skip-browser-warning: 1' "https://$domain/v1/voice/status" \
    | "$PY" -I -c '
import json, sys
v = json.load(sys.stdin)
assert v and v["livekit"] and v["worker"] and v["proxy"], v
print("through the tunnel: 403 without the key; with the key the voice status is good and the signal is forwarded")
'
}

c_e2e() {
  "$PY" "$ROOT/scripts/e2e_voice.py" >"$LOGS/e2e.log" 2>&1
  local rc=$?
  # the lines up to PASS or FAIL: LiveKit's client prints a harmless traceback when the Python process exits
  if grep -qE '^(PASS|FAIL)$' "$LOGS/e2e.log"; then
    sed -E -n '1,/^(PASS|FAIL)$/p' "$LOGS/e2e.log"
  else
    tail -n 20 "$LOGS/e2e.log"
  fi
  return $rc
}

c_no_secrets_in_logs() { # the values (never printed) of the secrets of .env must not be in a log of the stack
  local hits
  hits="$(grep -lF -f <(for n in ORB_SHARE_KEY LIVEKIT_API_SECRET CARTESIA_API_KEY GOOGLE_API_KEY; do
      v="$(setting "$n")"; [[ ${#v} -ge 12 ]] && printf '%s\n' "$v"; done) \
      "$ROOT"/data/logs/{ollama,agent,livekit,speech,voice,ngrok}.log "$LOGS"/*.log 2>/dev/null)"
  if [[ -n "$hits" ]]; then
    echo "a secret of .env is in these logs:"
    echo "$hits" | sed "s#$ROOT/#  #"
    return 1
  fi
  echo "no secret of .env is in the logs of the stack"
}

c_closed() {
  local busy="" p left
  for p in $PORTS; do
    lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1 && busy="$busy $p"
  done
  [[ -z "$busy" ]] || { echo "these ports still have a listener:$busy"; return 1; }
  left="$(ps -eo pid,command | grep -E 'orb_agent\.server|orb_speech\.server|orb_voice dev|livekit-server --(config|dev)|ngrok http 127\.0\.0\.1:8100' | grep -v grep)"
  if [[ -n "$left" ]]; then
    echo "an orb process still runs:"
    echo "$left" | cut -c1-110
    return 1
  fi
  echo "ports $PORTS are free, and no orb process runs"
}

summary() {
  local r status secs name
  echo
  echo "Summary ($((SECONDS - START)) s)"
  for r in ${RESULTS[@]+"${RESULTS[@]}"}; do
    IFS='|' read -r status secs name <<<"$r"
    printf '  %s  %3s s  %s\n' "$status" "$secs" "$name"
  done
  echo
  if (( FAILED == 0 )); then
    echo "ALL CHECKS PASSED. Logs: ${LOGS#$ROOT/}/"
  else
    echo "$FAILED CHECK(S) FAILED. Logs: ${LOGS#$ROOT/}/"
  fi
}

# --- the run ------------------------------------------------------------------------------------------------------
echo "orb stack check, $(date '+%Y-%m-%d %H:%M:%S')"
if ! preflight; then
  exit 2
fi
mkdir -p "$LOGS"
WITH_TUNNEL=0
if (( TUNNEL )) && [[ -n "$(setting ORB_NGROK_DOMAIN)" ]]; then WITH_TUNNEL=1; fi

if (( UNIT )); then
  run_check "unit tests (make test)" unit_tests
fi

echo "Starting the stack (scripts/dev.sh)..."
start_stack
if run_check "the stack starts" wait_ready; then
  run_check "Ollama and the LLM" c_ollama
  run_check "agent service health" c_agent
  run_check "Gemini key (Gemini Live mode)" c_gemini_key
  run_check "voice status" c_voice_status
  run_check "speech service" c_speech
  run_check "speech round trip (TTS, then STT)" c_speech_roundtrip
  run_check "access gate and voice signal forwarder" c_gate
  run_check "one text turn (Jev, debt rules, LLM)" c_text_turn
  if (( WITH_TUNNEL )); then
    run_check "through the ngrok domain" c_tunnel
  fi
  if (( E2E )); then
    run_check "voice end-to-end (a synthetic person talks to the agent)" c_e2e
  fi
fi

stop_stack
run_check "no secret in the logs" c_no_secrets_in_logs
run_check "everything is closed" c_closed
summary
exit $(( FAILED > 0 ))
