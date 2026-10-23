# orb: every command for local work. Run `make help`.

PY      ?= $(HOME)/miniconda3/envs/orb/bin/python
CONDA   ?= $(HOME)/miniconda3/bin/conda
export NEEDLE_TELEMETRY := 0
export JEV_ENV_FILE ?= $(HOME)/jev-test/.env

.PHONY: help setup brew env models test test-py dev dev-text e2e check bench replay

help:
	@echo "make setup    install system libraries, the conda env and the speech models (once)"
	@echo "make test     every test (no network, no API cost)"
	@echo "make dev      start every service; then open http://127.0.0.1:8100"
	@echo "make dev-text start only the text chat (agent service and Ollama)"
	@echo "make e2e      the end-to-end voice test (needs 'make dev' in another terminal)"
	@echo "make check    start the whole stack, test it, and stop it (about 4 minutes; --quick: 1 minute)"
	@echo "make bench    build the git-ignored test audio, then open http://127.0.0.1:8100/bench"
	@echo "make replay CHAT=data/chats/<id>.json   replay a saved chat with live Jev calls (costs about \$$0.001)"

setup: brew env models

brew:
	HOMEBREW_NO_AUTO_UPDATE=1 brew install livekit ollama ffmpeg
	ollama pull gemma4:e2b

env:
	@if [ -x "$(PY)" ]; then echo "the orb env exists"; else $(CONDA) env create -f environment.yml; fi
	$(PY) -m pip install -q -e packages/python/orb-core -e packages/python/orb-discussion \
		-e "packages/python/orb-speech[speech-lm]" -e packages/python/orb-bench -e apps/agent -e apps/voice-agent

models:
	$(PY) -m orb_speech.models

test: test-py

test-py:
	cd packages/python/orb-core && $(PY) -m pytest -q
	cd packages/python/orb-discussion && $(PY) -m pytest -q
	cd packages/python/orb-speech && $(PY) -m pytest -q
	cd packages/python/orb-bench && $(PY) -m pytest -q
	cd apps/agent && $(PY) -m pytest -q -W ignore::jwt.warnings.InsecureKeyLengthWarning
	cd apps/voice-agent && $(PY) -m pytest -q

dev:
	scripts/dev.sh

dev-text:
	scripts/dev.sh --no-voice

e2e:
	$(PY) scripts/e2e_voice.py

check:
	scripts/check_stack.sh $(ARGS)

bench:
	$(PY) -m orb_bench corpus

replay:
	$(PY) -m orb_discussion.replay $(CHAT)
