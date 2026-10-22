# orb: every command for local work. Run `make help`.

PY      ?= $(HOME)/miniconda3/envs/orb/bin/python
CONDA   ?= $(HOME)/miniconda3/bin/conda
export CGO_ENABLED := 1
export NEEDLE_TELEMETRY := 0
export JEV_ENV_FILE ?= $(HOME)/jev-test/.env
GO_LDFLAGS := -ldflags=-extldflags=-Wl,-no_warn_duplicate_libraries
GO_MODULES := packages/go/lk-go-agent-sdk packages/go/lk-go-agent-sdk/plugins/silero packages/go/orb-plugins apps/voice

.PHONY: help setup brew env models test test-py test-go dev dev-text e2e replay

help:
	@echo "make setup    install system libraries, the conda env and the speech models (once)"
	@echo "make test     every Python and Go test (no network, no API cost)"
	@echo "make dev      start every service; then open http://127.0.0.1:8100"
	@echo "make dev-text start only the text chat (agent service and Ollama)"
	@echo "make e2e      the end-to-end voice test (needs 'make dev' in another terminal)"
	@echo "make replay CHAT=data/chats/<id>.json   replay a saved chat with live Jev calls (costs about \$$0.001)"

setup: brew env models

brew:
	HOMEBREW_NO_AUTO_UPDATE=1 brew install livekit opus opusfile libsoxr onnxruntime pkg-config go ollama
	ollama pull gemma4:e2b

env:
	@if [ -x "$(PY)" ]; then echo "the orb env exists"; else $(CONDA) env create -f environment.yml; fi
	$(PY) -m pip install -q -e packages/python/orb-core -e packages/python/orb-discussion -e packages/python/orb-speech -e apps/agent

models:
	$(PY) -m orb_speech.models

test: test-py test-go

test-py:
	cd packages/python/orb-core && $(PY) -m pytest -q
	cd packages/python/orb-discussion && $(PY) -m pytest -q
	cd packages/python/orb-speech && $(PY) -m pytest -q
	cd apps/agent && $(PY) -m pytest -q -W ignore::jwt.warnings.InsecureKeyLengthWarning

test-go:
	@for m in $(GO_MODULES); do echo "go test $$m"; (cd $$m && go vet ./... && go test $(GO_LDFLAGS) -race -count=1 ./...) || exit 1; done

dev:
	scripts/dev.sh

dev-text:
	scripts/dev.sh --no-voice

e2e:
	$(PY) scripts/e2e_voice.py

replay:
	$(PY) -m orb_discussion.replay $(CHAT)
