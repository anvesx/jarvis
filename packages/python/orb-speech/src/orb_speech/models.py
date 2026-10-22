"""Where the model files are, where they come from, and their SHA-256 checksums.

    python -m orb_speech.models          # fetch every missing file, then check every checksum
    python -m orb_speech.models --check  # only check

Files go to ORB_MODELS_DIR (default ~/.cache/orb/models). Whistle goes to the cache of the cactus-needle package.
A file with a wrong checksum is refused, because a changed model file can change what the voice agent does.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"


@dataclass(frozen=True)
class ModelFile:
    name: str
    sha256: str
    size: int
    url: str | None = None        # a direct download
    hf_repo: str | None = None    # or a Hugging Face file
    license: str = ""


FILES = {
    "kokoro-v1.0.int8.onnx": ModelFile("kokoro-v1.0.int8.onnx",
                                       "ae315a79b623f244700e4afb9246c46a26066782e049ba174bf3ba433970ee9c", 114119327,
                                       url=KOKORO_URL + "kokoro-v1.0.int8.onnx", license="Apache-2.0 (Kokoro-82M)"),
    "kokoro-v1.0.onnx": ModelFile("kokoro-v1.0.onnx",
                                  "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a", 325505369,
                                  url=KOKORO_URL + "kokoro-v1.0.onnx", license="Apache-2.0 (Kokoro-82M)"),
    "voices-v1.0.bin": ModelFile("voices-v1.0.bin",
                                 "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d", 28214398,
                                 url=KOKORO_URL + "voices-v1.0.bin", license="Apache-2.0 (Kokoro-82M)"),
    "smart-turn-v3.2-cpu.onnx": ModelFile("smart-turn-v3.2-cpu.onnx",
                                          "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f", 8679182,
                                          hf_repo="pipecat-ai/smart-turn-v3", license="BSD-2-Clause"),
}
WHISTLE_SHA256 = "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb"  # whistle.cact 2.0.0, Apache-2.0
REQUIRED = ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin", "smart-turn-v3.2-cpu.onnx")


def models_dir() -> Path:
    return Path(os.environ.get("ORB_MODELS_DIR", "~/.cache/orb/models")).expanduser()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def path(name: str, verify: bool = False) -> Path:
    """The path of a model file. Raises FileNotFoundError with the fix when it is missing."""
    p = models_dir() / name
    if not p.is_file():
        raise FileNotFoundError(f"The model file {p} is missing. Run: python -m orb_speech.models")
    if verify and FILES.get(name) and sha256(p) != FILES[name].sha256:
        raise ValueError(f"The checksum of {p} is wrong. Delete the file and run: python -m orb_speech.models")
    return p


def fetch(name: str) -> Path:
    spec = FILES[name]
    dest = models_dir() / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        if spec.hf_repo:
            from huggingface_hub import hf_hub_download
            shutil.copy(hf_hub_download(spec.hf_repo, name), dest)
        else:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(spec.url, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(dest)
    if sha256(dest) != spec.sha256:
        dest.unlink()
        raise ValueError(f"{name}: the downloaded file has the wrong checksum, so it was deleted.")
    return dest


def whistle_weights() -> Path:
    """Fetch Whistle through cactus-needle (telemetry off) and check its checksum."""
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    from needle.agent.whistle import _weights_path
    p = Path(_weights_path())
    if sha256(p) != WHISTLE_SHA256:
        raise ValueError(f"The checksum of {p} is wrong. Delete it, then run: python -m orb_speech.models")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only check the files that are there")
    ap.add_argument("--all", action="store_true", help="also fetch the full-size Kokoro model (325 MB)")
    args = ap.parse_args()
    names = list(FILES) if args.all else list(REQUIRED)
    ok = True
    for name in names:
        try:
            p = path(name, verify=True) if args.check else fetch(name)
            print(f"ok      {name}  ({p.stat().st_size // 1_000_000} MB, {FILES[name].license})")
        except (FileNotFoundError, ValueError, OSError) as e:
            ok = False
            print(f"FAILED  {name}: {e}")
    try:
        p = whistle_weights()
        print(f"ok      whistle.cact  ({p.stat().st_size // 1_000_000} MB, Apache-2.0) at {p}")
    except Exception as e:
        ok = False
        print(f"FAILED  whistle.cact: {e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
