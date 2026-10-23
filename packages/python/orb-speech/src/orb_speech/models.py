"""Where the model files are, where they come from, and their SHA-256 checksums.

    python -m orb_speech.models          # fetch every missing file, then check every checksum
    python -m orb_speech.models --check  # only check

    python -m orb_speech.models --all    # also the full Kokoro model and the maya1 and indic-speak TTS (4.3 GB)

Files go to ORB_MODELS_DIR (default ~/.cache/orb/models). Whistle goes to the cache of the cactus-needle package.
Parakeet, maya1, indic-speak and SNAC go to the Hugging Face cache, at pinned commits, so they are not stored twice.
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
}
WHISTLE_SHA256 = "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb"  # whistle.cact 2.0.0, Apache-2.0
PARAKEET_REPO = "nvidia/parakeet-tdt-0.6b-v3"  # CC-BY-4.0: credit NVIDIA where the model is named
PARAKEET_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
PARAKEET_FILES = {  # name: (SHA-256, size)
    "config.json": ("e747b85e1bdfd300c8b8ac63bac8dd5221f8fe9bc275b48d06c735fcd6971b6e", 1153),
    "generation_config.json": ("b141de6ec6d7f982ece13f98f604e3fe1807ea9c0e839185d0ab7064604209d0", 289),
    "model.safetensors": ("3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11", 2508311120),
    "processor_config.json": ("8346a93a3b987fa1dec57a78f045cd0817d21786589a5a096b41a57a446fd1d7", 392),
    "tokenizer.json": ("bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91", 1159960),
    "tokenizer_config.json": ("0b2fe0037599ee335f0b972fa682bf0ece74e4ccfec755cb7daa3405d3d3e874", 290),
}
# The speech LMs for text to speech (tts_lm.py), 4-bit MLX builds. Both are Llama 3.2 3B fine-tunes: the Llama 3.2
# Community License applies too. Both write SNAC codes.
MAYA1_REPO = "0xhb/maya1-mlx-4bit"  # Apache-2.0 (maya-research/maya1)
MAYA1_REVISION = "50f595b36cf1d59364b5b6d8356461a534930ad7"
MAYA1_FILES = {
    "config.json": ("d36ca2e5e2b60d2008309fc266c54b43f005f5de248e668843237d4762ffd076", 1176),
    "model.safetensors": ("3419d8c0e4c73d260480df16fc928cf7808442ac53ffd4fb18d120fddf5b14ae", 1857097241),
    "model.safetensors.index.json": ("6569c7e59e312065620645d79adae9481fa95285d2b850b9df1fd92d3d5d72bf", 45760),
    "tokenizer.json": ("6c5e5b1d89b7e3738e5a5a4f93c326d8f3292ea83f9c560b8dbb6d66fb851973", 22853258),
    "tokenizer_config.json": ("74b070928033d10c4d85c9dadcd22b4b713d7c937c0b3cacf380d2f17e3afbb1", 330),
}
INDIC_REPO = "iniquitous/indic-speak-mlx-4bit"  # Indic Open Model License (bodhan-ai/indic-speak)
INDIC_REVISION = "d137de5c6011b982635d4f7cada2b82b3bdd5d94"
INDIC_FILES = {
    "config.json": ("064b0b0b80623685ad8ca33ee74593f7fd519467aa722fe341d11d4a32d073de", 1234),
    "generation_config.json": ("690a45cdedd4a8a0a318a2b046c646c3e089c1934c2b3bd9eba8aab7523cf493", 205),
    "model.safetensors": ("e942e73e39f83c537ef05847b9da6aa25109307db27aedf81de863a4ac7cd325", 1857097241),
    "model.safetensors.index.json": ("6569c7e59e312065620645d79adae9481fa95285d2b850b9df1fd92d3d5d72bf", 45760),
    "tokenizer.json": ("645579dd8b7040a2e53808a10e6b5f6961d1b625a6bee768710fdf507465f53f", 22681215),
    "tokenizer_config.json": ("6fd058b81ceb4e4538305bb4a010454a0b6e52b5aa545099bb41308247c2ec7d", 402),
    # A pickle file: it is loaded with weights_only=True, so it can hold only tensors and plain data.
    "vocos/best.pt": ("867426748123693b8ca5b26d9b407af158ac0bbae3f140901bc004b7ebc5738d", 466966065),
    "Bodhan_AI_Open_Model_License.md": ("b39f2a9cb2d5ace3a03f4425cbc6db38ebd9a53ab7a7648d67dc1e0998ad75c2", 25546),
    "indic-open-license.md": ("5dd97d2327cdd4fc87e974528fee8bbe16420c1fed41f33778be58ab809386a0", 3565),
}
SNAC_REPO = "hubertsiuzdak/snac_24khz"  # MIT
SNAC_REVISION = "d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"
SNAC_FILES = {
    "config.json": ("e119b9366d4f5e73c6ca5f31137c4ff361578bbb132953a5203afe037c4012be", 300),
    "pytorch_model.bin": ("4b8164cc6606bfa627f1a784734c1e539891518f1191ed9194fe1e3b9b4bff40", 79488254),  # weights_only
}
REQUIRED = ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")


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


def hf_dir(repo: str, revision: str, files: dict[str, tuple[str, int]], fetch_missing: bool = True) -> Path:
    """The folder of a Hugging Face repo at a pinned commit, every file checked. Fetches missing files unless told
    not to."""
    from huggingface_hub import snapshot_download
    try:
        folder = Path(snapshot_download(repo, revision=revision, allow_patterns=list(files),
                                        local_files_only=not fetch_missing))
    except Exception as e:
        raise FileNotFoundError(f"{repo} is not fetched ({type(e).__name__}). Run: python -m orb_speech.models "
                                f"{'' if repo == PARAKEET_REPO else '--all'}".rstrip()) from e
    for name, (digest, size) in files.items():
        f = folder / name
        if not f.is_file() or f.stat().st_size != size or sha256(f) != digest:
            raise ValueError(f"{f} is missing or changed. Delete the Hugging Face cache of {repo} and run: "
                             "python -m orb_speech.models")
    return folder


def parakeet_dir(fetch_missing: bool = True) -> Path:
    return hf_dir(PARAKEET_REPO, PARAKEET_REVISION, PARAKEET_FILES, fetch_missing)


def maya1_dir(fetch_missing: bool = True) -> Path:
    return hf_dir(MAYA1_REPO, MAYA1_REVISION, MAYA1_FILES, fetch_missing)


def indic_speak_dir(fetch_missing: bool = True) -> Path:
    return hf_dir(INDIC_REPO, INDIC_REVISION, INDIC_FILES, fetch_missing)


def snac_dir(fetch_missing: bool = True) -> Path:
    return hf_dir(SNAC_REPO, SNAC_REVISION, SNAC_FILES, fetch_missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only check the files that are there")
    ap.add_argument("--all", action="store_true",
                    help="also the full-size Kokoro model (325 MB) and the maya1 and indic-speak TTS (4.3 GB)")
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
    folders = [(PARAKEET_REPO, parakeet_dir, "CC-BY-4.0")]
    if args.all:
        folders += [(MAYA1_REPO, maya1_dir, "Apache-2.0 + Llama 3.2"),
                    (INDIC_REPO, indic_speak_dir, "Indic Open Model License + Llama 3.2"), (SNAC_REPO, snac_dir, "MIT")]
    for repo, folder_of, license_ in folders:
        try:
            p = folder_of(fetch_missing=not args.check)
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) // 1_000_000
            print(f"ok      {repo}  ({size} MB, {license_}) at {p}")
        except (FileNotFoundError, ValueError, OSError) as e:
            ok = False
            print(f"FAILED  {repo}: {e}")
    try:
        p = whistle_weights()
        print(f"ok      whistle.cact  ({p.stat().st_size // 1_000_000} MB, Apache-2.0) at {p}")
    except Exception as e:
        ok = False
        print(f"FAILED  whistle.cact: {e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
