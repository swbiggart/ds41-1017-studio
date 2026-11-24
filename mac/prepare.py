import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from . import repack

INDEX_SHA256 = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"

ORIGINAL_PATTERNS = [
    "model-*.safetensors",
    "engram/*.safetensors",
    "*.json",
    "encoding/*.py",
    "encoding/README.md",
    "inference/*.py",
    "inference/README.md",
    "chat_template.jinja",
    "LICENSE",
    "README.md",
]

REPACK_PATTERNS = [
    "model-*.safetensors",
    "engram/*",
    "assets/*",
    "*.json",
    "encoding/*.py",
    "encoding/README.md",
    "inference/*.py",
    "inference/README.md",
    "chat_template.jinja",
    "LICENSE",
    "README.md",
]


def is_repacked(model_dir):
    return (Path(model_dir) / repack.MANIFEST_NAME).is_file()


def has_original_checkpoint(model_dir):
    """Return True when model_dir holds DeepSeek's pinned un-repacked index."""
    model_dir = Path(model_dir)
    if is_repacked(model_dir):
        return False
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    return hashlib.sha256(index_path.read_bytes()).hexdigest() == INDEX_SHA256


def _relocate_subdir(src_dir: Path, dst_dir: Path, sentinel: str) -> None:
    """Move downloaded engram/ or assets/ files when local.toml points outside model_dir."""
    if src_dir.resolve() == dst_dir.resolve():
        return
    if not (src_dir / sentinel).is_file() or (dst_dir / sentinel).is_file():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        target = dst_dir / item.name
        if not target.exists():
            shutil.move(str(item), str(target))


def fetch(s, *, original=False):
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    use_original = original or has_original_checkpoint(s.model)
    if use_original:
        snapshot_download(
            s.repo,
            revision=s.revision,
            local_dir=s.model,
            allow_patterns=ORIGINAL_PATTERNS,
        )
        return "original"

    snapshot_download(
        s.repack_repo,
        revision=s.repack_revision,
        local_dir=s.model,
        allow_patterns=REPACK_PATTERNS,
    )
    _relocate_subdir(s.model / "engram", s.engram, "engram-manifest.json")
    _relocate_subdir(s.model / "assets", s.assets, "engram-token-map.bin")
    return "repacked"


def check_shards(s):
    if is_repacked(s.model):
        repack.verify_repack(s.model, expected_source_index_sha256=INDEX_SHA256)
        return
    index_path = s.model / "model.safetensors.index.json"
    raw_index = index_path.read_bytes()
    if hashlib.sha256(raw_index).hexdigest() != INDEX_SHA256:
        raise ValueError(
            f"The index in {s.model} differs from DeepSeek's pinned original checkpoint. Run python -m mac fetch."
        )
    index = json.loads(raw_index)
    for shard in sorted(set(index["weight_map"].values())):
        if not (s.model / shard).is_file():
            raise FileNotFoundError(
                f"missing {shard} in {s.model}. Run python -m mac fetch"
            )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_token_map(s, table):
    meta = json.loads(Path(str(table) + ".json").read_text())
    for key, path in [
        ("packed_sha256", table),
        ("tokenizer_sha256", s.model / "tokenizer.json"),
        ("config_sha256", s.model / "config.json"),
    ]:
        if meta[key] != sha256(path):
            raise ValueError(f"token map checksum mismatch for {path}")


def check_engram_manifest(engram_dir):
    engram_dir = Path(engram_dir)
    manifest = json.loads((engram_dir / "engram-manifest.json").read_text())
    if sorted(t["layer"] for t in manifest["tables"]) != [1, 14]:
        raise ValueError("Engram manifest must include layers 1 and 14")
    for table in manifest["tables"]:
        path = engram_dir / table["file"]
        if not path.is_file() or path.stat().st_size != table["size"]:
            raise ValueError(f"incorrect packed table size for {path}")
    return manifest


def prepare(s):
    check_shards(s)
    s.assets.mkdir(parents=True, exist_ok=True)
    s.engram.mkdir(parents=True, exist_ok=True)
    with (s.engram / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        table = s.assets / "engram-token-map.bin"
        if table.exists() and Path(str(table) + ".json").exists():
            check_token_map(s, table)
        else:
            with tempfile.TemporaryDirectory(dir=s.assets) as tmp:
                pending = Path(tmp) / table.name
                subprocess.run(
                    [
                        s.bin / "tf-dsv41-token-map",
                        "--tokenizer",
                        s.model / "tokenizer.json",
                        "--config",
                        s.model / "config.json",
                        "--output",
                        pending,
                    ],
                    check=True,
                )
                check_token_map(s, pending)
                pending.replace(table)
                Path(str(pending) + ".json").replace(str(table) + ".json")
        has_manifest = (s.engram / "engram-manifest.json").is_file()
        if is_repacked(s.model):
            if not has_manifest:
                raise ValueError(
                    f"{s.model} is a repacked checkpoint without packed Engram tables at {s.engram}. "
                    "Run python -m mac fetch (pre-repacked HF release) or prepare from the original checkpoint."
                )
            check_engram_manifest(s.engram)
            return
        operation = "verify" if has_manifest else "pack"
        subprocess.run(
            [s.bin / "tf-dsv41-engram-pack", operation, s.model, s.engram], check=True
        )
