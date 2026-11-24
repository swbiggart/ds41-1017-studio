import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from . import repack

INDEX_SHA256 = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"


def fetch(s):
    from huggingface_hub import snapshot_download

    snapshot_download(
        s.repo,
        revision=s.revision,
        local_dir=s.model,
        allow_patterns=[
            "model-*.safetensors",
            "engram/*.safetensors",
            "*.json",
            "encoding/*.py",
            "inference/*.py",
            "chat_template.jinja",
            "LICENSE",
            "README.md",
        ],
    )


def is_repacked(model_dir):
    return (Path(model_dir) / repack.MANIFEST_NAME).is_file()


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
        operation = (
            "verify" if (s.engram / "engram-manifest.json").is_file() else "pack"
        )
        if operation == "pack" and is_repacked(s.model):
            raise ValueError(
                f"{s.model} is a repacked checkpoint without raw Engram shards. Run python -m mac prepare on the original checkpoint before repacking."
            )
        subprocess.run(
            [s.bin / "tf-dsv41-engram-pack", operation, s.model, s.engram], check=True
        )
