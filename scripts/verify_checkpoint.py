"""Check the local DeepSeek-V4.1-Flash-1017 files against DeepSeek's pinned Hub revision."""

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path

from huggingface_hub import HfApi

RECIPE = Path(__file__).resolve().parents[1] / "recipe.toml"
INDEX = "model.safetensors.index.json"
REQUIRED = {"config.json", "tokenizer.json", INDEX}


def verify_file(path, entry, checksums):
    size = path.stat().st_size
    if size != entry.size:
        raise ValueError(f"expected {entry.size:,} bytes, found {size:,}")
    if checksums:
        if entry.lfs:
            expected = entry.lfs.sha256
            digest = hashlib.sha256()
        else:
            expected = entry.blob_id
            digest = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"{digest.name} mismatch")
    return size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="also hash file contents (reads ~510 GB)",
    )
    args = parser.parse_args()
    with RECIPE.open("rb") as source:
        model = tomllib.load(source)["model"]
    repo, revision = model["source_repo"], model["source_revision"]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError("Hub returned an unexpected revision")
    manifest = {entry.rfilename: entry for entry in info.siblings}
    with (args.checkpoint / INDEX).open() as source:
        shards = set(json.load(source)["weight_map"].values())
    published = {name for name in manifest if name.endswith(".safetensors")}
    if shards != published:
        print(
            f"Index disagrees with Hub: missing {sorted(published - shards)}, "
            f"unexpected {sorted(shards - published)}",
            file=sys.stderr,
        )
        return 1
    files = (
        REQUIRED
        | shards
        | {name for name in manifest if (args.checkpoint / name).exists()}
    )
    errors = 0
    total = 0
    for name in sorted(files):
        try:
            total += verify_file(args.checkpoint / name, manifest[name], args.sha256)
        except (OSError, ValueError) as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            errors += 1
    if errors:
        print(f"{errors} file(s) failed verification", file=sys.stderr)
        return 1
    mode = "size and checksum" if args.sha256 else "size only"
    print(f"{repo}@{revision}: verified {len(files)} files, {total:,} bytes ({mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
