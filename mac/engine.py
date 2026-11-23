"""Build and verify the patched TensorFold v1.3.3 Metal engine and asset tools."""

import fcntl
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from .config import ROOT

MANIFEST_PATH = ROOT / "patches/manifest.json"
BINARIES = ("tensorfold-dsv41", "tf-dsv41-token-map", "tf-dsv41-engram-pack")


def load_manifest():
    manifest = json.loads(MANIFEST_PATH.read_bytes())
    if not re.fullmatch("[0-9a-f]{40}", manifest["upstream_commit"]):
        raise ValueError("Invalid upstream_commit in patches/manifest.json")
    if not re.fullmatch("[0-9a-f]{40}", manifest["source_tree"]):
        raise ValueError("Invalid source_tree in patches/manifest.json")
    for patch in manifest["patches"]:
        if not re.fullmatch("[0-9a-f]{64}", patch["sha256"]):
            raise ValueError(f"Invalid sha256 for patch {patch['file']}")
    return manifest


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def verify_patches(manifest):
    for patch in manifest["patches"]:
        path = ROOT / "patches" / patch["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing patch file {path}")
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != patch["sha256"]:
            raise ValueError(f"Patch checksum mismatch for {path.name}")


def verify_source(source, manifest):
    if git(source, "rev-parse", "HEAD") != manifest["upstream_commit"]:
        raise ValueError(f"Unexpected TensorFold commit in {source}")
    if git(source, "write-tree") != manifest["source_tree"]:
        raise ValueError(f"Unexpected patched source tree in {source}")
    subprocess.run(
        ["git", "-C", str(source), "diff", "--quiet", manifest["source_tree"], "--"],
        check=True,
    )
    if git(source, "ls-files", "--others", "--exclude-standard"):
        raise ValueError(f"Untracked source files in {source}")


def build(s):
    manifest = load_manifest()
    if (
        manifest["upstream_tag"] != s.engine_tag
        or manifest["upstream_commit"] != s.engine_sha
    ):
        raise ValueError("recipe.toml engine pin disagrees with patches/manifest.json")
    verify_patches(manifest)
    s.source.parent.mkdir(parents=True, exist_ok=True)
    with (s.source.parent / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run([ROOT / "scripts/build.sh", "--zig-only"], check=True)
        if not s.source.exists():
            with TemporaryDirectory(dir=s.source.parent) as tmp:
                stage = Path(tmp) / "source"
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--quiet",
                        "--branch",
                        s.engine_tag,
                        "--depth",
                        "1",
                        s.engine_repo,
                        str(stage),
                    ],
                    check=True,
                )
                if git(stage, "rev-parse", "HEAD") != s.engine_sha:
                    raise ValueError(
                        f"{s.engine_tag} differs from the pinned TensorFold commit"
                    )
                for patch in manifest["patches"]:
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(stage),
                            "apply",
                            "--index",
                            str(ROOT / "patches" / patch["file"]),
                        ],
                        check=True,
                    )
                verify_source(stage, manifest)
                stage.rename(s.source)
        verify_source(s.source, manifest)
        zig = ROOT / ".local/zig-pypi/ziglang/zig"
        with TemporaryDirectory(dir=s.source.parent) as tmp:
            stage = Path(tmp)
            subprocess.run(
                [
                    str(zig),
                    "build",
                    "dsv41-metal-serve",
                    "dsv41-assets",
                    "dsv41-engram-pack",
                    "--prefix",
                    str(stage),
                ],
                cwd=s.source,
                check=True,
            )
            out = s.source / "zig-out"
            out_bin = out / "bin"
            out_bin.mkdir(parents=True, exist_ok=True)
            binaries = {}
            for name in BINARIES:
                exe = stage / "bin" / name
                if not exe.is_file():
                    raise FileNotFoundError(f"Expected built binary {exe}")
                binaries[name] = sha256(exe.read_bytes()).hexdigest()
            receipt = stage / "build-receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "upstream_tag": manifest["upstream_tag"],
                        "upstream_commit": manifest["upstream_commit"],
                        "source_tree": manifest["source_tree"],
                        "zig_version": "0.17.0",
                        "patches": {p["file"]: p["sha256"] for p in manifest["patches"]},
                        "binaries": binaries,
                    },
                    indent=2,
                )
                + "\n"
            )
            for name in BINARIES:
                (stage / "bin" / name).replace(out_bin / name)
            receipt.replace(out / "build-receipt.json")


def verify_build(s):
    manifest = load_manifest()
    receipt_path = s.source / "zig-out/build-receipt.json"
    if not receipt_path.is_file() or not s.serve_binary.is_file():
        raise FileNotFoundError(
            f"Missing verified build at {s.serve_binary}. Run python -m mac build"
        )
    receipt = json.loads(receipt_path.read_bytes())
    if (
        receipt.get("upstream_commit") != manifest["upstream_commit"]
        or receipt.get("source_tree") != manifest["source_tree"]
    ):
        raise ValueError(
            "Built engine does not match patches/manifest.json. Run python -m mac build"
        )
    expected = receipt.get("binaries", {}).get("tensorfold-dsv41")
    actual = sha256(s.serve_binary.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise ValueError(
            "tensorfold-dsv41 binary differs from build-receipt.json. Run python -m mac build"
        )
    return receipt
