"""Build a 128B/16KiB-aligned, per-consumer lossless repack from the pinned official checkpoint.

Every tensor byte is copied verbatim from DeepSeek's pinned Flash-1017 checkpoint
(08c8657f02291e11b6f6b63e0e16014f8e748518), keeping compact FP8 wo_a, compact 32x32
dense scales, and exact MXFP4 routed experts. Shards are separated by consumer
(top, layer-00..39, mtp-0..2, vision), omitting the 4 raw .engram.embed. tensors
already served from the packed DSV41EN1 Engram tables.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import time

INDEX_SHA256 = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
DEFAULT_WORKERS = 10

F_NOCACHE = 48  # macOS fcntl flag to bypass unified page cache during bulk repack
HEADER_ALIGN = 16384  # 16 KiB page-aligned data start
TENSOR_ALIGN = 128  # 128-byte aligned tensor offsets (satisfies Metal uint4 & MTLTensor)
COPY_CHUNK = 16 * 1024 * 1024  # 16 MiB streaming copy chunks
MANIFEST_NAME = "repack-manifest.json"
REPACK_SCHEMA = 1

METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "LICENSE",
    "README.md",
)
METADATA_DIRS = ("encoding", "inference")


def _set_nocache(fd: int) -> None:
    if sys.platform == "darwin":
        try:
            fcntl.fcntl(fd, F_NOCACHE, 1)
        except OSError:
            pass


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def consumer_shard(tensor_name: str) -> str | None:
    """Map an official tensor name to its per-consumer shard filename, or None to skip raw Engram."""
    if ".engram.embed." in tensor_name:
        return None
    if tensor_name.startswith(("vision.", "aligner.", "image_")):
        return "model-vision.safetensors"
    if tensor_name.startswith("layers."):
        layer = int(tensor_name.split(".")[1])
        return f"model-layer-{layer:02d}.safetensors"
    if tensor_name.startswith("mtp."):
        stage = int(tensor_name.split(".")[1])
        return f"model-mtp-{stage}.safetensors"
    return "model-top.safetensors"


def _read_safetensors_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as f:
        n_bytes = f.read(8)
        if len(n_bytes) != 8:
            raise ValueError(f"Truncated safetensors header in {path}")
        (n,) = struct.unpack("<Q", n_bytes)
        if n > (1 << 27):
            raise ValueError(f"Oversized safetensors header in {path}")
        raw = f.read(n)
        if len(raw) != n:
            raise ValueError(f"Truncated safetensors JSON header in {path}")
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return 8 + n, header


def _build_aligned_header(entries: list[tuple[str, str, list[int], int, int]]) -> bytes:
    """Build a safetensors header whose total (8 + len) is a multiple of HEADER_ALIGN."""
    meta = {
        "__metadata__": {
            "format": "pt",
            "repack_schema": str(REPACK_SCHEMA),
            "tensor_align": str(TENSOR_ALIGN),
            "header_align": str(HEADER_ALIGN),
        }
    }
    for name, dtype, shape, start, end in entries:
        meta[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, end],
        }
    raw = json.dumps(meta, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    total = _align_up(8 + len(raw), HEADER_ALIGN)
    pad = total - (8 + len(raw))
    padded = raw + (b" " * pad)
    return struct.pack("<Q", len(padded)) + padded


def _write_one_shard(
    src_dir: Path,
    dst_dir: Path,
    shard_name: str,
    tensor_specs: list[tuple[str, str, str, list[int], int, int]],
    *,
    nocache: bool = True,
) -> dict:
    """Write one 128B/16KiB-aligned safetensors shard, returning integrity metadata.

    tensor_specs items: (tensor_name, src_file, dtype, shape, src_abs_start, nbytes)
    """
    # Deterministic order: group by source file and offset for sequential reads,
    # then assign exact 128B-aligned offsets in the destination shard.
    ordered = sorted(tensor_specs, key=lambda x: (x[1], x[4], x[0]))
    entries: list[tuple[str, str, list[int], int, int]] = []
    cursor = 0
    total_payload = 0
    for name, _src_file, dtype, shape, _src_start, nbytes in ordered:
        cursor = _align_up(cursor, TENSOR_ALIGN)
        entries.append((name, dtype, shape, cursor, cursor + nbytes))
        cursor += nbytes
        total_payload += nbytes

    header_bytes = _build_aligned_header(entries)
    assert len(header_bytes) % HEADER_ALIGN == 0
    for _name, _dt, _sh, st, _en in entries:
        assert (len(header_bytes) + st) % TENSOR_ALIGN == 0

    dst_path = dst_dir / shard_name
    tmp_path = dst_dir / f".{shard_name}.tmp.{os.getpid()}"
    sha = hashlib.sha256()
    open_fds: dict[str, int] = {}
    zero_pad = b"\x00" * TENSOR_ALIGN
    try:
        out_fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            if nocache:
                _set_nocache(out_fd)
            os.write(out_fd, header_bytes)
            sha.update(header_bytes)
            written_data = 0
            for (name, src_file, _dtype, _shape, src_abs_start, nbytes), (
                _en_name,
                _en_dt,
                _en_sh,
                dst_start,
                dst_end,
            ) in zip(ordered, entries):
                assert name == _en_name and (dst_end - dst_start) == nbytes
                if dst_start > written_data:
                    pad_len = dst_start - written_data
                    pad_bytes = zero_pad[:pad_len]
                    os.write(out_fd, pad_bytes)
                    sha.update(pad_bytes)
                    written_data += pad_len
                if src_file not in open_fds:
                    sfd = os.open(src_dir / src_file, os.O_RDONLY)
                    if nocache:
                        _set_nocache(sfd)
                    open_fds[src_file] = sfd
                sfd = open_fds[src_file]
                rem = nbytes
                off = src_abs_start
                while rem > 0:
                    take = min(rem, COPY_CHUNK)
                    buf = os.pread(sfd, take, off)
                    if len(buf) != take:
                        raise IOError(
                            f"Short read from {src_file} at {off} for {name}: got {len(buf)}/{take}"
                        )
                    os.write(out_fd, buf)
                    sha.update(buf)
                    off += take
                    rem -= take
                    written_data += take
            # Pad end of shard to 16 KiB page boundary so mmap length matches file size cleanly.
            total_size = len(header_bytes) + written_data
            tail_pad = _align_up(total_size, HEADER_ALIGN) - total_size
            if tail_pad:
                tail = b"\x00" * tail_pad
                os.write(out_fd, tail)
                sha.update(tail)
                total_size += tail_pad
            os.fsync(out_fd)
        finally:
            os.close(out_fd)
        os.replace(tmp_path, dst_path)
    finally:
        for sfd in open_fds.values():
            os.close(sfd)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return {
        "file": shard_name,
        "tensors": len(ordered),
        "payload_bytes": total_payload,
        "file_bytes": total_size,
        "header_bytes": len(header_bytes),
        "sha256": sha.hexdigest(),
    }


def repack_checkpoint(
    src_dir: Path,
    dst_dir: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    nocache: bool = True,
    force: bool = False,
    expected_source_index_sha256: str | None = INDEX_SHA256,
) -> dict:
    """Repack the verified official Flash-1017 checkpoint into per-consumer aligned shards."""
    src_dir = Path(src_dir).resolve()
    dst_dir = Path(dst_dir).resolve()
    if src_dir == dst_dir:
        raise ValueError("Source and destination directories must differ")

    index_path = src_dir / "model.safetensors.index.json"
    raw_index = index_path.read_bytes()
    source_sha = hashlib.sha256(raw_index).hexdigest()
    if expected_source_index_sha256 and source_sha != expected_source_index_sha256:
        raise ValueError(
            f"Source {src_dir} index SHA-256 does not match pinned official Flash-1017 checkpoint"
        )
    src_index = json.loads(raw_index)
    weight_map: dict[str, str] = src_index["weight_map"]

    dst_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dst_dir / MANIFEST_NAME
    if manifest_path.is_file() and not force:
        verify_repack(dst_dir, expected_source_index_sha256=expected_source_index_sha256)
        print(f"Repacked checkpoint already verified at {dst_dir}", flush=True)
        return json.loads(manifest_path.read_text())

    t0 = time.monotonic()
    # Read headers of all referenced non-Engram source shards once.
    grouped: dict[str, list[tuple[str, str, str, list[int], int, int]]] = defaultdict(list)
    new_weight_map: dict[str, str] = {}
    skipped_engram: list[str] = []
    needed_src_files = sorted(
        {
            src_file
            for name, src_file in weight_map.items()
            if consumer_shard(name) is not None
        }
    )
    src_headers: dict[str, tuple[int, dict]] = {}
    for sf in needed_src_files:
        p = src_dir / sf
        if not p.is_file():
            raise FileNotFoundError(f"Missing source shard {p}")
        src_headers[sf] = _read_safetensors_header(p)

    for name, src_file in weight_map.items():
        target = consumer_shard(name)
        if target is None:
            skipped_engram.append(name)
            continue
        data_base, hdr = src_headers[src_file]
        info = hdr[name]
        st, en = info["data_offsets"]
        grouped[target].append(
            (name, src_file, info["dtype"], info["shape"], data_base + st, en - st)
        )
        new_weight_map[name] = target

    # Copy metadata files & directories (encoding/, inference/).
    for rel in METADATA_FILES:
        src_f = src_dir / rel
        if src_f.is_file():
            shutil.copy2(src_f, dst_dir / rel)
    for rel_dir in METADATA_DIRS:
        src_d = src_dir / rel_dir
        dst_d = dst_dir / rel_dir
        if src_d.is_dir():
            if dst_d.exists():
                shutil.rmtree(dst_d)
            shutil.copytree(src_d, dst_d)

    # Stream-write all per-consumer aligned shards in parallel.
    shard_names = sorted(grouped.keys())
    print(
        f"Repacking {len(new_weight_map):,} tensors across {len(shard_names)} aligned shards "
        f"(skipping {len(skipped_engram)} raw Engram tensors) with {workers} workers...",
        flush=True,
    )
    shard_records: list[dict] = []
    completed = 0
    total_written = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                _write_one_shard,
                src_dir,
                dst_dir,
                sname,
                grouped[sname],
                nocache=nocache,
            ): sname
            for sname in shard_names
        }
        for fut in as_completed(futs):
            rec = fut.result()
            shard_records.append(rec)
            completed += 1
            total_written += rec["file_bytes"]
            if completed % 10 == 0 or completed == len(shard_names):
                elapsed = time.monotonic() - t0
                rate = (total_written / (1024**3)) / max(elapsed, 1e-3)
                print(
                    f"  [{completed:2d}/{len(shard_names)}] {total_written / (1024**3):.1f} GiB written "
                    f"({rate:.2f} GiB/s) elapsed={elapsed:.1f}s",
                    flush=True,
                )

    shard_records.sort(key=lambda r: r["file"])
    total_payload = sum(r["payload_bytes"] for r in shard_records)
    total_file_bytes = sum(r["file_bytes"] for r in shard_records)

    new_index = {
        "metadata": {
            "total_size": total_payload,
            "repack_schema": REPACK_SCHEMA,
            "source_index_sha256": INDEX_SHA256,
            "tensor_align": TENSOR_ALIGN,
            "header_align": HEADER_ALIGN,
        },
        "weight_map": dict(sorted(new_weight_map.items())),
    }
    index_bytes = (json.dumps(new_index, indent=2) + "\n").encode("utf-8")
    _atomic_write(dst_dir / "model.safetensors.index.json", index_bytes)
    repacked_index_sha = hashlib.sha256(index_bytes).hexdigest()

    manifest = {
        "schema": REPACK_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_index_sha256": source_sha,
        "repacked_index_sha256": repacked_index_sha,
        "tensor_align": TENSOR_ALIGN,
        "header_align": HEADER_ALIGN,
        "tensor_count": len(new_weight_map),
        "shard_count": len(shard_records),
        "skipped_engram_tensors": sorted(skipped_engram),
        "total_payload_bytes": total_payload,
        "total_file_bytes": total_file_bytes,
        "shards": shard_records,
    }
    _atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    elapsed = time.monotonic() - t0
    print(
        f"Repack complete in {elapsed:.1f}s: {len(shard_records)} shards, "
        f"{total_file_bytes / (1024**3):.2f} GiB on disk ({total_payload / (1024**3):.2f} GiB payload) -> {dst_dir}",
        flush=True,
    )
    return manifest


def _atomic_write(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def verify_repack(
    dst_dir: Path,
    *,
    check_sha256: bool = False,
    expected_source_index_sha256: str | None = INDEX_SHA256,
) -> dict:
    """Verify a repacked checkpoint directory's manifest, index, shard sizes, and alignment."""
    dst_dir = Path(dst_dir)
    manifest_path = dst_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != REPACK_SCHEMA:
        raise ValueError(f"Unsupported repack schema in {manifest_path}")
    if (
        expected_source_index_sha256
        and manifest.get("source_index_sha256") != expected_source_index_sha256
    ):
        raise ValueError(f"Repack in {dst_dir} does not derive from pinned Flash-1017 index")
    raw_index = (dst_dir / "model.safetensors.index.json").read_bytes()
    if hashlib.sha256(raw_index).hexdigest() != manifest["repacked_index_sha256"]:
        raise ValueError(f"Repacked index SHA-256 mismatch in {dst_dir}")
    for rec in manifest["shards"]:
        p = dst_dir / rec["file"]
        if not p.is_file():
            raise FileNotFoundError(f"Missing repacked shard {p}")
        if p.stat().st_size != rec["file_bytes"]:
            raise ValueError(f"Size mismatch for repacked shard {p}")
        data_base, hdr = _read_safetensors_header(p)
        if data_base != rec["header_bytes"] or data_base % HEADER_ALIGN != 0:
            raise ValueError(f"Header alignment mismatch in {p}")
        if len(hdr) != rec["tensors"]:
            raise ValueError(f"Tensor count mismatch in {p}")
        for name, info in hdr.items():
            st, _en = info["data_offsets"]
            if (data_base + st) % TENSOR_ALIGN != 0:
                raise ValueError(f"Unaligned tensor {name} in {p}")
        if check_sha256:
            with p.open("rb") as f:
                got = hashlib.file_digest(f, "sha256").hexdigest()
            if got != rec["sha256"]:
                raise ValueError(f"SHA-256 mismatch for repacked shard {p}")
    return manifest


def verify_against_source(src_dir: Path, dst_dir: Path, *, sample_tensors: int = 256) -> dict:
    """Spot-check (or full-check when sample_tensors==0) byte-for-byte identity vs official source."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    src_wm = json.loads((src_dir / "model.safetensors.index.json").read_text())["weight_map"]
    dst_wm = json.loads((dst_dir / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(dst_wm.keys())
    if sample_tensors and len(names) > sample_tensors:
        step = max(1, len(names) // sample_tensors)
        picked = names[::step]
        # Always include critical dense + head + mtp + vision tensors
        for must in (
            "embed.weight",
            "head.weight",
            "norm.weight",
            "layers.0.attn.wo_a.weight",
            "layers.0.ffn.experts.0.w1.weight",
            "layers.1.engram.wkv.weight",
            "layers.14.engram.wkv.weight",
            "mtp.0.ffn.experts.0.w1.weight",
            "vision.norm.weight",
        ):
            if must in dst_wm and must not in picked:
                picked.append(must)
        names = picked

    src_hdr_cache: dict[str, tuple[int, dict]] = {}
    dst_hdr_cache: dict[str, tuple[int, dict]] = {}
    checked_bytes = 0
    for name in names:
        sf = src_wm[name]
        df = dst_wm[name]
        if sf not in src_hdr_cache:
            src_hdr_cache[sf] = _read_safetensors_header(src_dir / sf)
        if df not in dst_hdr_cache:
            dst_hdr_cache[df] = _read_safetensors_header(dst_dir / df)
        sb, sh = src_hdr_cache[sf]
        db, dh = dst_hdr_cache[df]
        si = sh[name]
        di = dh[name]
        if si["dtype"] != di["dtype"] or si["shape"] != di["shape"]:
            raise ValueError(f"Metadata mismatch on {name}: {si} vs {di}")
        s0, s1 = si["data_offsets"]
        d0, d1 = di["data_offsets"]
        nbytes = s1 - s0
        if (d1 - d0) != nbytes:
            raise ValueError(f"Length mismatch on {name}")
        with (src_dir / sf).open("rb") as f1, (dst_dir / df).open("rb") as f2:
            f1.seek(sb + s0)
            f2.seek(db + d0)
            rem = nbytes
            while rem > 0:
                take = min(rem, 4 * 1024 * 1024)
                if f1.read(take) != f2.read(take):
                    raise ValueError(f"Byte mismatch on {name}")
                rem -= take
        checked_bytes += nbytes
    return {"checked_tensors": len(names), "checked_bytes": checked_bytes}
