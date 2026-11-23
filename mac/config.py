from dataclasses import dataclass
import os
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def recipe_path(value):
    return (ROOT / Path(value).expanduser()).resolve()


@dataclass(frozen=True)
class Settings:
    repo: str
    revision: str
    name: str
    model: Path
    engram: Path
    assets: Path
    source: Path
    state: Path
    host: str
    port: int
    context: int
    parallel: int
    drafts: bool
    prompt_cache_gib: int
    engine_repo: str
    engine_tag: str
    engine_sha: str
    engram_cache_rows: int = 65536
    metal4: str = "1"
    markov_topk: int = 64

    @property
    def bin(self):
        return self.source / "zig-out/bin"

    @property
    def serve_binary(self):
        return self.bin / "tensorfold-dsv41"

    @property
    def url(self):
        host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        return f"http://{host}:{self.port}"

    def command(self):
        args = [
            str(self.serve_binary),
            "serve",
            str(self.model),
            "--name",
            self.name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--context",
            str(self.context),
            "--parallel",
            str(self.parallel),
            "--prompt-cache-gib",
            str(self.prompt_cache_gib),
            "--no-update-check",
        ]
        if not self.drafts:
            args.append("--no-drafts")
        return args

    def launch_settings(self, environ=None):
        environ = os.environ if environ is None else environ
        pinned = {
            "TF_EXPERIMENTAL": "deepseek",
            "TF_METAL4": self.metal4,
            "TF_DSV41_ENGRAM_DIR": str(self.engram),
            "TF_DSV41_ASSETS": str(self.assets),
            "TF_DSV41_ENGRAM_CACHE": str(self.engram_cache_rows),
            "TF_DSV41_FP64_BF16": "1",
            "TF_DSV41_M5_FAST_DECODE": "1",
            "TF_DSV41_MARKOV_TOPK": str(self.markov_topk),
        }
        for key, value in environ.items():
            if key.startswith(("TF_", "TFDS41_")) and pinned.get(key) != value:
                raise ValueError(
                    f"Unset {key} to reproduce the verified recipe settings"
                )
        return pinned

    def server_env(self, environ=None):
        environ = os.environ if environ is None else environ
        return {**environ, **self.launch_settings(environ)}


LOCAL_KEYS = {
    "model": {"model_dir", "engram_dir", "assets_dir"},
    "local": {"host", "port", "state_dir"},
    "serve": {"drafts", "parallel", "context", "prompt_cache"},
}


def validate_serving(context, parallel):
    if not 1 <= context <= 1_048_576:
        raise ValueError("context must be between 1 and 1048576 tokens")
    if not 1 <= parallel <= 16:
        raise ValueError("parallel must be between 1 and 16")


def load():
    with (ROOT / "recipe.toml").open("rb") as f:
        config = tomllib.load(f)
    overrides = ROOT / "local.toml"
    if overrides.is_file():
        with overrides.open("rb") as f:
            sections = tomllib.load(f)
        for section, values in sections.items():
            if section not in LOCAL_KEYS or not isinstance(values, dict):
                raise ValueError(
                    f"Unsupported local.toml section [{section}]. See local.example.toml."
                )
            for key, value in values.items():
                if key not in LOCAL_KEYS[section]:
                    raise ValueError(
                        f"Unsupported local.toml setting {section}.{key}. See local.example.toml."
                    )
                if type(value) is not type(config[section][key]):
                    raise ValueError(
                        f"Incorrect type for local.toml setting {section}.{key}"
                    )
                config[section][key] = value
    model, engine, local, serve = (
        config[key] for key in ("model", "engine", "local", "serve")
    )
    if not 1 <= local["port"] <= 65535:
        raise ValueError("port must be between 1 and 65535")
    validate_serving(serve["context"], serve["parallel"])
    return Settings(
        repo=model["source_repo"],
        revision=model["source_revision"],
        name=model["served_name"],
        model=recipe_path(model["model_dir"]),
        engram=recipe_path(model["engram_dir"]),
        assets=recipe_path(model["assets_dir"]),
        source=recipe_path(engine["tensorfold_src"]),
        state=recipe_path(local["state_dir"]),
        host=local["host"],
        port=local["port"],
        context=serve["context"],
        parallel=serve["parallel"],
        drafts=serve["drafts"],
        prompt_cache_gib=4 if serve["prompt_cache"] else 0,
        engine_repo=engine["tensorfold_repo"],
        engine_tag=engine["tensorfold_tag"],
        engine_sha=engine["tensorfold_sha"],
    )
