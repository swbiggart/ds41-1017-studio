"""Exercise the Mac Studio serving recipe CLI and config without model files or Metal hardware."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mac import __main__ as cli
from mac import config, engine, prepare


DUMMY_MANIFEST = {
    "upstream_repo": "https://github.com/ashhart/TensorFold.git",
    "upstream_tag": "v1.3.3",
    "upstream_commit": "1e4a32cb5e80f2908e4642f9e7ffe71ae63792cc",
    "source_tree": "0" * 40,
    "patches": [],
}


class LauncherTests(unittest.TestCase):
    def setUp(self):
        work = TemporaryDirectory()
        self.addCleanup(work.cleanup)
        root = Path(work.name).resolve()
        (root / "recipe.toml").write_bytes((ROOT / "recipe.toml").read_bytes())
        self.local = root / "local.toml"
        self.enterContext(patch.object(config, "ROOT", root))
        self.enterContext(patch.object(engine, "ROOT", root))

    def run_command(self, *words):
        out = io.StringIO()
        with (
            patch.object(sys, "argv", ["mac", *words]),
            redirect_stdout(out),
            redirect_stderr(out),
        ):
            cli.main()
        return out.getvalue()

    def test_defaults_and_pinned_revisions(self):
        base = config.load()
        self.assertEqual(
            (base.parallel, base.context, base.drafts, base.prompt_cache_gib),
            (1, 139264, True, 4),
        )
        self.assertEqual(base.repo, "deepseek-ai/DeepSeek-V4.1-Flash-1017")
        self.assertEqual(base.revision, "08c8657f02291e11b6f6b63e0e16014f8e748518")
        self.assertEqual(base.engine_tag, "v1.3.3")
        self.assertEqual(base.engine_sha, "1e4a32cb5e80f2908e4642f9e7ffe71ae63792cc")
        self.assertEqual(base.model, self.local.parent / ".local/checkpoint")

    def test_build_invokes_engine_build(self):
        with patch.object(engine, "build") as build:
            self.run_command("build")
        build.assert_called_once()

    def test_cli_overrides_local_options(self):
        self.local.write_text("[serve]\ndrafts=false\nparallel=2\n")
        with patch.object(engine, "load_manifest", return_value=DUMMY_MANIFEST):
            current = json.loads(
                self.run_command("--drafts", "--context", "4096", "settings")
            )
        self.assertEqual(current["engine"]["tag"], "v1.3.3")
        self.assertEqual(
            current["serve"],
            {
                "context": 4096,
                "parallel": 2,
                "drafts": True,
                "prompt_cache": True,
            },
        )
        self.assertNotIn("--no-drafts", current["native_argv"])
        self.assertEqual(current["native_env"]["TF_EXPERIMENTAL"], "deepseek")
        self.assertEqual(current["native_env"]["TF_METAL4"], "1")
        self.assertEqual(current["native_env"]["TF_DSV41_ENGRAM_CACHE"], "65536")
        self.assertEqual(current["native_env"]["TF_DSV41_FP64_BF16"], "1")
        self.assertEqual(current["native_env"]["TF_DSV41_M5_FAST_DECODE"], "1")
        self.assertEqual(current["native_env"]["TF_DSV41_MARKOV_TOPK"], "64")

    def test_prompt_cache_toggle(self):
        for local, flag, expected in (
            ("true", "--no-prompt-cache", 0),
            ("false", "--prompt-cache", 4),
        ):
            with self.subTest(local=local):
                self.local.write_text(f"[serve]\nprompt_cache={local}\n")
                self.assertEqual(
                    config.load().prompt_cache_gib, 4 if local == "true" else 0
                )
                with patch.object(engine, "load_manifest", return_value=DUMMY_MANIFEST):
                    info = json.loads(self.run_command(flag, "settings"))
                self.assertEqual(info["serve"]["prompt_cache"], expected > 0)
                argv = info["native_argv"]
                self.assertEqual(
                    argv[argv.index("--prompt-cache-gib") + 1], str(expected)
                )
        self.local.write_text("[serve]\nprompt_cache=4\n")
        with self.assertRaisesRegex(ValueError, "Incorrect type"):
            config.load()

    def test_serving_flags_rejected_on_non_serving_commands(self):
        with self.assertRaises(SystemExit):
            self.run_command("--no-prompt-cache", "build")
        with self.assertRaises(SystemExit):
            self.run_command("--no-drafts", "down")

    def test_unknown_options_and_local_keys_rejected(self):
        for args in (
            ["--target", "upstream"],
            ["--format", "mlx-lossless"],
            ["--tree-branches", "2"],
        ):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.run_command(*args, "settings")
        for legacy in (
            '[engine]\ntarget="optimized"\n',
            '[model]\nweight_format="mlx-lossless"\n',
            "[verifier]\nfp8_groups=16\n",
        ):
            self.local.write_text(legacy)
            with self.subTest(local=legacy), self.assertRaisesRegex(
                ValueError, "local.toml"
            ):
                config.load()

    def test_conflicting_ambient_tf_env_rejected(self):
        base = config.load()
        with self.assertRaisesRegex(ValueError, "TF_METAL4"):
            base.launch_settings({"TF_METAL4": "0"})
        with self.assertRaisesRegex(ValueError, "TF_DSV41_FOO"):
            base.launch_settings({"TF_DSV41_FOO": "1"})

    def test_wrong_weight_index_fails_before_loading(self):
        self.local.with_name("model.safetensors.index.json").write_text(
            '{"weight_map":{}}'
        )
        with self.assertRaisesRegex(ValueError, "pinned original checkpoint"):
            prepare.check_shards(replace(config.load(), model=self.local.parent))

    def test_engine_verify_patches_and_build_receipt(self):
        from hashlib import sha256

        root = self.local.parent
        patches_dir = root / "patches"
        patches_dir.mkdir()
        patch_file = patches_dir / "01-test.patch"
        patch_bytes = b"diff --git a/x b/x\n"
        patch_file.write_bytes(patch_bytes)
        manifest = {
            **DUMMY_MANIFEST,
            "patches": [
                {"file": "01-test.patch", "sha256": sha256(patch_bytes).hexdigest()}
            ],
        }
        (patches_dir / "manifest.json").write_text(json.dumps(manifest))
        with patch.object(engine, "MANIFEST_PATH", patches_dir / "manifest.json"):
            engine.verify_patches(engine.load_manifest())
            s = config.load()
            bin_dir = s.source / "zig-out/bin"
            bin_dir.mkdir(parents=True)
            exe = bin_dir / "tensorfold-dsv41"
            exe.write_bytes(b"fake-binary")
            (s.source / "zig-out/build-receipt.json").write_text(
                json.dumps(
                    {
                        "upstream_commit": manifest["upstream_commit"],
                        "source_tree": manifest["source_tree"],
                        "binaries": {
                            "tensorfold-dsv41": sha256(b"fake-binary").hexdigest()
                        },
                    }
                )
            )
            receipt = engine.verify_build(s)
            self.assertEqual(receipt["source_tree"], manifest["source_tree"])
            exe.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "differs from build-receipt"):
                engine.verify_build(s)


if __name__ == "__main__":
    unittest.main()
