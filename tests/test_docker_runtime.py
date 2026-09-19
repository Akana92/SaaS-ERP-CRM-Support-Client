"""CPU tests for container entrypoint and isolation boundaries; no model loading."""
import inspect
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_demo


class DockerRuntimeTests(unittest.TestCase):
    def test_posix_sigterm_replay_returns_and_restores_original_handler(self):
        server = MagicMock()
        previous = object()
        with patch.object(live_demo.os, "name", "posix"), patch.object(live_demo.signal, "signal", return_value=previous) as install:
            def replay():
                install.call_args_list[0].args[1](signal.SIGTERM, None)
            server.run.side_effect = replay
            live_demo.run_server(server)
            self.assertTrue(server.should_exit)
            self.assertEqual(install.call_count, 2)
            install.assert_called_with(signal.SIGTERM, previous)

    def test_server_failure_propagates_and_restores_handler(self):
        server = MagicMock()
        server.run.side_effect = RuntimeError("server failed")
        previous = object()
        with patch.object(live_demo.os, "name", "posix"), patch.object(live_demo.signal, "signal", return_value=previous) as install:
            with self.assertRaisesRegex(RuntimeError, "server failed"):
                live_demo.run_server(server)
            install.assert_called_with(signal.SIGTERM, previous)

    def test_windows_keeps_uvicorn_signal_handling(self):
        server = MagicMock()
        with patch.object(live_demo.os, "name", "nt"), patch.object(live_demo.signal, "signal") as install:
            live_demo.run_server(server)
            server.run.assert_called_once_with()
            install.assert_not_called()

    @unittest.skipUnless(sys.platform == "linux", "real Linux SIGTERM regression")
    def test_real_uvicorn_sigterm_completes_launcher_cleanup(self):
        script = textwrap.dedent('''
            from pathlib import Path
            import sys
            import uvicorn
            from live_demo import run_server

            output = Path(sys.argv[1])
            async def app(scope, receive, send):
                assert scope["type"] == "lifespan"
                await receive()
                await send({"type": "lifespan.startup.complete"})
                await receive()
                (output / "asgi-stopped").touch()
                await send({"type": "lifespan.shutdown.complete"})

            class ReadyServer(uvicorn.Server):
                async def startup(self, sockets=None):
                    await super().startup(sockets=sockets)
                    (output / "ready").touch()

            server = ReadyServer(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
            try:
                run_server(server)
                (output / "stopped").touch()
            finally:
                (output / "runtime-closed").touch()
        ''')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "scripts")
            with subprocess.Popen([sys.executable, "-c", script, directory], cwd=ROOT,
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True) as process:
                try:
                    deadline = time.monotonic() + 20
                    while not (output / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.05)
                    self.assertTrue((output / "ready").exists(), "Uvicorn did not become ready")
                    process.send_signal(signal.SIGTERM)
                    stdout, stderr = process.communicate(timeout=20)
                    self.assertEqual(process.returncode, 0, stdout + stderr)
                    for marker in ("asgi-stopped", "stopped", "runtime-closed"):
                        self.assertTrue((output / marker).exists(), marker)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()

    def test_loopback_remains_default(self):
        for function in (live_demo.serve, live_demo.start, live_demo.assert_port_free):
            self.assertEqual(inspect.signature(function).parameters["host"].default, "127.0.0.1")

    def test_probe_uses_requested_bind_host(self):
        with patch.object(live_demo.socket, "socket") as factory:
            live_demo.assert_port_free(7860, "0.0.0.0")
            factory.return_value.__enter__.return_value.bind.assert_called_once_with(("0.0.0.0", 7860))

    def test_cli_propagates_host_and_preserves_default(self):
        for flags, expected in (([], "127.0.0.1"), (["--host", "0.0.0.0"], "0.0.0.0")):
            with patch.object(sys, "argv", ["live_demo.py", "serve", "--run-id", "test", *flags]), patch.object(live_demo, "serve") as serve:
                live_demo.main()
                self.assertEqual(serve.call_args.args[-1], expected)

    def test_start_forwards_host_to_child(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(live_demo, "OUTPUT", Path(directory)), patch.object(live_demo, "assert_port_free") as probe, patch.object(live_demo.subprocess, "Popen") as spawn:
            spawn.return_value.pid = 123
            live_demo.start(7860, host="0.0.0.0")
            command = spawn.call_args.args[0]
            self.assertEqual(command[command.index("--host") + 1], "0.0.0.0")
            probe.assert_called_once_with(7860, "0.0.0.0")

    def test_runtime_stays_offline_and_keeps_adapter_integrity_check(self):
        env = live_demo.offline_environment()
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(live_demo.serving_info("quality-f")["adapter_model_sha256"], "6efc00de7bcecc24872b87de797abd4d9d43ea10e679b23d3b4c13ecb7f7109b")

    def test_compose_runtime_boundary(self):
        # No YAML dependency needed for these explicit security contract checks.
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        app, assets = compose.split("  assets:", 1)
        self.assertIn('"127.0.0.1:7860:7860"', app)
        self.assertEqual(app.count("read_only: true"), 2)
        self.assertEqual(app.count("create_host_path: false"), 2)
        self.assertIn("stop_grace_period: 10m", app)
        self.assertIn("driver: nvidia", app)
        self.assertNotIn("hf_token", app)
        self.assertIn("profiles: [setup]", assets)
        self.assertIn('HF_HUB_OFFLINE: "0"', assets)
        self.assertNotIn("driver: nvidia", assets)
        self.assertIn("7860/health", app)

    def test_build_context_is_allowlisted(self):
        lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        self.assertEqual(next(line for line in lines if line and not line.startswith("#")), "**")
        self.assertNotIn("!models/", lines)
        self.assertNotIn("!.secrets/", lines)
        self.assertNotIn("!data/**", lines)
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("COPY . ", dockerfile)
        self.assertNotIn("HF_TOKEN", dockerfile)


if __name__ == "__main__":
    unittest.main()
