"""A single visible startup path; never resets user memory."""

import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
command = sys.argv[1] if len(sys.argv) > 1 else "doctor"
if command == "migration-status":
    for item in json.loads((ROOT / "docs/PROJECT_INVENTORY.json").read_text()):
        print(item["name"] + ": " + item["state"])
    sys.exit()
if command == "doctor":
    import urllib.request

    print("Python:", sys.version.split()[0])
    for name in ["node", "npm", "ffmpeg", "ffprobe", "caddy"]:
        print(name + ": " + str(shutil.which(name)))
    try:
        with urllib.request.urlopen(
            os.getenv("GM_LM_URL", "http://127.0.0.1:1234/v1") + "/models", timeout=4
        ) as r:
            print("LM Studio:", ", ".join(x["id"] for x in json.load(r)["data"]))
    except Exception as e:
        print("LM Studio unavailable:", e)
    for module in ["aiortc", "torch", "transformers", "sam2", "mlx_whisper"]:
        import importlib.util

        print(module + ": " + ("installed" if importlib.util.find_spec(module) else "missing"))
    print("Data:", os.getenv("GM_DATA_DIR", str(Path.home() / "Library/Application Support/Grounded Memory")))
    sys.exit()
env = os.environ.copy()
env.setdefault("NEXT_TELEMETRY_DISABLED", "1")
env.setdefault("GM_PAIR_CODE", str(secrets.randbelow(900000) + 100000))
env.setdefault(
    "GM_CA_CERT", str(Path.home() / "Library/Application Support/Caddy/pki/authorities/local/root.crt")
)
print("Pairing code:", env["GM_PAIR_CODE"], flush=True)
children = []


def start(args, **options):
    """Each service leads its own process group so shutdown also reaches npm's next-server."""
    return subprocess.Popen(args, start_new_session=True, **options)


def stop_group(child, sig):
    try:
        os.killpg(child.pid, sig)
    except ProcessLookupError:
        pass


def interrupt(signum, frame):
    raise KeyboardInterrupt  # closing the terminal or SIGTERM gets the same cleanup as Ctrl-C


signal.signal(signal.SIGTERM, interrupt)
signal.signal(signal.SIGHUP, interrupt)
try:
    children.append(
        start(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "grounded_memory.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8100",
            ],
            env=env,
        )
    )
    if command == "phone":
        subprocess.run(["npm", "run", "build"], cwd=ROOT / "web", check=True)
    children.append(
        start(["npm", "run", "start" if command == "phone" else "dev"], cwd=ROOT / "web", env=env)
    )
    print("Mac: http://localhost:3100", flush=True)
    if command == "phone":
        if not shutil.which("caddy"):
            raise RuntimeError("Install phone HTTPS support with: brew install caddy. See docs/PHONE.md.")
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("192.0.2.1", 1))
            ip = probe.getsockname()[0]
        finally:
            probe.close()
        folder = ROOT / ".phone"
        folder.mkdir(exist_ok=True)
        config = folder / "Caddyfile"
        config.write_text(
            "{\n skip_install_trust\n}\nhttps://"
            + ip
            + ":3443 {\n tls internal\n @backend path /api/*\n reverse_proxy @backend 127.0.0.1:8100\n reverse_proxy 127.0.0.1:3100\n}\n"
        )
        children.append(start(["caddy", "run", "--config", str(config)], env=env))
        print("Phone: https://" + ip + ":3443 — certificate instructions: docs/PHONE.md", flush=True)
        # The iPhone app finds this Mac through Bonjour and pins Caddy's root after comparing this fingerprint.
        children.append(
            start(
                [
                    "dns-sd",
                    "-R",
                    "Grounded Memory",
                    "_groundedmemory._tcp",
                    "local",
                    "3443",
                    "url=https://" + ip + ":3443",
                ],
                stdout=subprocess.DEVNULL,
            )
        )
        root = Path(env["GM_CA_CERT"])
        for _ in range(20):
            if root.exists():
                import hashlib
                import ssl

                digest = hashlib.sha256(ssl.PEM_cert_to_DER_cert(root.read_text())).hexdigest().upper()
                print("iPhone app certificate check: " + digest[:8] + "…" + digest[-6:], flush=True)
                break
            time.sleep(0.5)
    while True:
        time.sleep(1)
        failed = [p for p in children if p.poll() is not None]
        if failed:
            raise RuntimeError("A service stopped; inspect its error above.")
except KeyboardInterrupt:
    pass
finally:
    for child in children:
        stop_group(child, signal.SIGTERM)
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        stop_group(child, signal.SIGKILL)
