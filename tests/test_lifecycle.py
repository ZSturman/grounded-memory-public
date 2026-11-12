import os
import subprocess
import sys
import time

# A stand-in server: start the perception pool the way api.worker does, report the worker's PID,
# then die without any cleanup, as the 2026-11-02 Numba abort did.
CRASHING_SERVER = """
import multiprocessing, os
from concurrent.futures import ProcessPoolExecutor
from grounded_memory.lifecycle import exit_with_parent

if __name__ == "__main__":
    pool = ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=exit_with_parent,
        initargs=(os.getpid(), 0.1),
    )
    print(pool.submit(os.getpid).result(), flush=True)
    os._exit(1)
"""


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_worker_exits_when_server_crashes():
    server = subprocess.run(
        [sys.executable, "-c", CRASHING_SERVER], capture_output=True, text=True, timeout=60
    )
    worker = int(server.stdout.split()[0])
    deadline = time.time() + 5
    while alive(worker) and time.time() < deadline:
        time.sleep(0.1)
    if alive(worker):
        os.kill(worker, 9)
        raise AssertionError("perception worker outlived its crashed server")
