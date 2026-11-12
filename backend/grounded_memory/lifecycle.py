"""Keep helper processes from outliving the server that started them."""

import os
import threading
import time


def exit_with_parent(parent_pid, interval=1.0):
    """Pool initializer: leave as soon as the parent is gone (crash, SIGKILL or a closed terminal).

    A spawned worker is otherwise re-parented to launchd and keeps its model memory and its
    multiprocessing resource tracker alive indefinitely.
    """

    def watch():
        while os.getppid() == parent_pid:
            time.sleep(interval)
        os._exit(0)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()
