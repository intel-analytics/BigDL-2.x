import os
import sys
import signal
import psutil
import logging
logging.basicConfig(filename='daemon.log', level=logging.INFO)


def stop(pgid):
    logging.info(f"Stopping pgid {pgid} by ray_daemon.")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        logging.error("Cannot kill pgid: {}".format(pgid))


def manager():
    pid_to_watch = int(sys.argv[1])
    pgid_to_kill = int(sys.argv[2])
    import time
    while psutil.pid_exists(pid_to_watch):
        time.sleep(1)
    stop(pgid_to_kill)


if __name__ == "__main__":
    manager()
