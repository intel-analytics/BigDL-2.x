import os
import sys
import signal
import psutil
import logging
logging.basicConfig(filename='daemon.log', level=logging.INFO)

def is_jvm_alive(spark_executor_pid):
    return psutil.pid_exists(spark_executor_pid)


def stop_ray(pgid):
    logging.info(f"Stopping pgid {pgid} by ray_daemon.")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        logging.error("WARNING: cannot kill pgid: {}".format(pgid))


def manager():
    pgid = int(sys.argv[1])
    spark_executor_pid = int(sys.argv[2])
    import time
    while is_jvm_alive(spark_executor_pid):
        time.sleep(1)
    stop_ray(pgid)


if __name__ == "__main__":
    manager()
