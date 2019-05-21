import os
import signal


def gen_shutdown_per_node(pgids):
    def _shutdown_per_node(iter):
        print("shutting down pgid: {}".format(pgids))
        for pgid in pgids:
            print("killing {}".format(pgid))
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                print("WARNING: cannot find pgid: {}".format(pgid))

    return _shutdown_per_node

def is_local(sc):
    master = sc._conf.get("spark.master")
    return master == "local" or master.startswith("local[")


