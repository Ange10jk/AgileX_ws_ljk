"""Private model-process entry point with no hardware or recording dependencies."""
from __future__ import annotations

import sys
import traceback

from ..ipc import read_message, write_message
from .adapter import PolicyRunner
from .interfaces import PolicyError
from .lcgen_http import LCGenHTTP


def main():
    incoming, outgoing = sys.stdin.buffer, sys.stdout.buffer
    # Libraries sometimes print; stdout belongs exclusively to framed IPC.
    sys.stdout = sys.stderr
    runner = None
    while True:
        try:
            message = read_message(incoming)
        except EOFError:
            break
        kind = message["kind"]
        response = {"kind": kind}
        if kind in ("infer", "observe"):
            response.update(generation=message["generation"], seq=message.get("seq"))
        try:
            if kind == "load":
                config = message["config"]
                backend = LCGenHTTP if config["policy"].get("backend") == "lcgen_http" else PolicyRunner
                runner = backend(config).load()
                response = {"kind": "ready", "execution": runner.mode,
                            "action_names": runner.names,
                            "rlt_horizon": runner.rlt.horizon if runner.rlt else None,
                            "episode_id": runner.rlt.episode_id if runner.rlt else 0,
                            "model_contract_sha256": getattr(runner, "contract_sha256", None)}
                response.update(getattr(runner, "capabilities", {}))
            elif runner is None:
                raise PolicyError("Worker is not loaded")
            elif kind == "infer":
                response.update(runner.infer(message["observation"]))
            elif kind == "observe":
                runner.observe(message["observation"])
            elif kind == "reset":
                runner.invalidate()
            elif kind in ("record_step", "boundary", "end_episode", "update", "save"):
                if runner.rlt is None:
                    raise PolicyError("RLT commands require workflow.mode=online_rl and learner=pi05_rlt")
                arguments = {k: v for k, v in message.items() if k != "kind"}
                response["result"] = getattr(runner.rlt, kind)(**arguments)
            else:
                raise PolicyError("Unknown worker command %s" % kind)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response["error"] = "%s: %s" % (type(exc).__name__, exc)
        write_message(outgoing, response)
    if runner is not None:
        runner.close()


if __name__ == "__main__":
    main()
