"""Process start token and process-group liveness, the canonical process-identity probes.

``run_evidence`` imports this module, and ``python_programs`` embeds its text
verbatim in the ``managed_scheduler.process_launch`` wrapper, so the token the
wrapper records at launch and the token observation later compares come from
one source. Embedding constrains it: stdlib imports only, no ``agent_tools``
imports and no ``from __future__`` import, since it is concatenated mid-program.
"""

import os
import subprocess
import sys


def start_token(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as stat_file:
            stat_text = stat_file.read()
    except OSError:
        if sys.platform == "darwin":
            # macOS has no /proc; libproc exposes the kernel start timestamp without a fragile ps parse.
            import ctypes

            class ProcBsdInfo(ctypes.Structure):
                _fields_ = [
                    ("flags", ctypes.c_uint32),
                    ("status", ctypes.c_uint32),
                    ("xstatus", ctypes.c_uint32),
                    ("pid", ctypes.c_uint32),
                    ("ppid", ctypes.c_uint32),
                    ("uid", ctypes.c_uint32),
                    ("gid", ctypes.c_uint32),
                    ("ruid", ctypes.c_uint32),
                    ("rgid", ctypes.c_uint32),
                    ("svuid", ctypes.c_uint32),
                    ("svgid", ctypes.c_uint32),
                    ("rfu", ctypes.c_uint32),
                    ("comm", ctypes.c_char * 16),
                    ("name", ctypes.c_char * 32),
                    ("nfiles", ctypes.c_uint32),
                    ("pgid", ctypes.c_uint32),
                    ("pjobc", ctypes.c_uint32),
                    ("e_tdev", ctypes.c_uint32),
                    ("e_tpgid", ctypes.c_uint32),
                    ("nice", ctypes.c_int32),
                    ("start_sec", ctypes.c_uint64),
                    ("start_usec", ctypes.c_uint64),
                ]

            info = ProcBsdInfo()
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
            size = libproc.proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
            if size == ctypes.sizeof(info) and info.pid == pid:
                return f"darwin:{info.start_sec}:{info.start_usec}"
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return "ps:" + result.stdout.strip()
    return "proc:" + stat_text.rsplit(")", 1)[1].split()[19]


def process_group_running(pgid: int, proc_root: str = "/proc") -> bool | None:
    """Whether ``pgid`` still has a non-zombie member; None when the process table cannot settle it.

    Stat reads replace undecodable bytes: only the numeric fields after the
    process name are parsed, and a name may hold arbitrary bytes."""
    leader_uncertain = False
    leader_stat: str | None
    try:
        with open(os.path.join(proc_root, str(pgid), "stat"), encoding="utf-8", errors="replace") as stat_file:
            leader_stat = stat_file.read()
    except (FileNotFoundError, ProcessLookupError):
        # ESRCH, like ENOENT, means the leader exited while its stat was read.
        leader_stat = None
    except OSError:
        leader_stat = None
        leader_uncertain = True
    if leader_stat:
        fields = leader_stat.rsplit(")", 1)[-1].split()
        try:
            leader_pgid = int(fields[2])
        except (IndexError, ValueError):
            leader_uncertain = True
        else:
            if leader_pgid == pgid and fields[0] not in {"Z", "X"}:
                return True

    try:
        entries = os.listdir(proc_root)
    except FileNotFoundError:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    except OSError:
        return None

    uncertain = leader_uncertain
    for entry in entries:
        if not entry.isdigit() or entry == str(pgid):
            continue
        try:
            with open(os.path.join(proc_root, entry, "stat"), encoding="utf-8", errors="replace") as stat_file:
                stat_text = stat_file.read()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            uncertain = True
            continue
        fields = stat_text.rsplit(")", 1)[-1].split()
        if len(fields) < 3:
            uncertain = True
            continue
        try:
            process_pgid = int(fields[2])
        except ValueError:
            uncertain = True
            continue
        if process_pgid == pgid and fields[0] not in {"Z", "X"}:
            return True
    return None if uncertain else False
