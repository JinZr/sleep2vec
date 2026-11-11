#!/usr/bin/env python3
"""Opt-in synthetic public-monitor benchmark against fake local Slurm clients."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

STATE_ENV = "SLURM_MONITOR_BENCH_STATE"
TOOLS = ("squeue", "scontrol", "sacct", "sbatch", "scancel", "srun", "nvidia-smi")
DENIED = {"sbatch", "scancel", "srun", "nvidia-smi"}


def record(state, kind, **fields):
    with Path(state["events"]).open("a") as stream:
        stream.write(json.dumps({"kind": kind, **fields}, sort_keys=True) + "\n")


def fake_tool(tool):
    state = json.loads(Path(os.environ[STATE_ENV]).read_text())
    argv = sys.argv[1:]
    if tool in DENIED:
        record(state, "denied", tool=tool, argv=argv)
        raise RuntimeError(f"Benchmark forbids {tool}")
    if tool == "squeue" and "--jobs" not in argv:
        raise RuntimeError("Benchmark forbids an unbounded squeue query")
    requested = argv[argv.index("--jobs") + 1].split(",") if "--jobs" in argv else [argv[-1]]
    # Job ids are unique across synthetic routes, so the first id names the frozen route.
    cluster = next((name for name, route in state["routes"].items() if requested[0] in route["jobs"]), "")
    record(state, "scheduler", tool=tool, cluster=cluster, argv=argv)
    time.sleep(state["latency_ms"] / 1000)
    route = state["routes"].get(cluster)
    if route is None:
        print(f"{tool}: error: Invalid job id specified", file=sys.stderr)
        return 1
    cluster_args = [value for value in argv if value.startswith("--clusters=")]
    expected_cluster_args = [] if route["direct_controller"] else [f"--clusters={cluster}"]
    if cluster_args != expected_cluster_args:
        raise RuntimeError(f"Unexpected frozen route: {cluster} {argv}")
    if tool == "squeue":
        for job_id in requested:
            if job_id not in route["jobs"]:
                print("squeue: error: Invalid job id specified", file=sys.stderr)
                return 1
        if state["scenario"] == "batch-error" and len(requested) > 1:
            job_id = requested[0]
            print(f"{job_id}|PENDING|Resources||{route['jobs'][job_id]}")
            print("squeue: synthetic batch failure after partial stdout", file=sys.stderr)
            return 1
        if state["scenario"] in {"queue", "batch-error"}:
            for job_id in requested:
                print(f"{job_id}|PENDING|Resources||{route['jobs'][job_id]}")
        return 0
    if tool == "scontrol":
        job_id = argv[-1]
        token = route["jobs"][job_id]
        print(
            f"JobId={job_id} JobState=PENDING Reason=Resources NodeList=(null) Comment={token} "
            "ExitCode=0:0 Priority=42 Nice=0 Partition=benchmark Account=fixture QOS=normal "
            "SubmitTime=2026-08-30T00:00:00 StartTime=Unknown TimeLimit=01:00:00"
        )
        return 0
    if tool == "sacct":
        return 0
    raise RuntimeError(f"Unexpected benchmark tool: {tool}")


def make_fakebin(directory, *, interpreter, harness):
    directory.mkdir()
    for tool in TOOLS:
        target = directory / tool
        target.write_text(
            f"#!{interpreter}\nimport runpy, os\n"
            f"os.environ['SLURM_MONITOR_BENCH_TOOL'] = {tool!r}\n"
            f"runpy.run_path({str(harness)!r}, run_name='__main__')\n"
        )
        target.chmod(0o755)


def prepare_fixture(repo, root, count, routes, state):
    sys.path[:0] = [str(repo), str(repo / "tests"), str(repo / "tests" / "agent_tools")]
    from agent_tool_test_helpers import run_execution_preflight_fixture
    import pytest
    from test_agent_tools_hparam_runtime import _write_slurm_plan

    from agent_tools import execution_snapshot, manifests
    from agent_tools.experiment_workspace import read_run_manifest

    state_path = Path(os.environ[STATE_ENV])
    state_path.write_text(json.dumps(state))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(execution_snapshot, "run_execution_command", run_execution_preflight_fixture)
        plan_dir, plan = _write_slurm_plan(
            root,
            run_count=count,
            direct_controller=state["routes"]["benchmark-cluster-0"]["direct_controller"],
        )
    rows = read_run_manifest(root)
    selected_keys = {(run["step_id"], run["run_id"]) for run in plan["runs"]}
    selected = [row for row in rows if (row["step_id"], row["run_id"]) in selected_keys]
    if len(selected) != count:
        raise RuntimeError("Synthetic fixture has unexpected canonical run count")
    for index, row in enumerate(selected):
        cluster = f"benchmark-cluster-{index % routes}"
        route = state["routes"][cluster]
        job_id = str(10000 + index)
        route["jobs"][job_id] = row["scheduler_submit_token"]
        row.update(
            target="local",
            status="queued",
            scheduler_job_id=job_id,
            scheduler_cluster=cluster,
            scheduler_direct_controller=str(route["direct_controller"]).lower(),
            scheduler_raw_state="PENDING",
            scheduler_reason="Resources",
            launched_at="2026-08-30T00:00:00Z",
        )
        log = Path(row["log_path"])
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("Synthetic monitor benchmark: no trainer was launched.\n")
        runtime = Path(row["runtime_dir"])
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / "run_manifest.json").write_text(json.dumps({"benchmark": True}))
        checkpoint = Path(row["checkpoint_dir"])
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "synthetic.ckpt").write_text("Not a model checkpoint.\n")
    manifests.write_rows(root / "run_manifest.tsv", selected)
    state_path.write_text(json.dumps(state))
    return plan_dir, selected


def summarize_events(path):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if any(event["kind"] == "denied" for event in events):
        raise RuntimeError("A forbidden runtime or scheduler action was attempted")
    counts = Counter()
    for event in events:
        counts[event["kind"]] += 1
        if event["kind"] == "scheduler":
            counts[event["tool"]] += 1
            counts[f"{event['cluster']}:{event['tool']}"] += 1
    return dict(sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", default="1,6,12,50")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--modes", default="ordinary,health")
    parser.add_argument("--monitor", choices=("hparam", "experiment", "both"), default="both")
    parser.add_argument("--routes", type=int, choices=(1, 2), default=1)
    parser.add_argument("--direct-controller", action="store_true")
    parser.add_argument("--latency-ms", type=float, default=0)
    parser.add_argument("--scenario", choices=("queue", "controller", "batch-error"), default="queue")
    args = parser.parse_args()
    counts = [int(value) for value in args.runs.split(",")]
    modes = args.modes.split(",")
    if any(value <= 0 for value in counts) or args.samples <= 0 or args.latency_ms < 0:
        parser.error("Run counts and samples must be positive; latency must be nonnegative")
    if any(mode not in {"ordinary", "health"} for mode in modes):
        parser.error("Modes must be ordinary and/or health")
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    fakebin = output / "fakebin"
    make_fakebin(fakebin, interpreter=sys.executable, harness=Path(__file__).resolve())
    os.environ["PATH"] = str(fakebin) + os.pathsep + os.environ["PATH"]
    # Slurm clients run through `bash -lc`; re-export PATH after login profiles so the fakes stay first.
    bash_env = output / "bash-env.sh"
    bash_env.write_text(f"export PATH={shlex.quote(os.environ['PATH'])}\n")
    os.environ["BASH_ENV"] = str(bash_env)
    os.environ["PYTHONPATH"] = os.pathsep.join((str(repo), str(repo / "tests"), str(repo / "tests" / "agent_tools")))
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    results = []
    for count in counts:
        fixture = output / f"n{count}"
        fixture.mkdir()
        events = output / f"n{count}-events.jsonl"
        state = {
            "events": str(events),
            "latency_ms": args.latency_ms,
            "scenario": args.scenario,
            "routes": {
                f"benchmark-cluster-{index}": {
                    "direct_controller": args.direct_controller,
                    "jobs": {},
                }
                for index in range(args.routes)
            },
        }
        os.environ[STATE_ENV] = str(output / f"n{count}-state.json")
        plan_dir, rows = prepare_fixture(repo, fixture, count, args.routes, state)
        cases = [("hparam", mode) for mode in modes] if args.monitor != "experiment" else []
        if args.monitor != "hparam":
            cases.append(("experiment", "health"))
        for monitor, mode in cases:
            for sample in range(args.samples):
                events.write_text("")
                command = [sys.executable, "-m", "agent_tools", monitor + "-monitor", "--run-dir"]
                command.append(str(plan_dir if monitor == "hparam" else fixture))
                if monitor == "hparam":
                    command += ["--once"] + (["--health"] if mode == "health" else [])
                started = time.perf_counter()
                process = subprocess.run(command, cwd=repo, text=True, capture_output=True)
                elapsed = time.perf_counter() - started
                label = f"n{count}-{monitor}-{mode}-{sample}"
                (output / f"{label}.stdout").write_text(process.stdout)
                (output / f"{label}.stderr").write_text(process.stderr)
                (output / f"{label}.events.jsonl").write_bytes(events.read_bytes())
                if process.returncode:
                    raise RuntimeError(f"Public monitor failed ({label}): {process.stderr}")
                from agent_tools.experiment_workspace import read_run_manifest

                observed = read_run_manifest(fixture)
                identity_fields = ("step_id", "run_id", "target", "scheduler_job_id", "scheduler_cluster")
                if [tuple(row[field] for field in identity_fields) for row in observed] != [
                    tuple(row[field] for field in identity_fields) for row in rows
                ] or any(row["status"] != "queued" for row in observed):
                    raise RuntimeError("Public monitor changed synthetic frozen identity or expected queued status")
                result = {
                    "monitor": monitor,
                    "mode": mode,
                    "runs": count,
                    "routes": args.routes,
                    "sample": sample,
                    "wall_seconds": elapsed,
                    "counts": summarize_events(events),
                }
                results.append(result)
                print(json.dumps(result), flush=True)
    summary = []
    for key in dict.fromkeys((row["monitor"], row["mode"], row["runs"]) for row in results):
        samples = [row for row in results if (row["monitor"], row["mode"], row["runs"]) == key]
        summary.append(
            {
                "monitor": key[0],
                "mode": key[1],
                "runs": key[2],
                "median_wall_seconds": statistics.median(row["wall_seconds"] for row in samples),
                "counts_stable": all(row["counts"] == samples[0]["counts"] for row in samples),
                "counts": samples[0]["counts"],
            }
        )
    report = {
        "repo": str(repo),
        "commit": commit,
        "product_sha256": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                repo / "agent_tools" / name
                for name in (
                    "managed_scheduler.py",
                    "slurm.py",
                    "hparam_runtime.py",
                    "experiments.py",
                    "experiment_tracking.py",
                )
            )
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.executable,
        "transport": "real_subprocess_fake_slurm",
        "latency_ms": args.latency_ms,
        "scenario": args.scenario,
        "direct_controller": args.direct_controller,
        "fixture_and_preflight_timed": False,
        "cli_process_startup_timed": True,
        "samples": results,
        "summary": summary,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    selected_tool = os.environ.pop("SLURM_MONITOR_BENCH_TOOL", "")
    raise SystemExit(fake_tool(selected_tool) if selected_tool else main())
