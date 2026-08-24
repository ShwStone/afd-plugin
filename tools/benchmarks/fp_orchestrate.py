# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Orchestrate the full-prefill performance experiment (32-card stages).

Drives the two experiment pods via ``itask exec`` from the local workstation:
server lifecycle (baseline DP4xTP8 EP32 / AFD 16A16F with A0/A1/A2 variants),
phase-zero data acceptance, fixed-batch mechanism runs, profiler replays, and
capacity replays of scaled Mooncake-derived arrival plans.

Plan: docs/npu/DEEPSEEK_V3_2_FULL_PREFILL_PERFORMANCE_PLAN.zh-CN.md
Dataset: tools/datasets/moonconv-wildchat-prefill (pinned revision, bundle
fe4f751b6dab). Sync the repo (including the dataset and any freshly
generated plan files) to the pods BEFORE running a phase: ``itask sync``.

Required environment:
  FP_NODE0 / FP_NODE1      itask pod names (node0 = master/attention)
  FP_NODE0_IP / FP_NODE1_IP pod IPs (same super-node)
Optional environment:
  FP_MBT                   common batch-token cap (default 65536)
  FP_RESULT_ROOT           results dir on the NAS share
                           (default <repo>/bench_results/full_prefill_performance)

Typical usage:
  python3 -m tools.benchmarks.fp_orchestrate --phase smoke --system baseline
  python3 -m tools.benchmarks.fp_orchestrate --phase accept --system baseline
  python3 -m tools.benchmarks.fp_orchestrate --phase accept --system afd --variant A2
  python3 -m tools.benchmarks.fp_orchestrate --phase fixed --system afd --variant A2
  python3 -m tools.benchmarks.fp_orchestrate --phase capacity --system baseline \
      --window screening --target-tokens-per-s 20000
  python3 -m tools.benchmarks.fp_orchestrate --phase capacity --system afd \
      --variant A2 --window formal_0 --target-tokens-per-s 40000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_LOCAL = Path(__file__).resolve().parents[2]
REPO_NAS = "/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin"
DEFAULT_RESULT_ROOT = f"{REPO_NAS}/bench_results/full_prefill_performance"

FIXED_BATCHES = ("fixed_8k_balanced", "fixed_32k_balanced", "fixed_32k_long_short")
# Plan section 9.1: which (system-role, batch) cells must run.
FIXED_MATRIX = {
    "baseline": ("fixed_8k_balanced", "fixed_32k_balanced", "fixed_32k_long_short"),
    "A0": ("fixed_8k_balanced", "fixed_32k_balanced"),
    "A1": ("fixed_32k_balanced", "fixed_32k_long_short"),
    "A2": ("fixed_8k_balanced", "fixed_32k_balanced", "fixed_32k_long_short"),
}

# Mooncake-derived WildChat prefill workload (pinned revision, bundle
# fe4f751b6dab). Paths are repo-relative; the repo is synced to the NAS.
MW_DIR = "tools/datasets/moonconv-wildchat-prefill"
MW_WORKLOADS = f"{MW_DIR}/workloads"
MW_PLANS = f"{MW_DIR}/plans"

SERVER_READY_TIMEOUT_S = 2400
SERVER_POLL_S = 15
KILL_PATTERN = "[/]vllm serve|[Vv][Ll][Ll][Mm]::|[V]LLMWorker|multiproc_[e]xecutor"


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"environment variable {name} is required")
    return value


def result_root() -> str:
    return os.environ.get("FP_RESULT_ROOT", DEFAULT_RESULT_ROOT)


def mbt() -> int:
    return int(os.environ.get("FP_MBT", "65536"))


def itask_exec(pod: str, command: str, *, retries: int = 2, timeout: int = 600) -> str:
    """Run a bash command in a pod; itask exec runs no shell itself."""
    full = ["itask", "exec", pod, "--tty=false", "--", "bash", "-c", command]
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(retries + 1):
        try:
            completed = subprocess.run(
                full,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return completed.stdout
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt < retries:
                time.sleep(10)
    assert last_error is not None
    raise RuntimeError(
        f"itask exec failed on {pod}: {command[:200]}\n"
        f"stdout: {last_error.stdout}\nstderr: {last_error.stderr}"
    )


def kill_all() -> None:
    # SIGKILL on purpose: crashed servers leave VLLM::Worker zombies holding
    # HBM and HCCL resources that SIGTERM does not reap, and the next boot
    # then dies with device-side timeouts (507014/507034).
    for pod in (env("FP_NODE0"), env("FP_NODE1")):
        itask_exec(
            pod,
            f"pkill -9 -f '{KILL_PATTERN}' || true; sleep 3; "
            f"pgrep -af '{KILL_PATTERN}' | head -5 || true",
        )
    print("[kill_all] done", flush=True)


def _launch(pod: str, log_path: str, envs: dict[str, str], script: str) -> None:
    """Launch one server in the pod.

    The itask exec channel intermittently hangs on background-launch commands
    even though the server starts fine. Never retry a launch (a retry would
    spawn a duplicate server): wait briefly, then verify the process exists
    with a separate short exec, and abandon the hung channel.
    """
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in envs.items())
    command = (
        f"cd {REPO_NAS} && mkdir -p $(dirname {log_path}) && "
        f"mkdir -p {shlex.quote(envs.get('PROMETHEUS_MULTIPROC_DIR', '/tmp'))} && "
        f"setsid env {env_prefix} bash {script} > {log_path} 2>&1 < /dev/null & "
        "echo LAUNCHED"
    )
    process = subprocess.Popen(
        ["itask", "exec", pod, "--tty=false", "--", "bash", "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        process.communicate(timeout=30)
        if process.returncode != 0:
            raise RuntimeError(
                f"launch on {pod} exited {process.returncode}: "
                f"{process.stderr}"
            )
        return
    except subprocess.TimeoutExpired:
        pass
    # Channel hung: check whether the server actually came up. The remote
    # side can be slow to even spawn the process when the channel hangs
    # (observed: vllm's first log line 45 s after launch), so probe twice
    # before declaring failure. Still never re-issue the launch itself.
    count_text = "0"
    for _ in range(2):
        time.sleep(5 if count_text == "0" else 0)
        probe = subprocess.run(
            [
                "itask", "exec", pod, "--tty=false", "--", "bash", "-c",
                "pgrep -cf '[v]llm serve' || true",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        count_text = (
            probe.stdout.strip().splitlines()[-1] if probe.stdout.strip() else "0"
        )
        if probe.returncode == 0 and count_text.isdigit() and int(count_text) > 0:
            break
        time.sleep(25)
    process.kill()
    process.communicate()
    if count_text.isdigit() and int(count_text) > 0:
        print(
            f"[_launch] exec channel hung but server is up on {pod}; continuing",
            flush=True,
        )
        return
    raise RuntimeError(f"launch on {pod} failed: no vllm serve process found")


def start_server(system: str, variant: str | None) -> str:
    """Start one experiment system; returns the server label."""
    node0, node1 = env("FP_NODE0"), env("FP_NODE1")
    node0_ip = env("FP_NODE0_IP")
    label = system if variant is None else f"{system}_{variant.lower()}"
    log_dir = f"{result_root()}/logs"
    profile_envs = _profile_envs(system, variant)
    prometheus_dir = f"{result_root()}/logs/prometheus/{label}"
    if system == "baseline":
        common = {
            "MAX_NUM_BATCHED_TOKENS": str(mbt()),
            "DP_ADDRESS": node0_ip,
            "PROMETHEUS_MULTIPROC_DIR": prometheus_dir,
            **profile_envs.get("baseline", {}),
        }
        _launch(
            node1,
            f"{log_dir}/{label}_node1.log",
            {**common, "DP_START_RANK": "2"},
            "tools/benchmarks/fp_launch_baseline_dp4tp8.sh",
        )
        time.sleep(10)
        _launch(
            node0,
            f"{log_dir}/{label}_node0.log",
            {**common, "DP_START_RANK": "0"},
            "tools/benchmarks/fp_launch_baseline_dp4tp8.sh",
        )
    elif system == "afd":
        assert variant in ("A0", "A1", "A2"), f"bad variant {variant}"
        common = {
            "MAX_NUM_BATCHED_TOKENS": str(mbt()),
            "AFD_HOST": node0_ip,
            "AFD_VARIANT": variant,
            "PROMETHEUS_MULTIPROC_DIR": prometheus_dir,
        }
        # AFD has no boot-order requirement: the CAM connector rendezvous
        # handles whichever side comes up first. Both sides are launched
        # back-to-back and will discover each other once their workers are
        # in the recv loop.
        _launch(
            node1,
            f"{log_dir}/{label}_node1_ffn.log",
            {**common, **profile_envs.get("ffn", {})},
            "tools/benchmarks/fp_launch_afd_ffn.sh",
        )
        _launch(
            node0,
            f"{log_dir}/{label}_node0_attn.log",
            {**common, **profile_envs.get("attention", {})},
            "tools/benchmarks/fp_launch_afd_attention.sh",
        )
    else:
        raise SystemExit(f"unknown system {system}")
    print(f"[start_server] {label} launching", flush=True)
    return label


def _profile_envs(
    system: str, variant: str | None
) -> dict[str, dict[str, str]]:
    """Profiler env per role when FP_PROFILE=1 (diagnostic runs only)."""
    if os.environ.get("FP_PROFILE") != "1":
        return {}
    trace_root = f"{result_root()}/02_profiles_32/traces"
    window = {
        "WAIT": "0",
        "WARMUP": "1",
        "ACTIVE": os.environ.get("FP_PROFILE_ACTIVE", "5"),
        "SKIP_FIRST": os.environ.get("FP_PROFILE_SKIP_FIRST", "10"),
        "REPEAT": "1",
    }
    # Correlation tracing writes cross-role timeline sidecars alongside
    # the torch_npu traces so the merge tool can align attention dispatch
    # and FFN compute events.
    label = system if variant is None else f"{system}_{variant.lower()}"
    corr_dir = f"{trace_root}/correlation/{label}"
    corr_id = hashlib.sha256(
        f"{label}_{time.time()}".encode()
    ).hexdigest()[:16]
    correlation = {
        "AFD_TRACE_ENABLE": "1",
        "AFD_TRACE_SESSION_ID": corr_id,
        "AFD_TRACE_DIR": corr_dir,
    }
    return {
        "baseline": {"VLLM_TORCH_PROFILER_DIR": f"{trace_root}/baseline"},
        "attention": {
            f"AFD_NPU_ATTENTION_PROFILER_{key}": value
            for key, value in window.items()
        }
        | {
            "AFD_NPU_ATTENTION_PROFILER_ENABLE": "1",
            "AFD_NPU_ATTENTION_PROFILER_DIR": f"{trace_root}/attention",
            **correlation,
        },
        "ffn": {
            f"AFD_NPU_FFN_PROFILER_{key}": value for key, value in window.items()
        }
        | {
            "AFD_NPU_FFN_PROFILER_ENABLE": "1",
            "AFD_NPU_FFN_PROFILER_DIR": f"{trace_root}/ffn",
            **correlation,
        },
    }


def _wait_log_marker(
    pod: str,
    log_path: str,
    marker: str,
    *,
    timeout_s: int,
) -> None:
    """Poll a pod log file for a readiness marker line."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        probe = subprocess.run(
            [
                "itask", "exec", pod, "--tty=false", "--", "bash", "-c",
                f"grep -c {shlex.quote(marker)} {log_path} 2>/dev/null || true",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if probe.returncode == 0 and probe.stdout.strip().isdigit():
            if int(probe.stdout.strip()) > 0:
                print(f"[_wait_log_marker] {marker!r} seen on {pod}", flush=True)
                return
        time.sleep(30)
    raise TimeoutError(f"marker {marker!r} not seen in {log_path} on {pod}")


def wait_server(timeout_s: int = SERVER_READY_TIMEOUT_S) -> None:
    node0 = env("FP_NODE0")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        probe = subprocess.run(
            [
                "itask", "exec", node0, "--tty=false", "--", "bash", "-c",
                "curl -sf http://127.0.0.1:8000/v1/models",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if probe.returncode == 0 and "deepseek_v3_2" in probe.stdout:
            print("[wait_server] ready", flush=True)
            return
        time.sleep(SERVER_POLL_S)
    raise TimeoutError(f"server not ready within {timeout_s}s")


def reset_prefix_cache() -> None:
    node0 = env("FP_NODE0")
    itask_exec(
        node0,
        "curl -s -X POST 'http://127.0.0.1:8000/reset_prefix_cache"
        "?reset_running_requests=true&reset_external=true' || true",
    )


def run_client_on_node0(command: str, *, timeout: int = 7200) -> None:
    """Run a benchmark client command on node0 with the repo env."""
    output = itask_exec(
        env("FP_NODE0"),
        f"cd {REPO_NAS} && {command}",
        retries=0,
        timeout=timeout,
    )
    print(output, flush=True)


def phase_smoke(system: str, variant: str | None) -> None:
    """Quick liveness: one short + one longest warmup request per topology."""
    kill_all()
    start_server(system, variant)
    wait_server()
    run_client_on_node0(
        "python3 -m tools.benchmarks.fp_smoke_client --mode shortest",
    )
    print("[smoke] short request OK; longest-request check next", flush=True)
    run_client_on_node0(
        "python3 -m tools.benchmarks.fp_smoke_client --mode longest",
        timeout=2400,
    )


def _check_boot_config(label: str, system: str) -> None:
    """Plan section 8.1: record the frozen-config evidence from the boot log.

    Fails fast if MC2 prefill, EPLB, or prefix caching show up enabled.
    """
    suffix = "node0.log" if system == "baseline" else "node0_attn.log"
    log_path = f"{result_root()}/logs/{label}_{suffix}"
    out = itask_exec(
        env("FP_NODE0"),
        f"grep -o \"'enable_prefill_mc2': [A-Za-z]*\" {log_path} | head -1; "
        f"grep -o \"'enable_eplb': [A-Za-z]*\" {log_path} | head -1; "
        f"grep -o \"'enable_prefix_caching': [A-Za-z]*\" {log_path} | head -1; "
        f"grep -o \"'max_num_batched_tokens': [0-9]*\" {log_path} | head -1; "
        f"grep -o \"'gpu_memory_utilization': [0-9.]*\" {log_path} | head -1",
    )
    print(f"[accept] boot config evidence:\n{out}", flush=True)
    bad = []
    for line in out.splitlines():
        if "'enable_prefill_mc2': True" in line:
            bad.append("enable_prefill_mc2")
        if "'enable_eplb': True" in line:
            bad.append("enable_eplb")
        if "'enable_prefix_caching': True" in line:
            bad.append("enable_prefix_caching")
    if bad:
        raise SystemExit(f"[accept] FORBIDDEN config enabled: {bad}")


def _launch_detached(pod: str, command: str) -> None:
    """Fire a background command in a pod, tolerating the hung-channel quirk.

    itask exec intermittently never returns for background launches even
    though the command runs. Give it 20 s, then abandon the channel; callers
    verify via the command's own output files.
    """
    process = subprocess.Popen(
        ["itask", "exec", pod, "--tty=false", "--", "bash", "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        process.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        print(f"[_launch_detached] channel hung on {pod}; continuing", flush=True)


def _run_mem_sampled_single(request_id: str, out_dir: str) -> None:
    """Replay one long request while sampling per-card HBM on both nodes."""
    node0, node1 = env("FP_NODE0"), env("FP_NODE1")
    for pod, tag in ((node0, "node0"), (node1, "node1")):
        _launch_detached(
            pod,
            f"setsid bash {REPO_NAS}/tools/benchmarks/fp_npu_mem_sample.sh "
            f"{result_root()}/logs/mem_{request_id}_{tag}.tsv 240 2 "
            f">/dev/null 2>&1 < /dev/null & echo SAMPLER_UP",
        )
    time.sleep(3)  # let both samplers write their first sample
    run_client_on_node0(
        "python3 -m tools.benchmarks.mw_replay_client "
        f"--requests {MW_WORKLOADS}/accept_long_singles.jsonl "
        f"--request-ids {request_id} "
        f"--output {out_dir}/accept_single_{request_id}.json",
        timeout=2400,
    )
    time.sleep(5)
    for pod, tag in ((node0, "node0"), (node1, "node1")):
        peak = itask_exec(
            pod,
            f"awk -F'\\t' '{{if ($3 > m[$2]) m[$2] = $3}} "
            f"END {{for (d in m) print d, m[d]}}' "
            f"{result_root()}/logs/mem_{request_id}_{tag}.tsv | sort -n",
        )
        print(f"[accept] peak HBM MB ({tag}, {request_id}):\n{peak}", flush=True)


def phase_accept(system: str, variant: str | None) -> None:
    """Plan section 8.2: data and long-input service acceptance.

    1. 8 warmup requests (short/mid/long) with prompt-token verification;
    2. full 32-request warmup burst, all must succeed with 1 output token;
    3. single-request replays near 32K / 52K / 63,778 with per-card peak HBM;
    4. baseline only: one ~52K request per DP replica concurrently (EP32 MoE
       workspace check with all four data replicas active);
    5. request IDs / lengths in results correspond one-to-one with the files
       (guaranteed by the replay client's per-request records).
    """
    label = system if variant is None else f"{system}_{variant.lower()}"
    out_dir = f"{result_root()}/00_accept/{label}"
    kill_all()
    start_server(system, variant)
    wait_server()
    _check_boot_config(label, system)

    print("[accept] step 1: 8-request token-ID service check", flush=True)
    run_client_on_node0(
        "python3 -m tools.benchmarks.mw_replay_client "
        f"--requests {MW_WORKLOADS}/accept_8_warmup.jsonl "
        f"--output {out_dir}/accept_8_warmup.json",
        timeout=3600,
    )
    print("[accept] step 2: full 32-request warmup burst", flush=True)
    run_client_on_node0(
        "python3 -m tools.benchmarks.mw_replay_client "
        f"--requests {MW_WORKLOADS}/warmup_requests.jsonl "
        f"--output {out_dir}/accept_warmup32.json",
        timeout=3600,
    )
    print("[accept] step 3: 32K / 52K / 63,778 single replays", flush=True)
    singles_path = REPO_LOCAL / MW_WORKLOADS / "accept_long_singles.jsonl"
    single_ids = [
        json.loads(line)["request_id"]
        for line in singles_path.read_text(encoding="utf-8").splitlines()
    ]
    for request_id in single_ids:
        _run_mem_sampled_single(request_id, out_dir)
    if system == "baseline":
        print("[accept] step 4: 4x~52K concurrent, one per DP replica", flush=True)
        run_client_on_node0(
            "python3 -m tools.benchmarks.mw_replay_client "
            f"--requests {MW_WORKLOADS}/accept_4x52k.jsonl "
            f"--output {out_dir}/accept_4x52k.json",
            timeout=3600,
        )
    print(f"[accept] {label} done; results under {out_dir}", flush=True)


def phase_fixed(system: str, variant: str | None, dp_rank: int) -> None:
    """Plan section 9.1: clean fixed-batch matrix for one system/variant."""
    key = "baseline" if system == "baseline" else (variant or "A2")
    batches = FIXED_MATRIX[key]
    label = system if variant is None else f"{system}_{variant.lower()}"
    out_dir = f"{result_root()}/01_fixed_batch/{label}"
    kill_all()
    start_server(system, variant)
    wait_server()
    for batch in batches:
        output = f"{out_dir}/{batch}.fixed_batch.json"
        run_client_on_node0(
            "python3 -m tools.benchmarks.fixed_batch_client "
            "--base-url http://127.0.0.1:8000 "
            f"--dataset {MW_WORKLOADS}/{batch}.jsonl "
            f"--dp-rank {dp_rank} --repeats 10 --warmups 1 "
            f"--output {output}",
            timeout=7200,
        )
    print(f"[fixed] {label} done", flush=True)


def phase_profile(
    system: str,
    variant: str | None,
    dp_rank: int,
    batch: str,
    precollect_batch: str | None = None,
) -> None:
    """Plan sections 8.3/9.2: bounded profiler replay of one fixed batch.

    Requires FP_PROFILE=1 so servers boot with profiler switches. Baseline
    uses the upstream /start_profile API; AFD roles use the plugin-owned
    env-scheduled profiler (see afd_plugin/compat/npu/profiler.py).

    ``precollect_batch`` re-runs one fixed batch un-instrumented right after
    boot (before profiling starts) — used to re-collect keepalive-tainted
    cells on a server that is booting anyway.
    """
    if os.environ.get("FP_PROFILE") != "1":
        raise SystemExit("phase=profile requires FP_PROFILE=1")
    label = system if variant is None else f"{system}_{variant.lower()}"
    out_dir = f"{result_root()}/02_profiles_32/{label}"
    kill_all()
    start_server(system, variant)
    wait_server()
    if precollect_batch:
        fixed_out = f"{result_root()}/01_fixed_batch/{label}"
        run_client_on_node0(
            "python3 -m tools.benchmarks.fixed_batch_client "
            "--base-url http://127.0.0.1:8000 "
            f"--dataset {MW_WORKLOADS}/{precollect_batch}.jsonl "
            f"--dp-rank {dp_rank} --repeats 10 --warmups 1 "
            f"--output {fixed_out}/{precollect_batch}.fixed_batch.json",
            timeout=7200,
        )
        print(f"[profile] pre-collected clean {precollect_batch}", flush=True)
    if system == "baseline":
        itask_exec(
            env("FP_NODE0"),
            "curl -s -X POST http://127.0.0.1:8000/start_profile || true",
        )
        time.sleep(5)
    run_client_on_node0(
        "python3 -m tools.benchmarks.fixed_batch_client "
        "--base-url http://127.0.0.1:8000 "
        f"--dataset {MW_WORKLOADS}/{batch}.jsonl "
        f"--dp-rank {dp_rank} --repeats 8 --warmups 2 "
        f"--output {out_dir}/{batch}.profile_replay.json",
        timeout=7200,
    )
    if system == "baseline":
        itask_exec(
            env("FP_NODE0"),
            "curl -s -X POST http://127.0.0.1:8000/stop_profile || true",
        )
    # Graceful shutdown flushes the correlation sidecar JSONL files
    # before the next phase's kill_all(9) discards in-memory events.
    _graceful_shutdown()
    # Traces flush asynchronously; give the profiler time to write.
    time.sleep(60)
    print(f"[profile] {label} replay done; traces under "
          f"{result_root()}/02_profiles_32/traces", flush=True)


def _graceful_shutdown() -> None:
    """SIGTERM the AFD servers and wait for them to call connector.close().

    The correlation trace recorder buffers events in memory and only
    flushes them to the JSONL sidecar during ``CAMAsyncAFDConnector.close()``
    (→ ``trace_recorder.close()``). pkill -9 skips that path entirely, so
    correlation files are lost on every hard-kill. Send SIGTERM first, let
    the workers run ``shutdown()`` → ``connector.close()``, then clean up
    any stragglers with SIGKILL.
    """
    node0, node1 = env("FP_NODE0"), env("FP_NODE1")
    for pod in (node0, node1):
        # 1. SIGTERM via pkill (no braces in the pattern to avoid
        #    bash -c quoting issues inside itask exec).
        itask_exec(
            pod,
            "pkill -f vllm serve; echo SIGNALED",
            retries=0,
        )
    time.sleep(15)
    # 2. SIGKILL any remaining stragglers.
    for pod in (node0, node1):
        itask_exec(
            pod,
            "pkill -9 -f '[/]vllm serve|[Vv][Ll][Ll][Mm]::|[V]LLMWorker|multiproc_[e]xecutor' || true",
            retries=0,
        )
    print("[shutdown] graceful pkill + 15s drain + SIGKILL clean", flush=True)


def _scaled_plan_path(window: str, target_tokens_per_s: str) -> Path:
    """Local path of the scaled arrival plan for one experiment point."""
    safe_rate = target_tokens_per_s.replace(".", "p")
    return (
        REPO_LOCAL / MW_PLANS / f"{window}_{safe_rate}tps.json"
    )


def ensure_scaled_plan(window: str, target_tokens_per_s: str) -> Path:
    """Generate the scaled plan locally if missing (deterministic output).

    Both systems MUST replay the same plan file, so plans are generated once
    into the synced dataset directory and reused across systems.
    """
    plan_path = _scaled_plan_path(window, target_tokens_per_s)
    if not plan_path.exists():
        subprocess.run(
            [
                sys.executable,
                str(REPO_LOCAL / "tools/benchmarks/mw_scale_arrivals.py"),
                "--dataset-dir",
                str(REPO_LOCAL / MW_DIR),
                "--window",
                window,
                "--target-tokens-per-second",
                target_tokens_per_s,
                "--output",
                str(plan_path),
            ],
            check=True,
        )
        print(
            f"[capacity] generated {plan_path} — run `itask sync` before "
            "the replay if the pods do not see it",
            flush=True,
        )
    return plan_path


def phase_capacity(
    system: str,
    variant: str | None,
    window: str,
    target_tokens_per_s: str,
    no_boot: bool = False,
) -> None:
    """Plan section 10: replay one scaled arrival plan for one window.

    Runs the 32-request warmup burst first (not part of the metrics), then
    replays the window open-loop against the frozen scaled plan. The replay
    client enforces the send-deviation gate (p99 <= 100 ms, max <= 250 ms);
    the plan file is shared verbatim between the two systems.

    --no-boot skips kill/boot/wait and reuses the live server — screening
    runs several rate points per boot (the server drains between runs; the
    prefix cache is off, so there is no cross-point state).
    """
    if window not in ("screening", "formal_0", "formal_1", "formal_2"):
        raise SystemExit(f"bad window {window}")
    plan_path = ensure_scaled_plan(window, target_tokens_per_s)
    plan_rel = plan_path.relative_to(REPO_LOCAL)
    plan_doc = json.loads(plan_path.read_text(encoding="utf-8"))

    label = system if variant is None else f"{system}_{variant.lower()}"
    safe_rate = target_tokens_per_s.replace(".", "p")
    run_dir = f"{result_root()}/03_capacity_32/{window}/{label}/{safe_rate}tps"
    if not no_boot:
        kill_all()
        start_server(system, variant)
        wait_server()
    warmup_out = f"{run_dir}/warmup32.json"
    run_client_on_node0(
        "python3 -m tools.benchmarks.mw_replay_client "
        f"--requests {MW_WORKLOADS}/warmup_requests.jsonl "
        f"--output {warmup_out}",
        timeout=7200,
    )
    # Gate the measured replay on warmup health: a half-dead AFD pair (FFN
    # self-crashed while idle) still returns HTTP 200 instantly with garbage
    # and no usage field — 2026-08-20 afd_a2/2185p5tps measured exactly that.
    # Never run a measured replay against a server that cannot prove it
    # prefilled the warmup requests.
    warmup_doc = json.loads(
        itask_exec(env("FP_NODE0"), f"cat {warmup_out}")
    )
    wsum = warmup_doc["summary"]
    if (
        wsum["failed"] != 0
        or wsum["prompt_tokens_mismatched"] != 0
        or wsum["successful"] != wsum["request_count"]
    ):
        raise SystemExit(
            f"[capacity] warmup UNHEALTHY for {label}: {wsum}; "
            "aborting before the measured replay"
        )
    run_client_on_node0(
        "python3 -m tools.benchmarks.mw_replay_client "
        f"--requests {MW_WORKLOADS}/{window}_requests.jsonl "
        f"--plan {plan_rel} "
        f"--output {run_dir}/replay.json",
        timeout=86400,
    )
    print(
        f"[capacity] {label} window={window} "
        f"target={plan_doc['target_input_tokens_per_second']} tok/s "
        f"d={plan_doc['dilation_factor']} done",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("smoke", "accept", "fixed", "profile", "capacity"),
    )
    parser.add_argument("--system", required=True, choices=("baseline", "afd"))
    parser.add_argument("--variant", choices=("A0", "A1", "A2"), default=None)
    parser.add_argument("--dp-rank", type=int, default=0)
    parser.add_argument("--batch", default="fixed_32k_balanced")
    parser.add_argument("--window", default="screening")
    parser.add_argument("--target-tokens-per-s", default=None)
    parser.add_argument(
        "--precollect-batch",
        default=None,
        help="profile phase only: re-run this fixed batch un-instrumented "
        "right after boot, before the profiler replay",
    )
    parser.add_argument(
        "--no-boot",
        action="store_true",
        help="capacity phase only: reuse the live server (screening runs "
        "several rate points per boot)",
    )
    args = parser.parse_args()

    if args.system == "baseline" and args.variant is not None:
        raise SystemExit("--variant only applies to --system afd")
    if args.system == "afd" and args.variant is None:
        raise SystemExit("--system afd requires --variant A0|A1|A2")

    if args.phase == "smoke":
        phase_smoke(args.system, args.variant)
    elif args.phase == "accept":
        phase_accept(args.system, args.variant)
    elif args.phase == "fixed":
        phase_fixed(args.system, args.variant, args.dp_rank)
    elif args.phase == "profile":
        phase_profile(
            args.system,
            args.variant,
            args.dp_rank,
            args.batch,
            args.precollect_batch,
        )
    elif args.phase == "capacity":
        if not args.target_tokens_per_s:
            raise SystemExit("--phase capacity requires --target-tokens-per-s")
        phase_capacity(
            args.system,
            args.variant,
            args.window,
            args.target_tokens_per_s,
            args.no_boot,
        )


if __name__ == "__main__":
    main()
