# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Patches for AFD async-DP engine scheduling.

This module patches:
1. ``vllm.v1.engine.core.EngineCoreProc.run_engine_core``
2. ``vllm.v1.engine.utils.launch_core_engines``
3. ``vllm.v1.engine.utils.DPCoordinator`` construction during launch
4. ``vllm.v1.engine.coordinator.DPCoordinatorProc.run_coordinator``
5. ``vllm.v1.engine.coordinator.DPCoordinatorProc.process_input_socket``
6. ``vllm.v1.engine.core_client.DPAsyncMPClient.add_request_async``
7. ``vllm.v1.engine.core_client.DPLBAsyncMPClient.get_core_engine_for_request``
8. ``vllm.v1.engine.core_client.DPLBAsyncMPClient.process_engine_outputs``

Why:
    vLLM 0.26.0's native MoE DP path uses ``DPEngineCoreProc`` and DP wave
    notifications. AFD async-DP Attention ranks are connector-driven and must
    step independently while keeping the original DP/EP topology for expert
    placement and weight loading.

How:
    AFD async configs are selected by plugin-owned
    ``additional_config["afd"]["async"]``. Attention-side MoE DP engine
    processes instantiate ``EngineCoreProc`` instead of ``DPEngineCoreProc``;
    coordinator stats remain enabled, but wave coordination and client
    ``FIRST_REQ`` wakeups are disabled.

Future plan:
    Remove the coordinator portion when the pinned vLLM includes upstream
    #49204's non-lockstep stats handling. Remove the remaining patch when vLLM
    exposes an external async-DP scheduling hook selectable by plugin config.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import vllm.v1.engine.coordinator as coordinator_module
import vllm.v1.engine.core as engine_core_module
import vllm.v1.engine.core_client as core_client_module
import vllm.v1.engine.utils as engine_utils_module
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import DPAsyncMPClient, DPLBAsyncMPClient

from afd_plugin.compat.patches.async_dp_stats import (
    AFD_DPLB_STATS_KEY,
    AFD_DPLB_STATS_VERSION,
    PREFILL_TOKEN_DEBT_STALE_AFTER_MS,
)
from afd_plugin.compat.vllm import TARGET_VLLM_VERSION
from afd_plugin.config import is_afd_async_dp, parse_optional_afd_config

if TYPE_CHECKING:
    from multiprocessing.queues import Queue

    from vllm.config import VllmConfig
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.coordinator import DPCoordinator, DPCoordinatorProc
    from vllm.v1.engine.utils import (
        CoreEngineActorManager,
        CoreEngineProcManager,
        EngineZmqAddresses,
    )
    from vllm.v1.executor import Executor


_DPLB_STATS_VERSION_INDEX = 2
_DPLB_PREFILL_TOKEN_DEBT_INDEX = 3
_AFD_BLOCKED_REQUESTS_ATTR = "_afd_prefill_token_dplb_blocked_requests"


# Patch reason: vLLM's MoE DP engine process uses DPEngineCoreProc, but AFD
# async Attention ranks are connector-driven and must not run DP wave logic.
# Patch functionality: keep upstream startup flow while selecting EngineCoreProc
# for AFD async Attention configs.
# Signature: matches upstream; no added parameters.
def run_engine_core(
    *args,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
    **kwargs,
):
    """Replace MoE DP proc selection for AFD async Attention engines."""

    engine_core_module.maybe_register_config_serialize_by_value()

    engine_core = None
    signal_callback = None
    try:
        vllm_config = kwargs["vllm_config"]
        parallel_config = vllm_config.parallel_config
        data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
        if data_parallel:
            parallel_config.data_parallel_rank_local = local_dp_rank
            process_title = f"EngineCore_DP{dp_rank}"
        else:
            process_title = "EngineCore"
        engine_core_module.set_process_title(process_title)
        engine_core_module.maybe_init_worker_tracer(
            "vllm.engine_core",
            "engine_core",
            process_title,
        )
        engine_core_module.decorate_logs()
        if parallel_config.numa_bind:
            engine_core_module.numa_utils.log_current_affinity_state(process_title)

        if data_parallel and vllm_config.kv_transfer_config is not None:
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{dp_rank}"
            )
            engine_core_module.logger.debug(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        parallel_config.data_parallel_index = dp_rank
        if data_parallel and vllm_config.model_config.is_moe:
            parallel_config.data_parallel_rank = dp_rank
            # ### PATCH START: AFD async-DP Attention engine selection
            # Async-DP Attention ranks are connector-driven, so use the regular
            # EngineCoreProc instead of DPEngineCoreProc while keeping the
            # original DP rank metadata.
            if _is_afd_async_attention_config(vllm_config):
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
            else:
                engine_core = engine_core_module.DPEngineCoreProc(*args, **kwargs)
            # ### PATCH END: AFD async-DP Attention engine selection
        else:
            parallel_config.data_parallel_size = 1
            parallel_config.data_parallel_size_local = 1
            parallel_config.data_parallel_rank = 0
            engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

        assert engine_core is not None

        def wakeup_engine() -> None:
            # Wakes up idle engine via input_queue when shutdown is requested
            # Not safe in a signal handler - we may interrupt the main thread
            # while it is holding the non-reentrant input_queue.mutex
            engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

        signal_callback = engine_core_module.SignalCallback(wakeup_engine)

        def signal_handler(signum, frame):
            signal_name = engine_core_module.signal.Signals(signum).name
            engine_core_module.logger.info(
                "[shutdown] EngineCore: trigger received signal=%s",
                signal_name,
            )
            engine_core.shutdown_state = (
                engine_core_module.EngineShutdownState.REQUESTED
            )
            signal_callback.trigger()

        engine_core_module.signal.signal(
            engine_core_module.signal.SIGTERM,
            signal_handler,
        )
        engine_core_module.signal.signal(
            engine_core_module.signal.SIGINT,
            signal_handler,
        )

        engine_core.run_busy_loop()

    except SystemExit:
        engine_core_module.logger.info_once("[shutdown] EngineCore: exiting busy loop")
        raise
    except Exception as exc:
        if engine_core is None:
            engine_core_module.logger.exception("EngineCore failed to start.")
        else:
            engine_core_module.logger.exception("EngineCore encountered a fatal error.")
            engine_core._send_engine_dead()
        raise exc
    finally:
        engine_core_module.signal.signal(
            engine_core_module.signal.SIGTERM,
            engine_core_module.signal.SIG_DFL,
        )
        engine_core_module.signal.signal(
            engine_core_module.signal.SIGINT,
            engine_core_module.signal.SIG_DFL,
        )
        if signal_callback is not None:
            signal_callback.stop()
        if engine_core is not None:
            engine_core.shutdown()


# Patch reason: vLLM enables MoE DP wave coordination when launching DP cores,
# while AFD async-DP only needs coordinator stats.
# Patch functionality: preserve upstream engine launch behavior but disable wave
# coordination for AFD async-DP configs.
# Signature: matches upstream; no added parameters.
@contextmanager
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
    num_api_servers: int = 1,
) -> Iterator[
    tuple[
        CoreEngineProcManager | CoreEngineActorManager | None,
        DPCoordinator | None,
        EngineZmqAddresses,
        Queue | None,
    ]
]:
    """Disable coordinator wave mode while launching AFD async-DP engines."""

    parallel_config = vllm_config.parallel_config
    dp_size = parallel_config.data_parallel_size
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_rank = parallel_config.data_parallel_rank
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    offline_mode = local_start_index is not None

    tensor_queue: Queue | None = None
    multimodal_config = vllm_config.model_config.multimodal_config
    if multimodal_config is not None and multimodal_config.mm_tensor_ipc == "torch_shm":
        tensor_queue = engine_utils_module.get_mp_context().Queue()

    run_coordinator = (
        vllm_config.needs_dp_coordinator and not offline_mode and dp_rank == 0
    )

    if run_coordinator:
        coordinator = engine_utils_module.DPCoordinator(
            parallel_config,
            # ### PATCH START: AFD async-DP coordinator wave mode
            # Keep DP coordinator stats, but disable wave coordination for
            # connector-driven async-DP.
            enable_wave_coordination=(
                vllm_config.model_config.is_moe and not is_afd_async_dp(vllm_config)
            ),
            # ### PATCH END: AFD async-DP coordinator wave mode
        )

        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses()
        )
        addresses.frontend_stats_publish_address = (
            coordinator.get_stats_publish_address()
        )

        engine_utils_module.logger.info(
            "Started DP Coordinator process (PID: %d)",
            coordinator.proc.pid,
        )
    else:
        coordinator = None

    if parallel_config.data_parallel_backend == "ray":
        engine_utils_module.logger.info("Starting ray-based data parallel backend")

        engine_actor_manager = engine_utils_module.CoreEngineActorManager(
            vllm_config=vllm_config,
            addresses=addresses,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        yield engine_actor_manager, coordinator, addresses, tensor_queue
        return

    if offline_mode:
        assert local_engine_count == 1
        engines_to_handshake = [
            engine_utils_module.CoreEngine(index=dp_rank, local=True),
        ]
    elif dp_rank == 0:
        engines_to_handshake = [
            engine_utils_module.CoreEngine(index=i, local=(i < local_engine_count))
            for i in range(dp_size)
        ]
    else:
        assert local_engines_only, (
            "Attempting to launch core_engines from dp_rank > 0, but "
            "found internal DPLB, which is incompatible."
        )
        engines_to_handshake = [
            engine_utils_module.CoreEngine(index=i, local=True)
            for i in range(dp_rank, dp_rank + local_engine_count)
        ]

    handshake_local_only = offline_mode or local_engine_count == dp_size
    if parallel_config.enable_elastic_ep:
        handshake_local_only = False

    rpc_port = (
        parallel_config.data_parallel_rpc_port or engine_utils_module.get_open_port()
    )
    handshake_address = engine_utils_module.get_engine_client_zmq_addr(
        handshake_local_only,
        host,
        rpc_port,
    )

    if local_engines_only and dp_rank > 0:
        assert not handshake_local_only
        local_handshake_address = engine_utils_module.get_open_zmq_ipc_path()
        client_handshake_address = local_handshake_address
    else:
        local_handshake_address = handshake_address
        client_handshake_address = None

    with engine_utils_module.zmq_socket_ctx(
        local_handshake_address,
        engine_utils_module.zmq.ROUTER,
        bind=True,
    ) as handshake_socket:
        if local_engine_count:
            local_engine_manager = engine_utils_module.CoreEngineProcManager(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                client_handshake_address=client_handshake_address,
                local_client=True,
                local_engine_count=local_engine_count,
                start_index=dp_rank,
                local_start_index=local_start_index or 0,
                tensor_queue=tensor_queue,
            )
        else:
            local_engine_manager = None

        yield local_engine_manager, coordinator, addresses, tensor_queue

        engine_utils_module.wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            parallel_config,
            dp_size > 1 and vllm_config.model_config.is_moe,
            vllm_config.cache_config,
            local_engine_manager,
            coordinator.proc if coordinator else None,
        )


# Patch reason: multiprocessing spawn imports the coordinator target in a fresh
# process, where parent-only class assignments are not retained.
# Patch functionality: use a plugin-owned process target that constructs the
# pinned coordinator and invokes AFD's patched socket loop after child import.
# Signature: matches upstream; no added parameters.
def run_coordinator(
    engine_count: int,
    front_publish_address: str,
    back_output_address: str,
    back_publish_address: str,
    zmq_addr_pipe=None,
    min_stats_update_interval_ms: int = 100,
    enable_wave_coordination: bool = True,
):
    # ### PATCH START: AFD copied-coordinator module bindings
    DPCoordinatorProc = coordinator_module.DPCoordinatorProc
    logger = coordinator_module.logger
    # ### PATCH END: AFD copied-coordinator module bindings

    coordinator = DPCoordinatorProc(
        engine_count=engine_count,
        min_stats_update_interval_ms=min_stats_update_interval_ms,
        enable_wave_coordination=enable_wave_coordination,
    )
    try:
        # ### PATCH START: AFD spawn-safe coordinator loop
        process_input_socket(
            coordinator,
            front_publish_address,
            back_output_address,
            back_publish_address,
            zmq_addr_pipe,
        )
        # ### PATCH END: AFD spawn-safe coordinator loop
    except KeyboardInterrupt:
        logger.info("DP Coordinator process exiting")
    finally:
        if zmq_addr_pipe is not None:
            zmq_addr_pipe.close()


# Patch reason: vLLM 0.26.0 groups request-count updates by globally synchronized
# DP steps and has no token-work statistic, but AFD async Attention engines
# advance independently and may opt into prefill-token load balancing.
# Patch functionality: preserve upstream coordinator transport and publication
# cadence, skip lockstep snapshot behavior when wave coordination is disabled,
# and forward a versioned, stale-safe scalar token debt without coordinating
# engine execution.
# Signature: matches upstream; no added parameters.
def process_input_socket(
    self,
    front_publish_address: str,
    back_output_address: str,
    back_publish_address: str,
    zmq_addr_pipe=None,
):
    # ### PATCH START: AFD copied-coordinator module bindings
    time = coordinator_module.time
    msgspec = coordinator_module.msgspec
    zmq = coordinator_module.zmq
    logger = coordinator_module.logger
    make_zmq_socket = coordinator_module.make_zmq_socket
    MsgpackDecoder = coordinator_module.MsgpackDecoder
    EngineCoreOutputs = engine_core_module.EngineCoreOutputs
    EngineState = coordinator_module.EngineState
    # ### PATCH END: AFD copied-coordinator module bindings

    decoder = MsgpackDecoder(EngineCoreOutputs)

    # For tracking request wave progression.
    current_wave = 0
    engines_running = False

    # For tracking request counts for internal load-balancing.
    stats_changed = False
    last_stats_step = -1
    last_stats_wave = -1
    last_step_counts: list[list[int]] | None = None

    # ### PATCH START: AFD asynchronous prefill-token debt state
    # The optional token-sum policy extends each frontend count row with a
    # version and one scalar debt. This state is latest-value only and never
    # coordinates engine execution.
    prefill_token_debts: list[int | None] = [None for _ in self.engines]
    prefill_token_debt_update_ms = [0 for _ in self.engines]
    prefill_token_debt_seen = False
    # ### PATCH END: AFD asynchronous prefill-token debt state

    with (
        make_zmq_socket(
            path=front_publish_address,  # IPC
            ctx=self.ctx,
            socket_type=zmq.XPUB,
            bind=True,
        ) as publish_front,
        make_zmq_socket(
            path=back_output_address,  # IPC or TCP
            ctx=self.ctx,
            socket_type=zmq.PULL,
            bind=True,
        ) as output_back,
        make_zmq_socket(
            path=back_publish_address,  # IPC or TCP
            ctx=self.ctx,
            socket_type=zmq.XPUB,
            bind=True,
        ) as publish_back,
    ):
        if zmq_addr_pipe is not None:
            try:
                zmq_addr_pipe.send(
                    (
                        publish_front.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        output_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        publish_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                    )
                )
            finally:
                zmq_addr_pipe.close()
        # Wait until all engines subscribe.
        for _ in self.engines:
            if publish_back.recv() != b"\x01":
                logger.error(
                    "DP Coordinator received unexpected message while "
                    "waiting for engines to subscribe"
                )
                return
        # Send ready message to engines.
        publish_back.send(b"READY")

        logger.info("All engine subscriptions received by DP coordinator")

        poller = zmq.Poller()
        poller.register(publish_front, zmq.POLLIN)
        poller.register(publish_back, zmq.POLLIN)
        poller.register(output_back, zmq.POLLIN)
        last_publish_time = 0
        while True:
            # ### PATCH START: AFD monotonic stats/debt publication deadline
            now_ms = int(time.monotonic() * 1000)
            publish_deadline_ms = _get_stats_publish_deadline_ms(
                last_publish_time,
                self.stats_update_interval_ms,
                stats_changed,
                prefill_token_debt_update_ms if prefill_token_debt_seen else None,
            )
            wait_for = max(0, publish_deadline_ms - now_ms)
            # ### PATCH END: AFD monotonic stats/debt publication deadline

            # ### PATCH START: AFD independent request-count cadence
            # Only lockstep DP engines have a shared step whose rank-local
            # updates need a short aggregation window.
            min_timeout = (
                50 if self.enable_wave_coordination and last_step_counts is None else 0
            )
            # ### PATCH END: AFD independent request-count cadence

            events = poller.poll(timeout=max(min_timeout, wait_for))
            if not events:
                # Poller timeout - publish current stats to front-ends.
                # ### PATCH START: AFD prefill-token debt publication
                if prefill_token_debt_seen:
                    engine_req_counts_list = _get_augmented_engine_counts(
                        self,
                        prefill_token_debts,
                        prefill_token_debt_update_ms,
                        int(time.monotonic() * 1000),
                    )
                    stats_changed = False
                # ### PATCH END: AFD prefill-token debt publication
                elif last_step_counts is not None:
                    engine_req_counts_list = last_step_counts
                    last_step_counts = None
                else:
                    engine_req_counts_list = self._get_engine_counts()
                    stats_changed = False

                to_publish = (engine_req_counts_list, current_wave, engines_running)
                publish_front.send(msgspec.msgpack.encode(to_publish))
                # ### PATCH START: AFD monotonic debt expiry publication
                last_publish_time = int(time.monotonic() * 1000)
                if prefill_token_debt_seen:
                    _clear_expired_prefill_token_debt_updates(
                        prefill_token_debt_update_ms,
                        last_publish_time,
                    )
                # ### PATCH END: AFD monotonic debt expiry publication
                continue

            events = dict(events)
            wave_state_changed = False

            if publish_back in events:
                buffer = publish_back.recv()
                if buffer == b"\x01":
                    # NOTE(yongji): newly started engine subscribed
                    # We need to send READY message here instead of receiving
                    # SCALE_ELASTIC_EP notification from engine core client
                    # as SCALE_ELASTIC_EP is only sent when
                    # new engines finished initialization.
                    # Subscription message, on the other hand, is sent
                    # by each engine during initialization
                    publish_back.send(b"READY")
                elif buffer != b"\x00":
                    logger.error(
                        "DP Coordinator received unexpected message from engines"
                    )

            if publish_front in events:
                buffer = publish_front.recv()
                if buffer in (b"\x01", b"\x00"):
                    # Ignore subscription messages.
                    continue

                decoded = msgspec.msgpack.decode(buffer)
                if (
                    isinstance(decoded, (list, tuple))
                    and len(decoded) == 2
                    and decoded[0] == "SCALE_ELASTIC_EP"
                ):
                    # Handle scale up notification
                    new_engine_count = decoded[1]
                    current_count = len(self.engines)
                    if new_engine_count > current_count:
                        for _ in range(new_engine_count - current_count):
                            self.engines.append(EngineState())
                            # ### PATCH START: AFD token-debt scale alignment
                            prefill_token_debts.append(None)
                            prefill_token_debt_update_ms.append(0)
                            # ### PATCH END: AFD token-debt scale alignment
                        # NOTE(yongji): handle the case
                        # where newly started engines have current_wave = 0
                        # if existing engines just finished a wave
                        # and engine_running isn't updated yet at
                        # CoordinatorProc requests routed to newly started
                        # engines may not wake up existing engines, as long
                        # as 0 < request.wave < existing engines'
                        # current_wave
                        # we note that 0 is the wave number for the new
                        # engine
                        logger.info(
                            "DPCoordinator scaled up from %s to %s engines",
                            current_count,
                            new_engine_count,
                        )
                    else:
                        self.engines = self.engines[:new_engine_count]
                        # ### PATCH START: AFD token-debt scale alignment
                        prefill_token_debts = prefill_token_debts[:new_engine_count]
                        prefill_token_debt_update_ms = prefill_token_debt_update_ms[
                            :new_engine_count
                        ]
                        # ### PATCH END: AFD token-debt scale alignment
                        logger.info(
                            "DPCoordinator scaled down from %s to %s engines",
                            current_count,
                            new_engine_count,
                        )
                    continue  # Skip normal engine notification processing

                # Wave coordination: handle new-request messages from front-end.
                # Only process these when wave coordination is enabled
                if self.enable_wave_coordination:
                    # We received a message on the front-end XPUB socket,
                    # from an API server sending a new request while the
                    # engines are paused, so that we can wake the other
                    # engines.
                    engine_to_exclude, wave = decoded
                    if not engines_running:
                        if wave < current_wave:
                            # If the wave number is stale, ensure the message
                            # is handled by all the engines.
                            engine_to_exclude = None

                        engines_running = True
                        wave_state_changed = True
                        self._send_start_wave(
                            publish_back, current_wave, engine_to_exclude
                        )

            if output_back in events:
                # We received a message from one of the engines.

                buffer = output_back.recv()
                outputs: EngineCoreOutputs = decoder.decode(buffer)

                assert not outputs.outputs
                assert outputs.utility_output is None

                eng_index = outputs.engine_index
                scheduler_stats = outputs.scheduler_stats
                if scheduler_stats:
                    # 1. Updated request load stats - update our local
                    # state with these.
                    stats = self.engines[eng_index].request_counts
                    # ### PATCH START: AFD independent request-count updates
                    # Step/wave ordering is meaningful only for lockstep DP.
                    # Async Attention ranks update their latest counts directly.
                    if self.enable_wave_coordination:
                        stats_step = scheduler_stats.step_counter
                        stats_wave = scheduler_stats.current_wave
                        if (
                            stats_wave > last_stats_wave
                            or stats_wave == last_stats_wave
                            and stats_step > last_stats_step
                        ):
                            if stats_changed:
                                last_step_counts = self._get_engine_counts(do_copy=True)
                            last_stats_step = stats_step
                            last_stats_wave = stats_wave
                        elif stats_wave != last_stats_wave or (
                            stats_step != last_stats_step
                        ):
                            logger.warning(
                                "Received stats for out-of-order "
                                "step (%d, %d) from engine %d (expected "
                                "> (%d, %d))",
                                stats_wave,
                                stats_step,
                                eng_index,
                                last_stats_wave,
                                last_stats_step,
                            )
                    # ### PATCH END: AFD independent request-count updates
                    stats[0] = scheduler_stats.num_waiting_reqs
                    stats[1] = scheduler_stats.num_running_reqs
                    # ### PATCH START: AFD asynchronous prefill-token debt
                    connector_stats = scheduler_stats.kv_connector_stats
                    if (
                        isinstance(connector_stats, dict)
                        and AFD_DPLB_STATS_KEY in connector_stats
                    ):
                        token_stats = connector_stats[AFD_DPLB_STATS_KEY]
                        if (
                            isinstance(token_stats, dict)
                            and token_stats.get("version") == AFD_DPLB_STATS_VERSION
                        ):
                            token_debt = token_stats.get("prefill_token_debt")
                            if not (
                                token_debt is None
                                or isinstance(token_debt, int)
                                and not isinstance(token_debt, bool)
                                and token_debt >= 0
                            ):
                                token_debt = None
                            prefill_token_debts[eng_index] = token_debt
                            prefill_token_debt_update_ms[eng_index] = int(
                                time.monotonic() * 1000
                            )
                            if not prefill_token_debt_seen and last_publish_time == 0:
                                last_publish_time = prefill_token_debt_update_ms[
                                    eng_index
                                ]
                            prefill_token_debt_seen = True
                    # ### PATCH END: AFD asynchronous prefill-token debt
                    stats_changed = True

                # Wave coordination: handle wave completion and start notifications
                # Only process these when wave coordination is enabled
                if self.enable_wave_coordination:
                    if (wave := outputs.wave_complete) is not None:
                        # 2. Notification from rank 0 engine that we've
                        # moved into the global paused state
                        # (engines_running==False).
                        if current_wave <= wave:
                            new_wave = wave + 1
                            logger.debug(
                                "Moving DP wave from %d to %d.",
                                current_wave,
                                new_wave,
                            )
                            current_wave = new_wave
                            engines_running = False
                            wave_state_changed = True
                    elif (wave := outputs.start_wave) is not None and (
                        wave > current_wave
                        or (wave == current_wave and not engines_running)
                    ):
                        # 3. The engine received request for a non-current wave
                        # so we must ensure that other engines progress to the
                        # next wave (race condition handling).
                        logger.debug(
                            "Starting wave %d after notification of "
                            "stale wave request from engine.",
                            wave,
                        )
                        current_wave = wave
                        engines_running = True
                        wave_state_changed = True
                        self._send_start_wave(publish_back, wave, eng_index)

            if wave_state_changed:
                message = (None, current_wave, engines_running)
                publish_front.send(msgspec.msgpack.encode(message))

            # ### PATCH START: AFD non-starvable token-stats publication
            # A continuously readable engine socket must not postpone either
            # the 100 ms frontend update or the 500 ms debt expiry deadline.
            if prefill_token_debt_seen:
                now_ms = int(time.monotonic() * 1000)
                publish_deadline_ms = _get_stats_publish_deadline_ms(
                    last_publish_time,
                    self.stats_update_interval_ms,
                    stats_changed,
                    prefill_token_debt_update_ms,
                )
                if now_ms >= publish_deadline_ms:
                    engine_req_counts_list = _get_augmented_engine_counts(
                        self,
                        prefill_token_debts,
                        prefill_token_debt_update_ms,
                        now_ms,
                    )
                    to_publish = (
                        engine_req_counts_list,
                        current_wave,
                        engines_running,
                    )
                    publish_front.send(msgspec.msgpack.encode(to_publish))
                    last_publish_time = now_ms
                    stats_changed = False
                    _clear_expired_prefill_token_debt_updates(
                        prefill_token_debt_update_ms,
                        now_ms,
                    )
            # ### PATCH END: AFD non-starvable token-stats publication


def _get_stats_publish_deadline_ms(
    last_publish_time_ms: int,
    stats_update_interval_ms: int,
    stats_changed: bool,
    prefill_token_debt_update_ms: list[int] | None,
) -> int:
    publish_interval_ms = stats_update_interval_ms if stats_changed else 5000
    publish_deadline_ms = last_publish_time_ms + publish_interval_ms
    if prefill_token_debt_update_ms is not None:
        debt_expiry_deadlines = [
            update_ms + PREFILL_TOKEN_DEBT_STALE_AFTER_MS
            for update_ms in prefill_token_debt_update_ms
            if update_ms > 0
        ]
        if debt_expiry_deadlines:
            publish_deadline_ms = min(
                publish_deadline_ms,
                min(debt_expiry_deadlines),
            )
    return publish_deadline_ms


def _clear_expired_prefill_token_debt_updates(
    prefill_token_debt_update_ms: list[int],
    now_ms: int,
) -> None:
    for engine_index, update_ms in enumerate(prefill_token_debt_update_ms):
        if update_ms > 0 and now_ms - update_ms >= PREFILL_TOKEN_DEBT_STALE_AFTER_MS:
            prefill_token_debt_update_ms[engine_index] = 0


def _get_augmented_engine_counts(
    coordinator: DPCoordinatorProc,
    prefill_token_debts: list[int | None],
    prefill_token_debt_update_ms: list[int],
    now_ms: int,
) -> list[list[int | None]]:
    """Append versioned, stale-safe token debt to coordinator count rows."""

    augmented_counts: list[list[int | None]] = []
    for engine_index, counts in enumerate(coordinator._get_engine_counts()):
        waiting, running = counts
        token_debt = prefill_token_debts[engine_index]
        update_ms = prefill_token_debt_update_ms[engine_index]
        if update_ms == 0 or now_ms - update_ms >= PREFILL_TOKEN_DEBT_STALE_AFTER_MS:
            token_debt = 0 if waiting == 0 and running == 0 else None
        augmented_counts.append(
            [
                waiting,
                running,
                AFD_DPLB_STATS_VERSION,
                token_debt,
            ]
        )
    return augmented_counts


# Patch reason: vLLM's native DPLB score only sees request counts, which treats
# long and short prefill-only prompts as equal work.
# Patch functionality: for the opt-in AFD policy, select the least live prefill
# token debt while retaining count-score tie-breaking, rotating ties, optimistic
# local increments, explicit-rank routing, and count-only fallback.
# Signature: matches upstream; no added parameters.
def get_core_engine_for_request(
    self,
    request: EngineCoreRequest,
) -> core_client_module.EngineIdentity:
    # Engines are in rank order.
    if (eng_index := request.data_parallel_rank) is None and (
        eng_index := core_client_module.get_late_interaction_engine_index(
            request.pooling_params, len(self.core_engines)
        )
    ) is None:
        # ### PATCH START: AFD prefill-token-sum routing
        current_counts = self.lb_engines
        token_policy_enabled = _prefill_token_dplb_enabled(self.vllm_config)
        request_is_eligible = token_policy_enabled and (
            _request_supports_prefill_token_dplb(request)
        )
        blocked_requests: set[str] | None = None
        if token_policy_enabled:
            blocked_requests = _get_blocked_prefill_token_dplb_requests(self)
            if not request_is_eligible:
                blocked_requests.add(request.request_id)
                _invalidate_local_prefill_token_debts(current_counts)
        use_token_score = bool(request_is_eligible and not blocked_requests)

        if use_token_score:
            for counts in current_counts:
                if (
                    len(counts) <= _DPLB_PREFILL_TOKEN_DEBT_INDEX
                    or counts[_DPLB_STATS_VERSION_INDEX] != AFD_DPLB_STATS_VERSION
                    or not isinstance(counts[_DPLB_PREFILL_TOKEN_DEBT_INDEX], int)
                    or isinstance(counts[_DPLB_PREFILL_TOKEN_DEBT_INDEX], bool)
                    or counts[_DPLB_PREFILL_TOKEN_DEBT_INDEX] < 0
                ):
                    use_token_score = False
                    break

        # TODO use P2C alg for larger DP sizes
        num_engines = len(current_counts)
        min_score = (core_client_module.sys.maxsize, core_client_module.sys.maxsize)
        eng_index = 0
        for i in range(num_engines):
            # Start from client_index to help with balancing when engines
            # are empty.
            idx = (self.eng_start_index + i) % num_engines
            waiting = current_counts[idx][0]
            running = current_counts[idx][1]
            count_score = waiting * 4 + running
            if use_token_score:
                token_debt = current_counts[idx][_DPLB_PREFILL_TOKEN_DEBT_INDEX]
                score = (token_debt, count_score)
            else:
                score = (count_score, 0)
            if score < min_score:
                min_score = score
                eng_index = idx

        # Increment local load for better balancing between coordinator stats
        # updates (which happen every 100ms).
        current_counts[eng_index][0] += self.client_count
        if use_token_score:
            prompt_token_ids = request.prompt_token_ids
            assert prompt_token_ids is not None
            prompt_tokens = len(prompt_token_ids)
            current_counts[eng_index][_DPLB_PREFILL_TOKEN_DEBT_INDEX] += (
                prompt_tokens * self.client_count
            )
        # Rotate the scan start so equal scores do not systematically favor the
        # same engine after a coordinator reset.
        self.eng_start_index = (self.eng_start_index + 1) % num_engines
        core_client_module.logger.debug(
            "AFD Attention DPLB selected engine %d with policy=%s scores=%s",
            eng_index,
            "prefill_token_sum" if use_token_score else "request_count",
            current_counts,
        )
    elif _prefill_token_dplb_enabled(self.vllm_config):
        # Explicit or late-interaction routing bypasses the load score. Wait for
        # the request to finish and for a new global snapshot rather than
        # retaining a partial local debt.
        blocked_requests = _get_blocked_prefill_token_dplb_requests(self)
        blocked_requests.add(request.request_id)
        _invalidate_local_prefill_token_debts(self.lb_engines)
        # ### PATCH END: AFD prefill-token-sum routing

    chosen_engine = self.core_engines[eng_index]
    # Record which engine is chosen for this request, to handle aborts.
    self.reqs_in_flight[request.request_id] = chosen_engine
    return chosen_engine


def _prefill_token_dplb_enabled(vllm_config: VllmConfig) -> bool:
    afd_config = parse_optional_afd_config(vllm_config, validate=False)
    return bool(
        afd_config is not None
        and afd_config.role == "attention"
        and afd_config.attention_dplb_policy == "prefill_token_sum"
    )


def _request_supports_prefill_token_dplb(request: EngineCoreRequest) -> bool:
    sampling_params = request.sampling_params
    return bool(
        sampling_params is not None
        and request.pooling_params is None
        and sampling_params.max_tokens == 1
        and sampling_params.n == 1
        and sampling_params.structured_outputs is None
        and request.prompt_token_ids is not None
        and request.prompt_embeds is None
        and not request.mm_features
        and request.lora_request is None
        and request.priority == 0
        and not request.resumable
        and not request.abort_immediately
    )


def _invalidate_local_prefill_token_debts(
    current_counts: list[list[int]],
) -> None:
    for counts in current_counts:
        if len(counts) > _DPLB_PREFILL_TOKEN_DEBT_INDEX:
            counts[_DPLB_PREFILL_TOKEN_DEBT_INDEX] = -1


def _get_blocked_prefill_token_dplb_requests(
    client: DPLBAsyncMPClient,
) -> set[str]:
    try:
        return client.__dict__[_AFD_BLOCKED_REQUESTS_ATTR]
    except KeyError:
        blocked_requests: set[str] = set()
        client.__dict__[_AFD_BLOCKED_REQUESTS_ATTR] = blocked_requests
        return blocked_requests


# Patch reason: vLLM completion cleanup only releases abort-routing state, while
# AFD's token policy must also retain count fallback for locally observed mixed
# traffic until every ineligible or explicitly routed request finishes.
# Patch functionality: preserve native cleanup and clear plugin-owned fallback
# state on completion, invalidating local debt until a fresh snapshot arrives.
# Signature: matches upstream; no added parameters.
@staticmethod
async def process_engine_outputs(
    self: DPLBAsyncMPClient,
    outputs: engine_core_module.EngineCoreOutputs,
):
    if outputs.finished_requests and self.reqs_in_flight:
        # ### PATCH START: AFD mixed-workload fallback lifecycle
        token_policy_enabled = _prefill_token_dplb_enabled(self.vllm_config)
        blocked_requests = (
            _get_blocked_prefill_token_dplb_requests(self)
            if token_policy_enabled
            else None
        )
        had_blocked_requests = bool(blocked_requests)
        # ### PATCH END: AFD mixed-workload fallback lifecycle
        for req_id in outputs.finished_requests:
            self.reqs_in_flight.pop(req_id, None)
            # ### PATCH START: AFD mixed-workload fallback lifecycle
            if blocked_requests is not None:
                blocked_requests.discard(req_id)
            # ### PATCH END: AFD mixed-workload fallback lifecycle
        # ### PATCH START: AFD mixed-workload fallback lifecycle
        if had_blocked_requests and not blocked_requests:
            _invalidate_local_prefill_token_debts(self.lb_engines)
        # ### PATCH END: AFD mixed-workload fallback lifecycle


# Patch reason: vLLM sends FIRST_REQ wakeups to coordinate DP waves, which AFD
# async-DP engines intentionally do not use.
# Patch functionality: preserve request routing and stats updates while skipping
# FIRST_REQ for AFD async-DP configs.
# Signature: matches upstream; no added parameters.
async def add_request_async(
    self,
    request: EngineCoreRequest,
) -> None:
    """Skip the DP wave ``FIRST_REQ`` notification for AFD async-DP."""

    self._ensure_stats_update_task()

    request.current_wave = self.current_wave
    request.client_index = self.client_index

    chosen_engine = self.get_core_engine_for_request(request)
    to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
    # ### PATCH START: AFD async-DP request wakeup
    # Async-DP engines step independently, so skip the coordinator FIRST_REQ
    # wakeup while preserving normal routing.
    try:
        if not self.engines_running and not is_afd_async_dp(self.vllm_config):
            req_msg = core_client_module.msgspec.msgpack.encode(
                ("FIRST_REQ", chosen_engine),
            )
            await self.first_req_send_socket.send(req_msg)
        # ### PATCH END: AFD async-DP request wakeup

        await to_await
    except Exception:
        # ### PATCH START: AFD token-fallback send rollback
        if isinstance(self, DPLBAsyncMPClient) and _prefill_token_dplb_enabled(
            self.vllm_config
        ):
            _get_blocked_prefill_token_dplb_requests(self).discard(request.request_id)
        # ### PATCH END: AFD token-fallback send rollback
        raise

    # The output queue task delivers completed responses and is independent of
    # the DP-wave FIRST_REQ coordination skipped above.
    self._ensure_output_queue_task()


def _is_afd_async_attention_config(vllm_config: VllmConfig) -> bool:
    afd_config = parse_optional_afd_config(vllm_config, validate=False)
    return (
        afd_config is not None
        and is_afd_async_dp(vllm_config)
        and afd_config.role == "attention"
    )


def _should_patch_pinned_dp_coordinator() -> bool:
    """Return whether the copied coordinator exactly matches the installed vLLM."""

    try:
        import vllm

        version_text = str(vllm.__version__)
    except (AttributeError, ImportError):
        return False
    return version_text == TARGET_VLLM_VERSION


def _is_target_vllm_compatible() -> bool:
    try:
        import vllm

        version_value = vllm.__version__
    except (AttributeError, ImportError):
        return True
    version_text = str(version_value)
    if "dev" in version_text:
        return True
    return version_text.startswith(TARGET_VLLM_VERSION)


def apply_async_dp_engine_patch() -> bool:
    """Install the async-DP patches against the current vLLM bindings.

    vLLM-Ascend installs platform patches after general plugins and also wraps
    ``EngineCoreProc.run_engine_core``. Re-applying this installer after the
    final AFD Attention config is built ensures the process target captured by
    vLLM uses AFD's async-DP entry point. The caller scopes that late install to
    AFD async Attention, so non-AFD and FFN Ascend scheduling remain untouched.
    """

    if not _is_target_vllm_compatible():
        return False

    EngineCoreProc.run_engine_core = staticmethod(run_engine_core)
    engine_utils_module.launch_core_engines = launch_core_engines
    core_client_module.launch_core_engines = launch_core_engines
    # The copied coordinator is exact-release-only. Newer development versions
    # retain their native non-lockstep implementation and payload fields.
    if _should_patch_pinned_dp_coordinator():
        coordinator_module.DPCoordinatorProc.run_coordinator = staticmethod(
            run_coordinator
        )
        coordinator_module.DPCoordinatorProc.process_input_socket = process_input_socket
    DPAsyncMPClient.add_request_async = add_request_async
    DPLBAsyncMPClient.get_core_engine_for_request = get_core_engine_for_request
    DPLBAsyncMPClient.process_engine_outputs = process_engine_outputs
    engine_core_module.logger.debug("AFD async-DP engine patch applied")
    return True


apply_async_dp_engine_patch()


__all__ = ["apply_async_dp_engine_patch"]
