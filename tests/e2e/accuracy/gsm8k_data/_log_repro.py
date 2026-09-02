#!/usr/bin/env python3
"""Repro: does an afd_plugin-namespaced INFO log survive vllm's logging setup?"""
import logging

import vllm  # noqa: F401  (triggers _configure_vllm_root_logger / dictConfig)
import afd_plugin  # noqa: F401  (applies the afd_plugin logger fix)
from vllm.logger import init_logger

logger = init_logger("afd_plugin.connectors.npu.async_cam")
root = logging.getLogger()
vllm_logger = logging.getLogger("vllm")

print(f"logger.disabled={logger.disabled}")
print(f"logger.level={logger.level} effective={logger.getEffectiveLevel()}")
print(f"root.level={root.level} root.handlers={root.handlers}")
print(f"vllm.handlers={vllm_logger.handlers} vllm.level={vllm_logger.level}")
print("--- emitting markers ---")
logger.info("INFO-MARKER-SHOULD-THIS-APPEAR")
logger.warning("WARN-MARKER-SHOULD-THIS-APPEAR")
print("--- done ---")
