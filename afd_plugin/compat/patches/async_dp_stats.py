# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Internal wire constants for asynchronous AFD DP load statistics."""

AFD_DPLB_STATS_KEY = "_afd_internal_dp_lb"
AFD_DPLB_STATS_VERSION = 1
PREFILL_TOKEN_DEBT_REPORT_INTERVAL_S = 0.1
PREFILL_TOKEN_DEBT_STALE_AFTER_MS = 500
PREFILL_TOKEN_DEBT_MAX_LIVE_REQUESTS = 4096

__all__ = [
    "AFD_DPLB_STATS_KEY",
    "AFD_DPLB_STATS_VERSION",
    "PREFILL_TOKEN_DEBT_MAX_LIVE_REQUESTS",
    "PREFILL_TOKEN_DEBT_REPORT_INTERVAL_S",
    "PREFILL_TOKEN_DEBT_STALE_AFTER_MS",
]
